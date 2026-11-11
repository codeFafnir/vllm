# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import heapq
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum

from vllm.v1.request import Request


class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""

    FCFS = "fcfs"
    PRIORITY = "priority"
    CONTINUUM = "continuum"


class RequestQueue(ABC):
    """Abstract base class for request queues."""

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to the policy."""
        pass

    @abstractmethod
    def pop_request(self) -> Request:
        """Pop a request from the queue according to the policy."""
        pass

    @abstractmethod
    def peek_request(self) -> Request:
        """Peek at the request at the front of the queue without removing it."""
        pass

    @abstractmethod
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        pass

    @abstractmethod
    def prepend_requests(self, requests: Iterable[Request]) -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get number of requests in queue."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to the policy."""
        pass


class FCFSRequestQueue(deque[Request], RequestQueue):
    """A first-come-first-served queue that supports deque operations."""

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)

    def prepend_requests(self, requests: Iterable[Request]) -> None:
        """Prepend all requests from another queue to the front of this
        queue.

        Note: The requests will be prepended in reverse order of their
        appearance in the `requests` queue.
        """
        self.extendleft(requests)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        filtered_requests = [req for req in self if req not in requests_to_remove]
        # deque does not support in-place filtering, so we need to clear
        # and extend
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()


class PriorityRequestQueue(RequestQueue):
    """A priority queue that supports heap operations.

    Respects the ordering defined in the Request class, where
    requests with a smaller value of `priority` are processed first.
    If multiple requests have the same priority, the one with the earlier
    `arrival_time` is processed first.
    """

    def __init__(self) -> None:
        self._heap: list[Request] = []

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy."""
        heapq.heappush(self._heap, request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to priority policy."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        return heapq.heappop(self._heap)

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0]

    def prepend_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        self.add_request(request)

    def prepend_requests(self, requests: Iterable[Request]) -> None:
        """Add all requests from another queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self._heap.remove(request)
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = requests if isinstance(requests, set) else set(requests)
        self._heap = [r for r in self._heap if r not in requests_to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._heap)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to priority policy."""
        heap_copy = self._heap[:]
        while heap_copy:
            yield heapq.heappop(heap_copy)


class ContinuumRequestQueue(RequestQueue):
    """Program-level FCFS with pinned sessions preferred.

    A program is identified by ``session_id`` when present, else ``request_id``.
    The scheduler calls ``set_pinned`` each schedule step with the sessions that
    currently hold a Continuum KV pin.
    """

    def __init__(self) -> None:
        self._queue: deque[Request] = deque()
        self._first_entry: dict[str, float] = {}
        self._pinned_sessions: set[str] = set()

    @staticmethod
    def program_key(request: Request) -> str:
        if request.session_id is not None:
            return request.session_id
        return request.request_id

    def set_pinned(self, sessions: Iterable[str]) -> None:
        self._pinned_sessions = set(sessions)

    def _record_first_entry(self, request: Request) -> None:
        key = self.program_key(request)
        prev = self._first_entry.get(key)
        if prev is None or request.arrival_time < prev:
            self._first_entry[key] = request.arrival_time

    def _select(self) -> Request:
        if not self._queue:
            raise IndexError("peek from an empty queue")
        earliest: Request | None = None
        earliest_time = float("inf")
        for request in self._queue:
            key = self.program_key(request)
            if key not in self._pinned_sessions:
                continue
            entry = self._first_entry.get(key, request.arrival_time)
            if entry < earliest_time:
                earliest_time = entry
                earliest = request
        if earliest is not None:
            return earliest
        earliest = None
        earliest_time = float("inf")
        for request in self._queue:
            key = self.program_key(request)
            entry = self._first_entry.get(key, request.arrival_time)
            if entry < earliest_time:
                earliest_time = entry
                earliest = request
        assert earliest is not None
        return earliest

    def add_request(self, request: Request) -> None:
        self._record_first_entry(request)
        self._queue.append(request)

    def pop_request(self) -> Request:
        request = self._select()
        self._queue.remove(request)
        return request

    def peek_request(self) -> Request:
        return self._select()

    def prepend_request(self, request: Request) -> None:
        self._record_first_entry(request)
        self._queue.appendleft(request)

    def prepend_requests(self, requests: Iterable[Request]) -> None:
        for request in requests:
            self.prepend_request(request)

    def remove_request(self, request: Request) -> None:
        self._queue.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        to_remove = requests if isinstance(requests, set) else set(requests)
        self._queue = deque(req for req in self._queue if req not in to_remove)

    def __bool__(self) -> bool:
        return bool(self._queue)

    def __len__(self) -> int:
        return len(self._queue)

    def __iter__(self) -> Iterator[Request]:
        return iter(list(self._queue))


def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """Create request queue based on scheduling policy."""
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    elif policy == SchedulingPolicy.CONTINUUM:
        return ContinuumRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
