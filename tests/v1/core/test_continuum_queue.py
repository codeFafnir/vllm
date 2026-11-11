# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.request_queue import ContinuumRequestQueue
from vllm.v1.request import Request


def _req(
    request_id: str,
    arrival: float,
    session_id: str | None = None,
) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        arrival_time=arrival,
        session_id=session_id,
    )


def test_program_fcfs_orders_by_session_first_entry() -> None:
    queue = ContinuumRequestQueue()
    a1 = _req("a1", 10.0, "A")
    b1 = _req("b1", 11.0, "B")
    a2 = _req("a2", 12.0, "A")
    queue.add_request(b1)
    queue.add_request(a2)
    queue.add_request(a1)
    # Session A first entered at 10, B at 11, so both A turns pop before B.
    first = queue.pop_request()
    second = queue.pop_request()
    third = queue.pop_request()
    assert first.session_id == "A"
    assert second.session_id == "A"
    assert {first.request_id, second.request_id} == {"a1", "a2"}
    assert third is b1


def test_pinned_session_jumps_ahead() -> None:
    queue = ContinuumRequestQueue()
    early = _req("early", 1.0, "early")
    late = _req("late", 5.0, "late")
    queue.add_request(early)
    queue.add_request(late)
    queue.set_pinned(["late"])
    assert queue.peek_request() is late
    assert queue.pop_request() is late
    assert queue.pop_request() is early


def test_missing_session_keeps_arrival_order() -> None:
    queue = ContinuumRequestQueue()
    first = _req("r1", 1.0, None)
    second = _req("r2", 2.0, None)
    queue.add_request(second)
    queue.add_request(first)
    # Without session_id, program_key is request_id, ordered by first entry.
    assert queue.pop_request().request_id == "r1"
    assert queue.pop_request().request_id == "r2"
