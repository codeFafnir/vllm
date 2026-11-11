# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.request_queue import ContinuumRequestQueue, SchedulingPolicy
from vllm.v1.request import Request, RequestStatus
from vllm.v1.core.kv_cache_utils import init_none_hash, get_request_block_hasher
from vllm.utils.hashing import sha256

from .utils import create_scheduler

init_none_hash(sha256)
_BLOCK_HASHER = get_request_block_hasher(16, sha256)


def _make_request(
    request_id: str,
    session_id: str | None,
    *,
    num_tokens: int = 32,
    is_last_step: bool = False,
    arrival: float | None = None,
) -> Request:
    extra = {"is_last_step": True} if is_last_step else None
    params = SamplingParams(max_tokens=8, extra_args=extra)
    return Request(
        request_id=request_id,
        prompt_token_ids=[7] * num_tokens,
        sampling_params=params,
        pooling_params=None,
        arrival_time=arrival if arrival is not None else time.time(),
        session_id=session_id,
        block_hasher=_BLOCK_HASHER,
    )


def _finish(scheduler, request: Request) -> None:
    request.status = RequestStatus.FINISHED_STOPPED
    scheduler._free_request(request)


def test_continuum_pins_finished_turn_and_releases_on_next_turn() -> None:
    scheduler = create_scheduler(
        scheduling_policy="continuum",
        enable_prefix_caching=True,
        num_blocks=128,
        block_size=16,
        max_model_len=256,
        max_num_batched_tokens=256,
    )
    assert scheduler.policy == SchedulingPolicy.CONTINUUM
    turn1 = _make_request("t1", "sess-a")
    scheduler.add_request(turn1)
    # Allocate blocks so free() has something to hold.
    scheduler.kv_cache_manager.allocate_slots(
        turn1, num_new_tokens=len(turn1.prompt_token_ids)
    )
    _finish(scheduler, turn1)
    assert "sess-a" in scheduler.pinned
    assert turn1.request_id not in scheduler.requests

    turn2 = _make_request("t2", "sess-a")
    scheduler.add_request(turn2)
    scheduler.kv_cache_manager.allocate_slots(
        turn2, num_new_tokens=len(turn2.prompt_token_ids)
    )
    _finish(scheduler, turn2)
    # Previous pin released on next turn; turn2 is now pinned.
    assert scheduler.pinned["sess-a"][0] is turn2
    assert scheduler._continuum_unpins_next_turn >= 1


def test_continuum_skips_pin_without_session_or_on_last_step() -> None:
    scheduler = create_scheduler(
        scheduling_policy="continuum",
        enable_prefix_caching=True,
        num_blocks=128,
        block_size=16,
        max_model_len=256,
        max_num_batched_tokens=256,
    )
    no_session = _make_request("n1", None)
    scheduler.add_request(no_session)
    scheduler.kv_cache_manager.allocate_slots(
        no_session, num_new_tokens=len(no_session.prompt_token_ids)
    )
    _finish(scheduler, no_session)
    assert scheduler.pinned == {}

    last = _make_request("n2", "sess-b", is_last_step=True)
    scheduler.add_request(last)
    scheduler.kv_cache_manager.allocate_slots(
        last, num_new_tokens=len(last.prompt_token_ids)
    )
    _finish(scheduler, last)
    assert scheduler.pinned == {}


def test_continuum_ttl_sweep_releases_when_idle() -> None:
    scheduler = create_scheduler(
        scheduling_policy="continuum",
        enable_prefix_caching=True,
        num_blocks=128,
        block_size=16,
        max_model_len=256,
        max_num_batched_tokens=256,
    )
    scheduler.continuum_ttl_s = 0.01
    turn = _make_request("t1", "sess-c")
    scheduler.add_request(turn)
    # Leave the waiting queue so TTL sweep does not treat the session as pending.
    scheduler.waiting.remove_request(turn)
    scheduler.kv_cache_manager.allocate_slots(
        turn, num_new_tokens=len(turn.prompt_token_ids)
    )
    _finish(scheduler, turn)
    assert "sess-c" in scheduler.pinned
    # Force an expired deadline instead of sleeping through wall time.
    pinned_req, _ = scheduler.pinned["sess-c"]
    scheduler.pinned["sess-c"] = (pinned_req, time.monotonic() - 1.0)
    scheduler._sweep_pins(time.monotonic())
    assert "sess-c" not in scheduler.pinned
    assert scheduler._continuum_unpins_ttl >= 1


def test_continuum_queue_set_pinned() -> None:
    queue = ContinuumRequestQueue()
    assert isinstance(queue, ContinuumRequestQueue)
    queue.set_pinned(["a"])
    assert queue._pinned_sessions == {"a"}
