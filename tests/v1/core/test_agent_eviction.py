# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id

HASH_BLOCK_SIZE = 16


def _cached_pool(monkeypatch: pytest.MonkeyPatch, policy: str) -> BlockPool:
    monkeypatch.setenv("VLLM_KV_EVICTION_POLICY", policy)
    monkeypatch.setenv("VLLM_KV_SHARED_HITS", "2")
    return BlockPool(num_gpu_blocks=6, enable_caching=True, hash_block_size=HASH_BLOCK_SIZE)


def _cache(pool: BlockPool, block, tag: bytes) -> None:
    key = make_block_hash_with_group_id(BlockHash(tag), 0)
    block.set_block_hash(key, HASH_BLOCK_SIZE)
    pool._insert_block_hash(key, block, HASH_BLOCK_SIZE)


def _shared_then_private(pool: BlockPool):
    shared, private = pool.get_new_blocks(2)
    _cache(pool, shared, b"shared")
    _cache(pool, private, b"private")
    # Two later prefix hits make `shared` a prefix reused across sessions.
    pool.free_blocks([shared, private])
    for _ in range(2):
        pool.touch([shared])
        pool.free_blocks([shared])
    return shared, private


def test_agent_policy_reuses_private_block_before_shared(monkeypatch):
    pool = _cached_pool(monkeypatch, "agent")
    shared, private = _shared_then_private(pool)
    free_before = pool.get_num_free_blocks()
    # Re-free private last, so plain LRU would hand out shared before it.
    pool.touch([private])
    pool.free_blocks([private])
    evicted = pool.get_new_blocks(pool.free_block_queue.num_free_blocks)
    assert shared not in evicted
    assert shared.block_hash is not None
    assert private in evicted
    assert pool.get_num_free_blocks() == free_before - len(evicted)


def test_lru_policy_evicts_shared_when_it_is_oldest(monkeypatch):
    pool = _cached_pool(monkeypatch, "lru")
    shared, private = _shared_then_private(pool)
    pool.touch([private])
    pool.free_blocks([private])
    evicted = pool.get_new_blocks(pool.get_num_free_blocks())
    assert shared in evicted
