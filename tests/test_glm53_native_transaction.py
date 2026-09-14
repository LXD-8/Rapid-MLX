from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from vllm_mlx.speculative.native_mtp import transaction


class _AppendCache:
    def __init__(self, offset: int = 0):
        self.offset = offset

    def trim(self, count: int) -> None:
        self.offset -= count


class _TemporalCache:
    def __init__(self):
        self.state = "before"
        self._before = None
        self._generation = 0

    def start_speculation(self, length: int) -> int:
        self._before = self.state
        self._length = length
        self._generation += 1
        return self._generation

    def validate_speculation(self, lengths, generation) -> None:
        assert generation == self._generation
        assert all(0 <= length <= self._length for length in lengths)

    def commit_speculation(self, lengths, generation) -> None:
        self.validate_speculation(lengths, generation)
        self.state = (self._before, lengths[0])
        self._before = None

    def abort_speculation(self, generation) -> None:
        assert generation == self._generation
        if self._before is not None:
            self.state = self._before
            self._before = None


@pytest.fixture
def fake_cache_types(monkeypatch):
    monkeypatch.setattr(
        transaction,
        "_cache_types",
        lambda: {
            "temporal": (_TemporalCache,),
            "append": (_AppendCache,),
            "batch": (),
            "list": type("UnusedCacheList", (), {}),
        },
    )


def test_cache_transaction_commits_only_the_accepted_prefix(fake_cache_types) -> None:
    append = _AppendCache(offset=7)
    temporal = _TemporalCache()
    with transaction.CacheTransaction([append, temporal], 3) as active:
        append.offset += 3
        temporal.state = "after-three"
        active.commit([2])

    assert append.offset == 9
    assert temporal.state == ("before", 2)


def test_cache_transaction_aborts_both_cache_kinds(fake_cache_types) -> None:
    append = _AppendCache(offset=7)
    temporal = _TemporalCache()
    with transaction.CacheTransaction([append, temporal], 3):
        append.offset += 3
        temporal.state = "after-three"

    assert append.offset == 7
    assert temporal.state == "before"


def test_cache_transaction_rejects_partial_forward(fake_cache_types) -> None:
    append = _AppendCache(offset=7)
    with (
        pytest.raises(RuntimeError, match="full verification block"),
        transaction.CacheTransaction([append], 3) as active,
    ):
        append.offset += 2
        active.commit([1])
    assert append.offset == 7


def test_speculative_stats_snapshot_is_reporting_compatible() -> None:
    stats = transaction.SpeculativeStats()
    stats.record([1, 2, 3], [1, 2, 9])
    assert stats.snapshot() == (1, 2, 3)


def test_generation_hook_replaces_only_speculative_seams(monkeypatch) -> None:
    root = ModuleType("mlx_vlm")
    root.__path__ = []
    generate = ModuleType("mlx_vlm.generate")
    generate.__path__ = []
    ar = ModuleType("mlx_vlm.generate.ar")
    original_generate_step = object()
    ar.generate_step = original_generate_step
    ar.SpeculativePrefill = object()
    ar.run_speculative_rounds = object()
    ar.speculative_prefill_kwargs = object()
    generate.ar = ar
    monkeypatch.setitem(sys.modules, "mlx_vlm", root)
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate", generate)
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate.ar", ar)

    transaction.install_generation_hooks()

    assert ar.generate_step is original_generate_step
    assert ar.SpeculativePrefill is transaction.SpeculativePrefill
    assert ar.run_speculative_rounds is transaction.run_speculative_rounds
    assert ar.speculative_prefill_kwargs is transaction.speculative_prefill_kwargs


def test_generation_hook_fails_closed_without_complete_seam(monkeypatch) -> None:
    root = ModuleType("mlx_vlm")
    root.__path__ = []
    generate = ModuleType("mlx_vlm.generate")
    generate.__path__ = []
    generate.ar = SimpleNamespace(generate_step=object())
    monkeypatch.setitem(sys.modules, "mlx_vlm", root)
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate", generate)

    with pytest.raises(RuntimeError, match="qualified generation seam"):
        transaction.install_generation_hooks()
