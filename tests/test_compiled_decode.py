# SPDX-License-Identifier: Apache-2.0
"""Contracts for request-private whole-step compiled decode replay."""

from __future__ import annotations

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx
from mlx_lm.generate import BatchGenerator
from mlx_lm.models.cache import KVCache

import vllm_mlx.compiled_decode as compiled_decode
from vllm_mlx.compiled_decode import (
    CompiledDecodeStep,
    ShapeStableKVCache,
    convert_cache,
)
from vllm_mlx.compiled_precision import (
    compiled_decode_precision,
    gated_product,
)
from vllm_mlx.singleton_cache_fastpath import _promote_layer


class _ToyModel:
    def __call__(self, tokens, *, cache):
        batch, length = tokens.shape
        value = tokens.astype(mx.float32).reshape(batch, 1, length, 1)
        cache[0].update_and_fetch(value, value)
        values = tokens.astype(mx.float32)
        return mx.stack([values, -values], axis=-1)

    def make_cache(self):
        return [KVCache()]


def _filled_cache(token: int = 3) -> KVCache:
    cache = KVCache()
    value = mx.array([[[[token]]]], dtype=mx.float32)
    cache.update_and_fetch(value, value)
    mx.eval(cache.state)
    return cache


def test_compiled_step_replays_once_and_threads_cache_state() -> None:
    model = _ToyModel()
    stock = _filled_cache()
    stable = convert_cache([_filled_cache()])
    step = CompiledDecodeStep(model, stable)

    for token in (5, 7, 11):
        inputs = mx.array([[token]], dtype=mx.int32)
        expected = model(inputs, cache=[stock])
        actual = step(inputs)
        step.confirm_oldest(actual)
        assert bool(mx.array_equal(expected, actual).item())

    step.drain_pending()
    assert stable[0].size() == stock.size() == 4
    assert step.receipt() == {
        "traces": 1,
        "variants": 1,
        "submissions": 3,
        "completions": 3,
        "pending": 0,
        "poisoned": False,
        "poison_reason": None,
    }


def test_shape_stable_promotion_drains_and_returns_batched_cache() -> None:
    stable = ShapeStableKVCache.from_kv_cache(_filled_cache())

    class _Owner:
        calls = []

        def drain_pending(self, *, phase):
            self.calls.append(phase)

    owner = _Owner()
    stable._rapid_compiled_owner = owner
    promoted = _promote_layer(stable)

    assert owner.calls == ["cache conversion"]
    assert type(promoted).__name__ == "BatchKVCache"
    assert promoted.keys.shape[0] == 1
    assert promoted.offset == 1


def test_convert_cache_is_transactional_on_unsupported_layer() -> None:
    original = [_filled_cache(), object()]
    with pytest.raises(TypeError, match=r"cache\[1\]"):
        convert_cache(original)
    assert type(original[0]) is KVCache
    assert type(original[1]) is object


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
def test_compiled_gate_product_preserves_eager_bytes(dtype) -> None:
    gate = mx.array([-6.84375, -1.0, 0.0, 1.0], dtype=dtype)
    value = mx.array([0.25, -2.0, 3.0, 4.0], dtype=dtype)
    expected = value * mx.sigmoid(gate)
    with compiled_decode_precision():
        actual = gated_product(gate, value)
    assert bool(mx.array_equal(expected, actual).item())


def test_batch_generator_installer_replays_and_cleans_up(monkeypatch) -> None:
    monkeypatch.setattr(compiled_decode, "model_qualification_reason", lambda *_: None)
    monkeypatch.setattr(
        compiled_decode, "install_qwen35_attention_gate_precision", lambda _: 1
    )
    from vllm_mlx.singleton_cache_fastpath import install_singleton_cache_fastpath

    install_singleton_cache_fastpath()
    generator = BatchGenerator(_ToyModel(), max_tokens=4)
    assert compiled_decode.install_compiled_decode(
        generator, generator.model, model_name="qualified"
    )
    uid = generator.insert([[3, 2, 1]], max_tokens=[4])[0]
    responses = []
    for _ in range(8):
        _, generated = generator.next()
        responses.extend(generated)
        if any(
            response.uid == uid and response.finish_reason for response in generated
        ):
            break

    stats = generator._rapid_compiled_decode_stats
    assert [response.token for response in responses if response.uid == uid] == [0] * 4
    assert stats["attachments"] == 1
    assert stats["traces"] == 1
    assert stats["submissions"] == stats["completions"] == 4
    assert stats["poisoned"] is False


def test_batch_join_promotes_compiled_cache_then_new_singleton_reattaches(
    monkeypatch,
) -> None:
    monkeypatch.setattr(compiled_decode, "model_qualification_reason", lambda *_: None)
    monkeypatch.setattr(
        compiled_decode, "install_qwen35_attention_gate_precision", lambda _: 1
    )
    from vllm_mlx.singleton_cache_fastpath import install_singleton_cache_fastpath

    install_singleton_cache_fastpath()
    generator = BatchGenerator(
        _ToyModel(), max_tokens=8, prefill_batch_size=2, completion_batch_size=2
    )
    assert compiled_decode.install_compiled_decode(
        generator, generator.model, model_name="qualified"
    )
    first = generator.insert([[3, 2, 1]], max_tokens=[8])[0]
    generator.next()
    generator.next()
    second = generator.insert([[4, 3, 2]], max_tokens=[4])[0]

    finished = set()
    for _ in range(16):
        _, generated = generator.next()
        finished.update(
            response.uid for response in generated if response.finish_reason is not None
        )
        if finished == {first, second}:
            break
    assert finished == {first, second}

    third = generator.insert([[5, 4, 3]], max_tokens=[3])[0]
    for _ in range(8):
        _, generated = generator.next()
        if any(
            response.uid == third and response.finish_reason for response in generated
        ):
            break
    stats = generator._rapid_compiled_decode_stats
    assert stats["attachments"] == 2, stats
    assert stats["poisoned"] is False


def test_declined_request_does_not_retry_cache_conversion_each_token(
    monkeypatch,
) -> None:
    monkeypatch.setattr(compiled_decode, "model_qualification_reason", lambda *_: None)
    monkeypatch.setattr(
        compiled_decode, "install_qwen35_attention_gate_precision", lambda _: 1
    )
    attempts = 0

    def decline(_cache):
        nonlocal attempts
        attempts += 1
        raise ValueError("deliberate boundary decline")

    monkeypatch.setattr(compiled_decode, "convert_cache", decline)
    generator = BatchGenerator(_ToyModel(), max_tokens=5)
    assert compiled_decode.install_compiled_decode(
        generator, generator.model, model_name="qualified"
    )
    uid = generator.insert([[3, 2, 1]], max_tokens=[5])[0]
    for _ in range(10):
        _, generated = generator.next()
        if any(
            response.uid == uid and response.finish_reason for response in generated
        ):
            break

    assert attempts == 1
    assert generator._rapid_compiled_decode_stats["last_decline_reason"] == (
        "deliberate boundary decline"
    )
