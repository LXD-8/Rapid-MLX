# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_mlx.spec_decode.mtp.prompt_lookup import (
    PromptLookupIndex,
    PromptLookupPolicy,
)


def test_prompt_lookup_policy_validates_model_qualified_defaults() -> None:
    policy = PromptLookupPolicy(
        enabled_by_default=True,
        min_ngram=16,
        max_ngram=64,
        max_tokens=8,
    )

    assert policy.enabled_by_default is True
    assert (policy.min_ngram, policy.max_ngram, policy.max_tokens) == (16, 64, 8)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_ngram": 1}, "min_ngram"),
        ({"min_ngram": 4, "max_ngram": 3}, "max_ngram"),
        ({"max_tokens": 0}, "max_tokens"),
    ],
)
def test_prompt_lookup_policy_rejects_invalid_configuration(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        PromptLookupPolicy(**kwargs)


def test_prompt_lookup_returns_prompt_continuation() -> None:
    index = PromptLookupIndex([1, 2, 3, 4, 5, 6, 7, 8], min_ngram=3)

    match = index.propose([9, 1, 2, 3], max_tokens=3)

    assert match is not None
    assert match.start == 3
    assert match.matched_suffix == 3
    assert match.tokens == (4, 5, 6)


def test_prompt_lookup_prefers_longest_suffix_match() -> None:
    prompt = [8, 1, 2, 3, 4, 9, 1, 2, 3, 5]
    index = PromptLookupIndex(prompt, min_ngram=3, max_ngram=4)

    match = index.propose([0, 8, 1, 2, 3], max_tokens=2)

    assert match is not None
    assert match.start == 4
    assert match.matched_suffix == 4
    assert match.tokens == (4, 9)


def test_prompt_lookup_never_copies_generated_only_text() -> None:
    index = PromptLookupIndex([1, 2, 3, 4], min_ngram=2)

    assert index.propose([7, 8, 7, 8], max_tokens=4) is None


def test_prompt_lookup_excludes_prompt_edge_without_continuation() -> None:
    index = PromptLookupIndex([1, 2, 3, 4], min_ngram=2)

    assert index.propose([3, 4], max_tokens=4) is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_ngram": 1}, "min_ngram"),
        ({"min_ngram": 4, "max_ngram": 3}, "max_ngram"),
        ({"max_candidates": 0}, "max_candidates"),
    ],
)
def test_prompt_lookup_rejects_invalid_configuration(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        PromptLookupIndex([1, 2, 3], **kwargs)


def test_prompt_lookup_policy_admits_greedy_for_every_family() -> None:
    """Greedy copying is the qualified route everywhere it is enabled at all.

    ``enabled_under_sampling`` is a second, narrower gate, so it must not be
    able to take away what a family already had: a policy that has qualified
    greedy-only still admits temperature 0.
    """
    policy = PromptLookupPolicy(enabled_by_default=True)

    assert policy.enabled_under_sampling is False
    assert policy.admits_temperature(0.0) is True
    assert policy.admits_temperature(0) is True


def test_prompt_lookup_policy_withholds_sampling_until_a_family_qualifies() -> None:
    """A family that has not measured the sampled route stays greedy-only.

    This is the conservative half of the contract and the reason the flag is
    per-family: correctness holds at any temperature, but how much of the
    copy speedup survives once acceptance is probabilistic is an empirical
    question per model family, so the route stays off until measured.
    """
    greedy_only = PromptLookupPolicy(enabled_by_default=True)
    qualified = PromptLookupPolicy(enabled_by_default=True, enabled_under_sampling=True)

    assert greedy_only.admits_temperature(0.7) is False
    assert greedy_only.admits_temperature(1e-6) is False
    assert qualified.admits_temperature(0.7) is True
    assert qualified.admits_temperature(0.0) is True
