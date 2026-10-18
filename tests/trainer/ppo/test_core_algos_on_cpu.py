# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import unittest
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import verl.trainer.ppo.core_algos
from verl.trainer.ppo.core_algos import (
    agg_loss,
    compute_gae_advantage_return,
    compute_grpo_outcome_advantage,
    compute_grpo_vectorized_outcome_advantage,
    compute_outcome_nll_loss,
    compute_rloo_outcome_advantage,
    compute_rloo_vectorized_outcome_advantage,
    compute_self_distillation_loss,
    contrastive_target_log_probs,
    finalize_ratio_metrics,
    get_adv_estimator_fn,
    kl_penalty,
    register_adv_est,
    reverse_kl_logit_grad_norm,
)


def mock_test_fn():
    pass


class TestRegisterAdvEst(unittest.TestCase):
    def setUp(self):
        """Clear the registry before each test"""
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY.clear()
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY = {
            "gae": lambda x: x * 2,
            "vtrace": lambda x: x + 1,
        }
        self.ADV_ESTIMATOR_REGISTRY = verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY

    def tearDown(self) -> None:
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY.clear()
        return super().tearDown()

    def test_register_new_function(self):
        """Test registering a new function with a string name"""

        @register_adv_est("test_estimator")
        def test_fn():
            pass

        self.assertIn("test_estimator", self.ADV_ESTIMATOR_REGISTRY)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["test_estimator"], test_fn)

    def test_register_with_enum(self):
        """Test registering with an enum value (assuming AdvantageEstimator exists)"""
        from enum import Enum

        class AdvantageEstimator(Enum):
            TEST = "test_enum_estimator"

        @register_adv_est(AdvantageEstimator.TEST)
        def test_fn():
            pass

        self.assertIn("test_enum_estimator", self.ADV_ESTIMATOR_REGISTRY)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["test_enum_estimator"], test_fn)

    def test_duplicate_registration_same_function(self):
        """Test that registering the same function twice doesn't raise an error"""
        register_adv_est("duplicate_test")(mock_test_fn)
        register_adv_est("duplicate_test")(mock_test_fn)

        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["duplicate_test"], mock_test_fn)

    def test_duplicate_registration_different_function(self):
        """Test that registering different functions with same name raises ValueError"""

        @register_adv_est("conflict_test")
        def test_fn1():
            pass

        with self.assertRaises(ValueError):

            @register_adv_est("conflict_test")
            def test_fn2():
                pass

    def test_decorator_preserves_function(self):
        """Test that the decorator returns the original function"""

        def test_fn():
            return "original"

        decorated = register_adv_est("preserve_test")(test_fn)
        self.assertEqual(decorated(), "original")

    def test_multiple_registrations(self):
        """Test registering multiple different functions"""
        init_adv_count = len(self.ADV_ESTIMATOR_REGISTRY)

        @register_adv_est("estimator1")
        def fn1():
            pass

        @register_adv_est("estimator2")
        def fn2():
            pass

        self.assertEqual(len(self.ADV_ESTIMATOR_REGISTRY), 2 + init_adv_count)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["estimator1"], fn1)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["estimator2"], fn2)

    def test_get_adv_estimator_fn_valid_names(self):
        """Test that valid names return the correct function from registry."""
        # Test GAE
        gae_fn = get_adv_estimator_fn("gae")
        assert gae_fn(5) == 10  # 5 * 2 = 10

        # Test Vtrace
        vtrace_fn = get_adv_estimator_fn("vtrace")
        assert vtrace_fn(5) == 6  # 5 + 1 = 6

    def test_get_adv_estimator_fn_invalid_name(self):
        """Test that invalid names raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            get_adv_estimator_fn("invalid_name")
        assert "Unknown advantage estimator simply: invalid_name" in str(excinfo.value)

    def test_get_adv_estimator_fn_case_sensitive(self):
        """Test that name lookup is case-sensitive."""
        with pytest.raises(ValueError):
            get_adv_estimator_fn("GAE")  # Different case


def test_multi_turn_compute_gae_advantage_return():
    """Test multi-turn GAE skip observation tokens."""
    gamma = random.uniform(0.0, 1.0)
    lam = random.uniform(0.0, 1.0)

    rewards = torch.tensor([[0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.1, 1.0, 0.0, 0.0]], dtype=torch.float)

    values1 = torch.tensor(
        [
            [
                random.uniform(-100.0, 100.0),
                random.random(),
                4.0,
                5.0,
                6.0,
                random.uniform(-100.0, 0),
                random.random(),
                7.0,
                9.0,
                0.0,
                0.0,
            ]
        ],
        dtype=torch.float,
    )

    values2 = torch.tensor(
        [
            [
                random.random(),
                random.uniform(-100.0, 100.0),
                4.0,
                5.0,
                6.0,
                random.random(),
                random.uniform(0.0, 100.0),
                7.0,
                9.0,
                0.0,
                0.0,
            ]
        ],
        dtype=torch.float,
    )

    response_mask = torch.tensor([[0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0]], dtype=torch.float)

    adv1, ret1 = compute_gae_advantage_return(rewards, values1, response_mask, gamma, lam)
    adv2, ret2 = compute_gae_advantage_return(rewards, values2, response_mask, gamma, lam)

    ret1 *= response_mask
    ret2 *= response_mask
    assert torch.equal(adv1, adv2), f"{adv1=}, {adv2=}"
    assert torch.equal(ret1, ret2), f"{ret1=}, {ret2=}"
    print(f" [CORRECT] \n\n{adv1=}, \n\n{ret1=}")


def _make_group_index(batch_size: int, num_groups: int) -> np.ndarray:
    """Create a numpy index array ensuring each group has at least 2 samples."""
    assert num_groups * 2 <= batch_size, "batch_size must allow >=2 samples per group"
    counts: list[int] = [2] * num_groups
    remaining = batch_size - 2 * num_groups
    for _ in range(remaining):
        counts[random.randrange(num_groups)] += 1
    index = []
    for gid, c in enumerate(counts):
        index.extend([gid] * c)
    random.shuffle(index)
    return np.asarray(index, dtype=np.int64)


def _rand_mask(batch_size: int, seq_len: int) -> torch.Tensor:
    mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.int64).float()
    rows_without_one = (mask.sum(dim=-1) == 0).nonzero(as_tuple=True)[0]
    if len(rows_without_one) > 0:
        mask[rows_without_one, -1] = 1.0
    return mask


@pytest.mark.parametrize(
    "batch_size,seq_len,num_groups,seed",
    [
        (64, 128, 5, 0),
        (128, 256, 8, 1),
        (512, 512, 10, 2),
    ],
)
def test_rloo_and_vectorized_equivalence(batch_size: int, seq_len: int, num_groups: int, seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    index = _make_group_index(batch_size, num_groups)
    response_mask = _rand_mask(batch_size, seq_len)
    base_rewards = torch.randn(batch_size, seq_len, dtype=torch.float32)
    token_level_rewards = base_rewards * response_mask
    adv1, ret1 = compute_rloo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    adv2, ret2 = compute_rloo_vectorized_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    # Print concise diagnostics for visibility during test runs
    adv_max_diff = (adv1 - adv2).abs().max().item()
    ret_max_diff = (ret1 - ret2).abs().max().item()
    total_mask_tokens = int(response_mask.sum().item())
    print(
        f"[RLOO] seed={seed} groups={num_groups} shape={adv1.shape} "
        f"mask_tokens={total_mask_tokens} adv_max_diff={adv_max_diff:.3e} ret_max_diff={ret_max_diff:.3e}"
    )
    assert adv1.shape == adv2.shape == (batch_size, seq_len)
    assert ret1.shape == ret2.shape == (batch_size, seq_len)
    assert torch.allclose(adv1, adv2, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ret1, ret2, rtol=1e-5, atol=1e-6)


def test_grpo_vectorized_matches_original_for_low_variance_rewards():
    token_level_rewards = torch.tensor([[1.0], [1.00001], [2.0], [2.00001]], dtype=torch.float32)
    response_mask = torch.ones_like(token_level_rewards)
    index = np.array(["prompt-a", "prompt-a", "prompt-b", "prompt-b"], dtype=object)

    adv1, ret1 = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    adv2, ret2 = compute_grpo_vectorized_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    assert torch.allclose(adv1, adv2, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ret1, ret2, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "batch_size,seq_len,num_groups,seed",
    [
        (64, 128, 5, 0),
        (128, 256, 8, 1),
        (512, 512, 10, 2),
    ],
)
def test_grpo_and_vectorized_equivalence(batch_size: int, seq_len: int, num_groups: int, seed: int):
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Generate group indices (numpy array of shape [batch_size])
    index = _make_group_index(batch_size, num_groups)

    # Generate binary response mask (at least one valid token per row)
    response_mask = _rand_mask(batch_size, seq_len)

    # Generate token-level rewards and apply mask
    base_rewards = torch.randn(batch_size, seq_len, dtype=torch.float32)
    token_level_rewards = base_rewards * response_mask

    # Compute GRPO outcome advantage (original implementation)
    adv1, ret1 = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    # Compute GRPO outcome advantage (vectorized implementation)
    adv2, ret2 = compute_grpo_vectorized_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    # Diagnostic info for visibility (same style as RLOO test)
    adv_max_diff = (adv1 - adv2).abs().max().item()
    ret_max_diff = (ret1 - ret2).abs().max().item()
    total_mask_tokens = int(response_mask.sum().item())
    print(
        f"[GRPO] seed={seed} groups={num_groups} shape={adv1.shape} "
        f"mask_tokens={total_mask_tokens} adv_max_diff={adv_max_diff:.3e} ret_max_diff={ret_max_diff:.3e}"
    )

    # Assert shape and numerical equivalence
    assert adv1.shape == adv2.shape == (batch_size, seq_len)
    assert ret1.shape == ret2.shape == (batch_size, seq_len)
    assert torch.allclose(adv1, adv2, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ret1, ret2, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "name,base",
    [
        ("k1+", "k1"),
        ("kl+", "kl"),
        ("abs+", "abs"),
        ("k3+", "k3"),
        ("low_var_kl+", "low_var_kl"),
    ],
)
def test_kl_penalty_straight_through_value_matches_base(name, base):
    """The ``+`` suffix is a straight-through trick that swaps in the k2
    gradient while keeping the base estimator's value. Therefore the forward
    value of e.g. ``k3+`` must match the value of plain ``k3``.

    Regression test for the bug where ``kl_penalty(..., "k3+")`` raised
    ``NotImplementedError`` because the wrapper forwarded the ``+`` suffix to
    ``kl_penalty_forward`` without stripping it.
    """
    torch.manual_seed(0)
    logprob = torch.randn(4, 8, requires_grad=True)
    ref_logprob = torch.randn(4, 8)

    plus_value = kl_penalty(logprob, ref_logprob, name)
    base_value = kl_penalty(logprob, ref_logprob, base)
    assert torch.allclose(plus_value, base_value)


def test_kl_penalty_k3_plus_uses_k2_gradient():
    """With ``k3+`` the gradient w.r.t. ``logprob`` should equal the gradient
    obtained from the ``k2`` (``0.5 * log_ratio**2``) estimator, since the
    straight-through trick routes the backward pass through ``k2``.
    """
    torch.manual_seed(0)
    logprob = torch.randn(4, 8, requires_grad=True)
    ref_logprob = torch.randn(4, 8)

    out_plus = kl_penalty(logprob, ref_logprob, "k3+").sum()
    (grad_plus,) = torch.autograd.grad(out_plus, logprob)

    logprob_k2 = logprob.detach().clone().requires_grad_(True)
    out_k2 = kl_penalty(logprob_k2, ref_logprob, "k2").sum()
    (grad_k2,) = torch.autograd.grad(out_k2, logprob_k2)

    assert torch.allclose(grad_plus, grad_k2)


def test_compute_self_distillation_loss_full_logit_alpha_endpoints():
    student_all_log_probs = torch.log_softmax(torch.randn(2, 3, 7), dim=-1)
    teacher_all_log_probs = torch.log_softmax(torch.randn(2, 3, 7), dim=-1)
    student_log_probs = student_all_log_probs[..., 0]
    teacher_log_probs = teacher_all_log_probs[..., 0]
    response_mask = torch.ones(2, 3)

    cfg_alpha_fwd = SimpleNamespace(
        full_logit_distillation=True,
        distillation_topk=None,
        distillation_add_tail=True,
        alpha=0.0,
        is_clip=None,
    )
    cfg_alpha_rev = SimpleNamespace(
        full_logit_distillation=True,
        distillation_topk=None,
        distillation_add_tail=True,
        alpha=1.0,
        is_clip=None,
    )

    loss_fwd, metrics_fwd = compute_self_distillation_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        self_distillation_config=cfg_alpha_fwd,
        student_all_log_probs=student_all_log_probs,
        teacher_all_log_probs=teacher_all_log_probs,
    )
    loss_rev, metrics_rev = compute_self_distillation_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        self_distillation_config=cfg_alpha_rev,
        student_all_log_probs=student_all_log_probs,
        teacher_all_log_probs=teacher_all_log_probs,
    )

    assert torch.isfinite(loss_fwd)
    assert torch.isfinite(loss_rev)
    assert not torch.allclose(loss_fwd, loss_rev)
    assert "self_distillation/alpha" in metrics_fwd
    assert "self_distillation/kl_type_code" in metrics_rev


def test_compute_self_distillation_loss_topk_and_masking():
    student_topk_log_probs = torch.log_softmax(torch.randn(1, 4, 3), dim=-1)
    teacher_topk_log_probs = torch.log_softmax(torch.randn(1, 4, 3), dim=-1)
    student_log_probs = torch.randn(1, 4)
    teacher_log_probs = torch.randn(1, 4)
    response_mask = torch.ones(1, 4)

    cfg = SimpleNamespace(
        full_logit_distillation=True,
        distillation_topk=3,
        distillation_add_tail=True,
        alpha=0.5,
        is_clip=None,
    )

    loss_topk, metrics_topk = compute_self_distillation_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        self_distillation_config=cfg,
        student_topk_log_probs=student_topk_log_probs,
        teacher_topk_log_probs=teacher_topk_log_probs,
    )
    assert torch.isfinite(loss_topk)
    assert metrics_topk["self_distillation/use_topk"] == 1.0

    zero_mask = torch.zeros(1, dtype=torch.float32)
    loss_masked, metrics_masked = compute_self_distillation_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        self_distillation_config=cfg,
        student_topk_log_probs=student_topk_log_probs,
        teacher_topk_log_probs=teacher_topk_log_probs,
        self_distillation_mask=zero_mask,
    )
    assert torch.allclose(loss_masked, torch.tensor(0.0, dtype=loss_masked.dtype, device=loss_masked.device))
    assert finalize_ratio_metrics(metrics_masked)["self_distillation/supervised_token_fraction"] == 0.0


def _log_dist(*probs):
    return torch.tensor(probs).log()


def test_contrastive_target_is_the_student_where_the_teachers_agree():
    student = torch.log_softmax(torch.randn(2, 3, 101), dim=-1)
    hint = torch.log_softmax(torch.randn(2, 3, 101), dim=-1)
    target, _ = contrastive_target_log_probs(student, hint, hint)
    torch.testing.assert_close(target, student, atol=1e-6, rtol=0)


def test_contrastive_target_is_the_hint_teacher_where_the_contrast_is_the_student():
    # every bin is above the floor for the student or the hint teacher, so the shift reaches all of them
    student = torch.log_softmax(torch.randn(2, 3, 9) * 0.5, dim=-1)
    hint = torch.log_softmax(torch.randn(2, 3, 9) * 0.5, dim=-1)
    target, reached = contrastive_target_log_probs(student, hint, student)
    assert reached.all()
    torch.testing.assert_close(target, hint, atol=1e-6, rtol=0)


def test_contrastive_target_leaves_the_bins_no_one_gives_mass_to():
    student = _log_dist(0.706, 0.28, 0.004, 0.002, 0.008)
    hint = _log_dist(0.50, 0.46, 0.003, 0.001, 0.036)
    contrast = _log_dist(0.60, 0.30, 0.0005, 0.0095, 0.09)
    target, reached = contrastive_target_log_probs(student, hint, contrast)

    # bins 2 and 3 are under the floor for both; the last is under it for the student only
    assert reached.tolist() == [True, True, False, False, True]
    shifted = student + torch.tensor([0.5 / 0.6, 0.46 / 0.3, 1.0, 1.0, 0.036 / 0.09]).log()
    torch.testing.assert_close(target, torch.log_softmax(shifted, dim=-1))
    # without the floor, bins 2 and 3 would move against each other by the teachers' noise
    unfloored = torch.log_softmax(student + hint - contrast, dim=-1)
    assert abs(float(unfloored[3] - unfloored[2] - (student[3] - student[2]))) > 1.0
    assert float(target[3] - target[2]) == pytest.approx(float(student[3] - student[2]))


def test_contrastive_target_is_renormalised_and_a_constant():
    student = torch.log_softmax(torch.randn(4, 101), dim=-1).requires_grad_(True)
    hint = torch.log_softmax(torch.randn(4, 101), dim=-1)
    contrast = torch.log_softmax(torch.randn(4, 101), dim=-1)
    target, _ = contrastive_target_log_probs(student, hint, contrast)
    torch.testing.assert_close(target.exp().sum(-1), torch.ones(4))
    assert not target.requires_grad


def _contrast_cfg(**overrides):
    cfg = dict(full_logit_distillation=True, distillation_topk=6, distillation_add_tail=True, alpha=1.0, is_clip=None)
    return SimpleNamespace(**{**cfg, **overrides})


def _with_tail(topk_log_probs):
    log_s = torch.logsumexp(topk_log_probs, -1, keepdim=True).clamp(max=-1e-7)
    return torch.cat([topk_log_probs, torch.log(-torch.expm1(log_s))], -1)


def _topk_batch(seed=0):
    generator = torch.Generator().manual_seed(seed)
    return [torch.log_softmax(torch.randn(2, 4, 8, generator=generator) * 0.3, dim=-1)[..., :6] for _ in range(3)]


def _contrast_loss(student, hint, contrast, cfg=None, mask=None):
    return compute_self_distillation_loss(
        student_log_probs=torch.zeros(2, 4),
        teacher_log_probs=torch.zeros(2, 4),
        response_mask=torch.ones(2, 4),
        self_distillation_config=cfg or _contrast_cfg(),
        student_topk_log_probs=student,
        teacher_topk_log_probs=hint,
        contrast_topk_log_probs=contrast,
        self_distillation_mask=mask,
    )


def test_a_contrast_that_matches_the_hint_teacher_is_a_zero_loss():
    student, hint, _ = _topk_batch()
    loss, metrics = _contrast_loss(student, hint, hint)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)
    assert metrics["self_distillation/kl_hint__sum"] > 0
    assert metrics["self_distillation/kl_target__sum"] == pytest.approx(0.0, abs=1e-5)


def test_a_contrast_that_matches_the_student_is_the_hint_teacher_loss():
    student, hint, _ = _topk_batch()
    plain, _ = compute_self_distillation_loss(
        student_log_probs=torch.zeros(2, 4),
        teacher_log_probs=torch.zeros(2, 4),
        response_mask=torch.ones(2, 4),
        self_distillation_config=_contrast_cfg(),
        student_topk_log_probs=student,
        teacher_topk_log_probs=hint,
    )
    loss, metrics = _contrast_loss(student, hint, student)
    torch.testing.assert_close(loss, plain, atol=1e-6, rtol=0)

    ratios = finalize_ratio_metrics(metrics)
    assert ratios["self_distillation/kl_target"] == pytest.approx(ratios["self_distillation/kl_hint"], rel=1e-4)
    assert ratios["self_distillation/contrast_floor_masked"] == 0.0
    assert metrics["self_distillation/contrast_bins__sum"] == 2 * 4 * 7  # 6 top-k bins and the tail, per token


def test_the_contrast_metrics_count_the_supervised_tokens_only():
    student, hint, contrast = _topk_batch()
    mask = torch.tensor([1.0, 0.0])
    _, metrics = _contrast_loss(student, hint, contrast, mask=mask)
    assert metrics["self_distillation/contrast_bins__sum"] == 4 * 7
    assert metrics["self_distillation/supervised_tokens__sum"] == 4
    _, none = _contrast_loss(student, hint, contrast, mask=torch.zeros(2))
    assert none["self_distillation/contrast_bins__sum"] == 0 and none["self_distillation/kl_target__sum"] == 0.0


def test_the_loss_follows_the_student_toward_a_constant_target():
    student, hint, contrast = _topk_batch()
    student = student.clone().requires_grad_(True)
    loss, _ = _contrast_loss(student, hint, contrast)
    (grad,) = torch.autograd.grad(loss, student)

    binned = _with_tail(student)
    target, _ = contrastive_target_log_probs(binned.detach(), _with_tail(hint), _with_tail(contrast))
    reference = (binned.exp() * (binned - target)).sum(-1).mean()
    (expected,) = torch.autograd.grad(reference, student)
    torch.testing.assert_close(loss, reference.detach())
    torch.testing.assert_close(grad, expected)


def test_a_contrast_needs_full_logit_top_k_distillation():
    student, hint, contrast = _topk_batch()
    for cfg in (_contrast_cfg(distillation_topk=None), _contrast_cfg(full_logit_distillation=False)):
        with pytest.raises(ValueError, match="distillation_topk"):
            _contrast_loss(student, hint, contrast, cfg=cfg)


def test_the_anchored_target_is_todays_where_the_student_is_the_anchor(monkeypatch):
    student = torch.log_softmax(torch.randn(2, 3, 101), dim=-1)
    hint = torch.log_softmax(torch.randn(2, 3, 101) * 0.1, dim=-1)
    contrast = torch.log_softmax(torch.randn(2, 3, 101) * 0.1, dim=-1)
    today, today_reached = contrastive_target_log_probs(student, hint, contrast)
    anchored, reached = contrastive_target_log_probs(student, hint, contrast, anchor=student)
    assert torch.equal(reached, today_reached)
    # the soft clip bends a shift of x nats by about x^3 / 300: nothing at these sizes (|x| < 1)
    assert (hint - contrast).abs().max() < 1.0
    torch.testing.assert_close(anchored, today, atol=1e-3, rtol=0)
    monkeypatch.setattr(verl.trainer.ppo.core_algos, "CONTRAST_CLIP", 1e6)
    unclipped, _ = contrastive_target_log_probs(student, hint, contrast, anchor=student)
    torch.testing.assert_close(unclipped, today, atol=1e-6, rtol=0)


def test_the_anchored_target_stays_put_while_the_student_drifts():
    anchor = torch.log_softmax(torch.randn(2, 3, 101), dim=-1)
    hint = torch.log_softmax(torch.randn(2, 3, 101), dim=-1)
    contrast = torch.log_softmax(torch.randn(2, 3, 101), dim=-1)
    drifted = torch.log_softmax(anchor + torch.randn(2, 3, 101), dim=-1)

    at_anchor, _ = contrastive_target_log_probs(anchor, hint, contrast, anchor=anchor)
    anchored, reached = contrastive_target_log_probs(drifted, hint, contrast, anchor=anchor)
    torch.testing.assert_close(anchored, at_anchor, atol=0, rtol=0)
    shift = torch.where(reached, 10.0 * torch.tanh((hint - contrast) / 10.0), torch.zeros_like(hint))
    torch.testing.assert_close(anchored, torch.log_softmax(anchor + shift, dim=-1))

    today, _ = contrastive_target_log_probs(drifted, hint, contrast)
    today_at_anchor, _ = contrastive_target_log_probs(anchor, hint, contrast)
    assert (today - today_at_anchor).abs().max() > 0.5, "the student-anchored target moves with the student"


def test_the_anchored_shift_is_soft_clipped_per_bin():
    anchor = torch.log_softmax(torch.zeros(5), dim=-1)
    hint = torch.log_softmax(torch.tensor([30.0, 0.0, 0.0, 0.0, 0.0]), dim=-1)
    contrast = torch.log_softmax(torch.zeros(5), dim=-1)
    target, reached = contrastive_target_log_probs(anchor, hint, contrast, anchor=anchor)
    assert reached.all()
    shift = hint - contrast  # about +1.6 nats on the first bin and -28.4 on the others
    moved = target - anchor
    clipped = 10.0 * torch.tanh(shift / 10.0)
    torch.testing.assert_close(moved - moved[1], clipped - clipped[1])
    assert float(moved.max() - moved.min()) < 20.0
    today, _ = contrastive_target_log_probs(anchor, hint, contrast)
    assert float((today - anchor).max() - (today - anchor).min()) == pytest.approx(30.0, rel=1e-4)


def _anchored_loss(student, hint, contrast, anchor, mask=None, grad_proxies=False):
    return compute_self_distillation_loss(
        student_log_probs=torch.zeros(2, 4),
        teacher_log_probs=torch.zeros(2, 4),
        response_mask=torch.ones(2, 4),
        self_distillation_config=_contrast_cfg(),
        student_topk_log_probs=student,
        teacher_topk_log_probs=hint,
        contrast_topk_log_probs=contrast,
        anchor_topk_log_probs=anchor,
        self_distillation_mask=mask,
        grad_proxies=grad_proxies,
    )


def test_the_anchored_loss_is_the_reverse_kl_to_the_anchor_plus_the_shift():
    student, hint, contrast = _topk_batch()
    anchor = _topk_batch(seed=1)[0]
    student = student.clone().requires_grad_(True)
    loss, metrics = _anchored_loss(student, hint, contrast, anchor)
    (grad,) = torch.autograd.grad(loss, student)

    binned = _with_tail(student)
    target, _ = contrastive_target_log_probs(
        binned.detach(), _with_tail(hint), _with_tail(contrast), anchor=_with_tail(anchor)
    )
    reference = (binned.exp() * (binned - target)).sum(-1).mean()
    (expected,) = torch.autograd.grad(reference, student)
    torch.testing.assert_close(loss, reference.detach())
    torch.testing.assert_close(grad, expected)

    # a contrast equal to the hint leaves the anchor itself as the target: zero loss only at the anchor
    same, _ = _anchored_loss(anchor, hint, hint, anchor)
    assert float(same) == pytest.approx(0.0, abs=1e-6)
    away, _ = _anchored_loss(student.detach(), hint, hint, anchor)
    assert float(away) > 1e-4


def test_the_target_entropy_metrics_are_against_the_student_and_the_anchor():
    student, hint, _ = _topk_batch()
    anchor = _topk_batch(seed=1)[0]
    _, plain = _contrast_loss(student, hint, student)
    assert "self_distillation/target_entropy_minus_student__sum" in plain
    assert "self_distillation/target_entropy_minus_anchor__sum" not in plain

    _, metrics = _anchored_loss(student, hint, hint, anchor, mask=torch.tensor([1.0, 0.0]))

    def entropy(log_probs):
        binned = _with_tail(log_probs)[0]
        return -(binned.exp() * binned).sum(-1)

    # the target is the anchor: zero against it, the anchor's entropy minus the student's against the student
    assert metrics["self_distillation/target_entropy_minus_anchor__sum"] == pytest.approx(0.0, abs=1e-4)
    expected = float((entropy(anchor) - entropy(student)).sum())
    assert metrics["self_distillation/target_entropy_minus_student__sum"] == pytest.approx(expected, abs=1e-4)
    ratios = finalize_ratio_metrics(metrics)
    assert ratios["self_distillation/target_entropy_minus_student"] == pytest.approx(expected / 4, abs=1e-4)


def test_an_anchor_needs_a_contrast():
    student, hint, _ = _topk_batch()
    with pytest.raises(ValueError, match="anchor"):
        _anchored_loss(student, hint, None, student)


def test_the_reverse_kl_grad_proxy_is_the_logit_gradient_norm():
    logits = torch.randn(3, 7, requires_grad=True)
    target = torch.log_softmax(torch.randn(3, 7), dim=-1)
    student = torch.log_softmax(logits, dim=-1)
    kl = (student.exp() * (student - target)).sum(-1)
    (grad,) = torch.autograd.grad(kl.sum(), logits)
    torch.testing.assert_close(reverse_kl_logit_grad_norm(student.detach(), target), grad.norm(dim=-1))


def test_the_sd_grad_proxy_sums_over_the_supervised_tokens_only():
    student, hint, contrast = _topk_batch()
    mask = torch.tensor([1.0, 0.0])
    _, quiet = _anchored_loss(student, hint, contrast, None, mask=mask)
    assert "self_distillation/grad_proxy_sd__sum" not in quiet
    _, metrics = _anchored_loss(student, hint, contrast, None, mask=mask, grad_proxies=True)
    binned = _with_tail(student)
    target, _ = contrastive_target_log_probs(binned, _with_tail(hint), _with_tail(contrast))
    norms = reverse_kl_logit_grad_norm(binned, target)
    assert metrics["self_distillation/grad_proxy_sd__sum"] == pytest.approx(float(norms[0].sum()), rel=1e-5)
    assert float(metrics["self_distillation/grad_proxy_sd_term"]) == pytest.approx(float(norms[0].mean()), rel=1e-5)


def _outcome(student_log_probs, outcome_mask, weights, **kwargs):
    return compute_outcome_nll_loss(
        student_log_probs=student_log_probs,
        response_mask=torch.ones_like(student_log_probs),
        outcome_mask=outcome_mask,
        outcome_weights=weights,
        loss_agg_mode=kwargs.pop("loss_agg_mode", "traj-mean-token-mean"),
        dp_size=1,
        global_batch_size=kwargs.pop("global_batch_size", 1),
        **kwargs,
    )


def test_the_outcome_term_is_the_mean_over_trajectories_of_their_token_mean_nll():
    log_probs = torch.log(torch.tensor([[0.5, 0.25, 0.5, 0.9], [0.1, 0.2, 0.4, 0.8]]))
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    # one trajectory in two segments (shares 0.5 each) and nothing else
    loss, metrics = _outcome(log_probs, mask, torch.tensor([0.5, 0.5]))
    first, second = -log_probs[0, :2].mean(), -log_probs[1].mean()
    torch.testing.assert_close(loss, 0.5 * first + 0.5 * second)
    assert metrics["outcome/tokens__sum"] == 6 and metrics["outcome/rows__sum"] == 2
    assert metrics["outcome/nll__sum"] == pytest.approx(float(-log_probs[mask.bool()].sum()))

    halved, _ = _outcome(log_probs, mask, torch.tensor([0.5, 0.5]), global_batch_size=2)
    torch.testing.assert_close(halved, loss / 2)
    # a signed per-row weight is the same term pushing the other way
    flipped, _ = _outcome(log_probs, mask, torch.tensor([-0.5, -0.5]))
    torch.testing.assert_close(flipped, -loss)
    with pytest.raises(ValueError, match="traj-mean-token-mean"):
        _outcome(log_probs, mask, torch.tensor([0.5, 0.5]), loss_agg_mode="token-mean")


def test_the_nll_grad_proxy_is_the_distance_from_the_one_hot_label():
    probs = torch.softmax(torch.randn(1, 3, 6), dim=-1)
    labels = torch.tensor([[2, 0, 5]])
    exact = (torch.nn.functional.one_hot(labels, 6).float() - probs).norm(dim=-1)
    label_log_probs = probs.log().gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    topk_log_probs, topk_indices = probs.log().topk(6, dim=-1)
    _, metrics = _outcome(
        label_log_probs,
        torch.ones(1, 3),
        torch.ones(1),
        student_topk_log_probs=topk_log_probs,
        student_topk_indices=topk_indices,
        labels=labels,
    )
    assert metrics["outcome/grad_proxy_nll__sum"] == pytest.approx(float(exact.sum()), rel=1e-5)
    assert float(metrics["outcome/grad_proxy_nll_term"]) == pytest.approx(float(exact.mean()), rel=1e-5)


def test_compute_self_distillation_loss_optional_is_clip():
    student_all_log_probs = torch.log_softmax(torch.randn(2, 3, 5), dim=-1)
    teacher_all_log_probs = torch.log_softmax(torch.randn(2, 3, 5), dim=-1)
    student_log_probs = student_all_log_probs[..., 0]
    teacher_log_probs = teacher_all_log_probs[..., 0]
    old_log_probs = student_log_probs + 5.0
    response_mask = torch.ones(2, 3)

    cfg_no_clip = SimpleNamespace(
        full_logit_distillation=True,
        distillation_topk=None,
        distillation_add_tail=True,
        alpha=0.5,
        is_clip=None,
    )
    cfg_clip = SimpleNamespace(
        full_logit_distillation=True,
        distillation_topk=None,
        distillation_add_tail=True,
        alpha=0.5,
        is_clip=2.0,
    )

    loss_no_clip, _ = compute_self_distillation_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        self_distillation_config=cfg_no_clip,
        old_log_probs=old_log_probs,
        student_all_log_probs=student_all_log_probs,
        teacher_all_log_probs=teacher_all_log_probs,
    )
    loss_clip, metrics_clip = compute_self_distillation_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        self_distillation_config=cfg_clip,
        old_log_probs=old_log_probs,
        student_all_log_probs=student_all_log_probs,
        teacher_all_log_probs=teacher_all_log_probs,
    )

    assert torch.isfinite(loss_no_clip)
    assert torch.isfinite(loss_clip)
    assert loss_clip < loss_no_clip
    assert metrics_clip["self_distillation/is_clip"] == 2.0


TOPK_CFG = SimpleNamespace(
    full_logit_distillation=True, distillation_topk=4, distillation_add_tail=True, alpha=1.0, is_clip=None
)
RKL_CFG = SimpleNamespace(
    full_logit_distillation=False, distillation_topk=None, distillation_add_tail=False, alpha=1.0, is_clip=None
)
ARGMAX_SUM = "self_distillation/teacher_prefers_other__sum"
ARGMAX_RATIO = "self_distillation/teacher_prefers_other_token"


def test_teacher_prefers_other_token_is_absent_rather_than_zero_without_the_argmax():
    """The step-1 gate: a 0.0 here reads as the teacher never disagreeing with the student, not
    as never measured, so the run without top-k must report nothing at all."""
    student_topk = torch.log_softmax(torch.randn(2, 3, 4), dim=-1)
    teacher_topk = torch.log_softmax(torch.randn(2, 3, 4), dim=-1)
    kwargs = dict(
        student_log_probs=student_topk[..., 0],
        teacher_log_probs=teacher_topk[..., 0],
        response_mask=torch.ones(2, 3),
    )

    _, topk = compute_self_distillation_loss(
        self_distillation_config=TOPK_CFG,
        student_topk_log_probs=student_topk,
        teacher_topk_log_probs=teacher_topk,
        **kwargs,
    )
    _, rkl = compute_self_distillation_loss(self_distillation_config=RKL_CFG, **kwargs)

    assert ARGMAX_SUM in topk and ARGMAX_RATIO in finalize_ratio_metrics(topk)
    assert ARGMAX_SUM not in rkl and ARGMAX_RATIO not in finalize_ratio_metrics(rkl)
    # the other ratios still finalize without it
    assert "self_distillation/gap_mean" in finalize_ratio_metrics(rkl)


def test_the_argmax_key_survives_a_micro_batch_with_no_supervised_token():
    """dp aggregation counts values per key, so with top-k on, the key set may not depend on
    whether this particular micro-batch had anything to supervise."""
    student_topk = torch.log_softmax(torch.randn(2, 3, 4), dim=-1)
    teacher_topk = torch.log_softmax(torch.randn(2, 3, 4), dim=-1)
    _, empty = compute_self_distillation_loss(
        student_log_probs=student_topk[..., 0],
        teacher_log_probs=teacher_topk[..., 0],
        response_mask=torch.ones(2, 3),
        self_distillation_config=TOPK_CFG,
        student_topk_log_probs=student_topk,
        teacher_topk_log_probs=teacher_topk,
        self_distillation_mask=torch.zeros(2, 3),
    )
    assert empty[ARGMAX_SUM] == 0.0 and empty["self_distillation/supervised_tokens__sum"] == 0.0


def test_agg_loss_traj_mean_token_mean_is_the_mean_of_trajectory_token_means():
    """Two trajectories of two segments each plus one unsupervised row, weighted the way the
    trainer does for this mode: raw shares (a segment's supervised tokens over its trajectory's)
    and the supervised-trajectory count as the denominator."""
    torch.manual_seed(0)
    loss_mat = torch.randn(5, 6)
    loss_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0, 0],  # traj A, segment 0: 3 tokens
            [1, 0, 0, 0, 0, 0],  # traj A, segment 1: 1 token
            [1, 1, 0, 0, 0, 0],  # traj B, segment 0: 2 tokens
            [1, 1, 0, 0, 0, 0],  # traj B, segment 1: 2 tokens
            [0, 0, 0, 0, 0, 0],  # unsupervised
        ],
        dtype=torch.float32,
    )
    shares = torch.tensor([3 / 4, 1 / 4, 1 / 2, 1 / 2, 0.0])
    got = agg_loss(loss_mat, loss_mask, "traj-mean-token-mean", seq_weights=shares, global_batch_size=2)

    traj_a = loss_mat[0:2][loss_mask[0:2].bool()].mean()
    traj_b = loss_mat[2:4][loss_mask[2:4].bool()].mean()
    assert torch.allclose(got, (traj_a + traj_b) / 2)
    # a micro-batch holding one of the trajectories returns that trajectory's share of the mean
    half = agg_loss(loss_mat[:2], loss_mask[:2], "traj-mean-token-mean", seq_weights=shares[:2], global_batch_size=2)
    assert torch.allclose(half, traj_a / 2)
    # the row-count denominator of seq-mean-token-mean is what the new mode replaces
    rows = agg_loss(loss_mat, loss_mask, "seq-mean-token-mean", seq_weights=shares, global_batch_size=5)
    assert torch.allclose(rows, (traj_a + traj_b) / 5)


def test_agg_loss_traj_mean_token_mean_all_masked_is_finite():
    loss_mat = torch.randn(3, 4)
    loss_mask = torch.zeros(3, 4)
    got = agg_loss(loss_mat, loss_mask, "traj-mean-token-mean", seq_weights=torch.zeros(3))
    assert got == 0.0 and torch.isfinite(got)
    got = agg_loss(loss_mat, loss_mask, "traj-mean-token-mean", seq_weights=torch.zeros(3), global_batch_size=1)
    assert got == 0.0 and torch.isfinite(got)


@pytest.mark.parametrize(
    "parts",
    [
        [torch.rand(1, 400, generator=torch.Generator().manual_seed(0)) * 1e-3,
         torch.rand(1, 50, generator=torch.Generator().manual_seed(1)) * 5.0],
        # the rkl_token loss is signed: a median below zero must not read as the smallest edge
        [-torch.rand(1, 300, generator=torch.Generator().manual_seed(2)) * 1e-2,
         torch.rand(1, 100, generator=torch.Generator().manual_seed(3)) * 1e-2],
    ],
)
def test_loss_quantiles_are_over_every_token_of_the_step(parts):
    """Summed over micro-batches, the histogram gives the step's own quantiles, not a mean of
    per-micro-batch ones: a heavy-tailed micro-batch must not pull the median up."""
    from verl.trainer.ppo.core_algos import _distillation_signal_metrics

    summed = {}
    for loss in parts:
        mask = torch.ones_like(loss)
        zeros = torch.zeros_like(loss)
        for key, value in _distillation_signal_metrics(loss, mask, zeros, zeros).items():
            summed[key] = summed.get(key, 0.0) + value
    out = finalize_ratio_metrics(summed)
    every = torch.cat([p.flatten() for p in parts])
    assert out["self_distillation/loss_p50"] == pytest.approx(torch.quantile(every, 0.5).item(), rel=0.2)
    assert out["self_distillation/loss_p90"] == pytest.approx(torch.quantile(every, 0.9).item(), rel=0.2)
    assert not any("loss_hist" in key for key in out)


if __name__ == "__main__":
    unittest.main()
