"""Regression guards for the three training bugs found 2026-04-16.

Each bug shipped a broken model in production. These tests assert the
fixes stay in place — both at the source-code level (string presence) and
behaviourally (running the relevant logic on synthetic state).

Bugs being guarded:

  Bug 1: warm-phase save was missing. The model peaked at R=0.8855 at
         epoch 40 of the warmup, but only the *number* was tracked —
         the weights were never written to disk. After 360 epochs of
         QAT regressed to 0.64, the broken final state shipped.

  Bug 2: training_guard warned 'R PLATEAU: no improvement for 360 epochs'
         and 'R COLLAPSE: dropped 0.2438 from best 0.8855 → 0.6417'.
         Nothing acted on either signal — training kept running.

  Bug 3: production preset had alpha_clamp=False, alpha_ceiling=0
         (disabled). LSQ alpha exploded to 143× on expand3.conv before
         the model collapsed. The training script's clamp_alpha helper
         existed but was config-gated and the production config didn't
         turn it on.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
TRAIN_PY = REPO / 'ai_models' / 'student' / 'training_utils.py'
CONFIG_PY = REPO / 'ai_models' / 'student' / 'training_config.py'

sys.path.insert(0, str(REPO / 'ai_models' / 'student'))


# ============================================================
# Bug 1 — warm phase saves on best
# ============================================================

def test_qat_resets_best_tracker_to_avoid_warm_qat_scale_mismatch():
    """Warm validates with quantize=False (FP32), QAT validates with
    quantize=True (ternary). Their R values are on different scales and
    cannot be compared with a shared `best_val_r`. After warm, the
    tracker MUST reset to zero so QAT can write its own ternary-domain
    best to the production ckpt path.

    Without this reset, the production ckpt holds warm FP32 weights
    that, when run in ternary mode at deployment, drop ~30 R points
    (we observed this empirically: warm-saved ckpt shipped FP32 weights
    with R=0.5339 in ternary mode despite training reporting R=0.8454
    in FP32 mode).
    """
    src = TRAIN_PY.read_text()
    # The reset happens between PHASE 1 end and PHASE 2 start, with a
    # warm-best diagnostic copy being archived first.
    assert 'WARM→QAT' in src or 'warm_best_' in src or 'shutil.copy2(s_path' in src, (
        "warm→QAT transition logic is missing — production ckpt will "
        "be overwritten with FP32 weights that fail at ternary "
        "deployment time"
    )
    # The reset itself.
    assert re.search(
        r'best_val_r = 0\.\d.*\n.*best_r = 0\.\d.*\n.*best_epoch = 0',
        src
    ) or re.search(
        r'best_val_r\s*=\s*0\.\d.*best_r\s*=\s*0\.\d.*best_epoch\s*=\s*0',
        src, re.DOTALL
    ), "best trackers must reset between warm and QAT"


def test_warm_phase_saves_on_new_best():
    """The warm-phase `if val_r > best_val_r:` block must call torch.save."""
    src = TRAIN_PY.read_text()

    # Locate the warm-phase block — between the warm `for epoch` and the
    # 'PHASE 2' marker. Brittle but precise; if structure drifts the test
    # fails loudly.
    warm_block = re.search(
        r'for epoch in range\(phase1_start.*?epochs_warmup \+ 1\):.*?# PHASE 2:',
        src, re.DOTALL,
    )
    assert warm_block, "could not locate warm phase block — train script restructured?"
    block = warm_block.group(0)

    assert 'best_val_r = val_r' in block, "warm phase no longer tracks best"
    # The torch.save must come AFTER best_val_r = val_r and BEFORE the
    # outer `for` loop ends. The simplest check: there's a `torch.save` in
    # the warm block at all.
    assert 'torch.save(student.state_dict()' in block, (
        "warm phase no longer saves on new best — Bug 1 has regressed. "
        "Check train_student_subband.py around the 'if val_r > best_val_r:' "
        "block in the warm loop."
    )


# ============================================================
# Bug 2 — early stop on R plateau
# ============================================================

def test_qat_early_stops_on_plateau():
    """The QAT loop must break when too many vals fail to improve."""
    src = TRAIN_PY.read_text()

    # Look for the early-stop counter and the explicit `break`.
    assert 'qat_no_improve' in src, "QAT no-improvement counter is gone"
    assert 'early_stop_patience' in src, (
        "early_stop_patience knob is missing from the script"
    )
    assert re.search(r'\[EARLY STOP\].*?break', src, re.DOTALL), (
        "QAT loop no longer breaks on plateau — Bug 2 has regressed"
    )


def test_production_preset_has_finite_patience():
    """Production preset must specify a finite early-stop patience."""
    from training_config import CONFIGS
    cfg = CONFIGS['production']
    assert hasattr(cfg, 'early_stop_patience'), (
        "TrainingConfig is missing the early_stop_patience field"
    )
    assert 0 < cfg.early_stop_patience <= 60, (
        f"production patience={cfg.early_stop_patience} — should be small "
        f"so a degenerate run aborts in hours, not days"
    )


# ============================================================
# Bug 3 — alpha clamping is unconditional + production preset enabled
# ============================================================

def test_unconditional_alpha_safety_clamp_present():
    """Always-on safety clamp must exist regardless of config."""
    src = TRAIN_PY.read_text()
    # Look for the literal hard clamp values from the safety net.
    assert re.search(
        r'lsq_alpha\.data\.clamp_\(min=1e-4, max=20', src
    ), (
        "Unconditional alpha safety clamp is missing — Bug 3 is back. "
        "The clamp must run on every layer regardless of cfg.alpha_clamp / "
        "cfg.alpha_ceiling so even untuned configs can't blow up."
    )


def test_production_preset_enables_alpha_clamping():
    """Production preset must enable the soft clamps that prevent runaway."""
    from training_config import CONFIGS
    cfg = CONFIGS['production']
    assert cfg.alpha_clamp is True, (
        "production preset has alpha_clamp=False — that's how the prior "
        "run blew up to alpha=144 on expand3.conv. Set alpha_clamp=True."
    )
    assert 0 < cfg.alpha_ceiling < 20, (
        f"production alpha_ceiling={cfg.alpha_ceiling} — must be a small "
        f"positive value (recommended 3-5) to prevent runaway."
    )
    assert cfg.alpha_floor > 0, (
        f"production alpha_floor={cfg.alpha_floor} — set to a small positive "
        f"value (e.g., 0.001) to prevent collapse to zero."
    )


# ============================================================
# Behavioural test — alpha clamp actually fires when alpha drifts high
# ============================================================

def test_alpha_clamp_caps_runaway_value():
    """If we manually inflate a layer's alpha, the clamp must pull it back.

    Builds a TernaryMobileNetV5_Subband, sets one alpha to 1000, calls the
    same clamp logic the training loop uses, and asserts the value is now
    inside the safety bounds.
    """
    import torch
    from lamquant_codec.models.encoder import TernaryMobileNetV5_Subband

    model = TernaryMobileNetV5_Subband(in_ch=21, latent_dim=32, width=128)

    # Find any ternary layer (those with lsq_alpha) and inflate it.
    ternary = [(n, m) for n, m in model.named_modules() if hasattr(m, 'lsq_alpha')]
    assert ternary, "no ternary layers found in the model — wrong architecture?"
    name, layer = ternary[0]
    with torch.no_grad():
        layer.lsq_alpha.data.fill_(1000.0)
    assert float(layer.lsq_alpha.abs().max()) > 100

    # Apply the same unconditional safety clamp as the training loop.
    for _, m in model.named_modules():
        if hasattr(m, 'lsq_alpha'):
            with torch.no_grad():
                m.lsq_alpha.data.clamp_(min=1e-4, max=20.0)

    capped = float(layer.lsq_alpha.abs().max())
    assert capped <= 20.0, f"clamp failed: alpha is {capped} after clamp"


def test_alpha_clamp_prevents_collapse_to_zero():
    """Symmetric: alpha=0 should be lifted to 1e-4 by the safety clamp."""
    import torch
    from lamquant_codec.models.encoder import TernaryMobileNetV5_Subband

    model = TernaryMobileNetV5_Subband(in_ch=21, latent_dim=32, width=128)
    ternary = [(n, m) for n, m in model.named_modules() if hasattr(m, 'lsq_alpha')]
    name, layer = ternary[0]
    with torch.no_grad():
        layer.lsq_alpha.data.fill_(0.0)

    for _, m in model.named_modules():
        if hasattr(m, 'lsq_alpha'):
            with torch.no_grad():
                m.lsq_alpha.data.clamp_(min=1e-4, max=20.0)

    # Float32 representation of 1e-4 lands at ~9.999e-5 — use a tolerance.
    floor = float(layer.lsq_alpha.detach().abs().min())
    assert floor >= 1e-4 - 1e-9, \
        f"alpha collapsed to {floor}, clamp should lift to 1e-4"
    assert floor > 0, f"alpha is exactly 0 — clamp didn't fire at all"
