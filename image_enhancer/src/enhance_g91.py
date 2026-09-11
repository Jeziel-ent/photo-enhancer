"""G9.1-B production-integration experiment (branch: feature/g9.1-b-integration).

Wraps the UNMODIFIED production pipeline (enhance.final_enhance, including
Candidate A billboard reconstruction) and appends the validated G9.1-B
controlled generative-detail post-process on top, at fixed, locked
parameters (strength=0.30, clip=+-5, gate=0.40 -- see
phase1_enhancement/reports/g9_1_exp/{B,real_world}/summary.json for the
tuning sweep and real-world validation this integration is based on).

Does NOT modify enhance.py, Candidate A billboard processing, or any
Phase 2/3/backend/frontend code. enhance.final_enhance remains the
production baseline, callable exactly as before -- this module is strictly
additive and opt-in via final_enhance_g91(); nothing imports or calls it
unless explicitly wired up.

No new model weights: reuses the SwinIR-GAN checkpoint already required by
g9_exp.recipe_g9 (003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import enhance as prod_enhance  # noqa: E402  (production, unmodified)
from g9_exp import recipe_g9  # noqa: E402  (validated G9 recipe, unmodified)

# Locked G9.1-B parameters, validated in reports/g9_1_exp/{B,real_world}.
# Set explicitly on every call (not left to recipe_g9's mutable tuning
# globals) so this integration can never silently pick up whatever an
# unrelated g9_exp tuning sweep last left those globals set to.
G91_B_STRENGTH = 0.30
G91_B_CLIP = 5.0
G91_B_GATE_T0 = 0.40


def _lock_g91_b_params():
    recipe_g9.G9_STRENGTH = G91_B_STRENGTH
    recipe_g9.G9_CLIP = G91_B_CLIP
    recipe_g9.G9_GATE_T0 = G91_B_GATE_T0


def final_enhance_g91(img, target=None, return_stages=False, device=None):
    """Production-shaped entry point: same contract as
    enhance.final_enhance(img, target, return_stages), plus the validated
    G9.1-B high-frequency detail pass layered on top.

    Candidate A billboard reconstruction is fully preserved: the internal
    baseline is exactly enhance.final_enhance()'s own output (unmodified
    pipeline, unmodified Candidate A), and G9's disable_on_boards=True
    feathers the generative contribution to zero inside every verified
    billboard box, so board pixels match enhance.final_enhance() output.
    """
    if target is None:
        target = (prod_enhance.OUT_W, prod_enhance.OUT_H)
    _lock_g91_b_params()
    out, stages = recipe_g9.g9_enhance(
        img, device=device, target=target, disable_on_boards=True,
        return_stages=True)
    if not return_stages:
        return out
    return out, stages
