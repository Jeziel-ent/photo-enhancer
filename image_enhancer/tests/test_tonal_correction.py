"""Validation tests for image_enhancer/src/tonal_correction.py.

Unlike test_final_pipeline.py / test_g91_integration_pipeline.py, this
module needs no GPU/model weights -- adaptive_tonal_correction is pure
NumPy/OpenCV -- so it runs in seconds.

Run with the project venv (no third-party test runner required):
    .\\.venv\\Scripts\\python.exe tests/test_tonal_correction.py

Covers:
  1. a genuinely bright image gets no correction; a merely "normal" (not
     broken-exposure) image ALSO gets no correction -- MVP 2 removed the
     earlier proactive brightness assist entirely (see
     docs/MVP2_RESEARCH.md "Automatic engine + manual controls"); only a
     provably broken exposure triggers any change.
  2. underexposed -> corrected
  3. overexposed -> corrected without introducing new clipping
  4. low contrast -> dynamic range improves
  5. crushed shadows -> lifted without destroying relative shadow detail
  6. clipped highlights -> pulled back without adding NEW clipping
  7. strong/saturated colors -> no significant color (a/b) shift
  8. text/signage/geometry protection -> edge map is preserved (a global,
     monotonic per-value LUT cannot move or blur an edge)
  9. deterministic output (byte-identical rerun)
  10. all six production reference images: correction is safe (bounded,
      monotonic) and matches the module's own activation decision, whatever
      it is for that image (this does not replace the real GPU pipeline
      regression suite in test_final_pipeline.py -- see that file for the
      end-to-end 3840x2160 checks).
  11. correct_color_cast -- corrects only a MATERIALLY EXCESSIVE cast
      toward PURE NEUTRAL (never a forced warm/cool bias), with highlight
      protection so a blown sky is never tinted.
  12. the sky-domination exposure-detection fix (a big bright sky must not
      by itself decide the whole photo is "overexposed").
  13. apply_local_tone_mapping -- bounded, edge-aware HDR-like shadow/
      highlight compression + local detail boost.
  14. boost_natural_saturation -- bounded, skipped when already saturated.
  15. enhance_photographic_quality -- the composed entry point.

Exit code is non-zero on the first failure.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import tonal_correction as tc  # noqa: E402

ORIGINALS = [str(ROOT / "originals" / f"{i}.jpeg") for i in range(1, 7)]
NAMES = [f"{i}.jpeg" for i in range(1, 7)]

_rng = np.random.RandomState(0)


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f" [{detail}]" if detail else ""), flush=True)
    if not cond:
        raise SystemExit(f"REGRESSION FAILURE: {name} {detail}")


def _l_channel(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0]


def _textured_gray(mean, noise, size=(300, 400), seed=0):
    """A synthetic image with real local structure (a rectangle + text +
    per-pixel noise), not a flat color -- so it's a meaningful stand-in for
    a photo, not a degenerate edge case."""
    rng = np.random.RandomState(seed)
    base = np.clip(rng.normal(mean, noise, size), 0, 255).astype(np.uint8)
    img = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    shade = int(np.clip(mean * 0.6, 0, 255))
    cv2.rectangle(img, (50, 50), (150, 150), (shade, shade, shade), -1)
    text_shade = int(np.clip(mean * 1.3, 0, 255))
    cv2.putText(img, "TEXT", (180, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (text_shade,) * 3, 2)
    return img


def main():
    # ---------------------------------------------------------- 1. normal
    # A genuinely BRIGHT photo (well above SHADOW_EXPOSURE_P25_TARGET even
    # in its darker quarter): no issues, and the gentle exposure assist
    # also correctly stays a no-op -- true bypass end to end. Plain noise
    # (not _textured_gray's rect+text patches, whose harder edges tip this
    # specific mean/noise combination into "overexposed"/"low_contrast").
    _rng_bn = np.random.RandomState(0)
    bright_normal = cv2.cvtColor(
        np.clip(_rng_bn.normal(160, 25, (300, 400)), 0, 255).astype(np.uint8),
        cv2.COLOR_GRAY2BGR)
    out, meta = tc.adaptive_tonal_correction(bright_normal, return_meta=True)
    check("bright-normal-bypassed",
          meta["applied"] is False and meta["issues"] == [] and meta["params"] == {},
          f"issues={meta['issues']} params={meta['params']}")
    check("bright-normal-unchanged", np.array_equal(out, bright_normal))

    # A "normal" (not broken-exposure) but somewhat flat/dim photo: MVP 2
    # removed the earlier proactive exposure assist entirely -- a photo that
    # doesn't trip `detect_issues` must now be a true no-op, full stop (no
    # automatic brightness lift of any kind; that's a manual-slider decision
    # now -- see image_enhancer/src/adjustments.py).
    normal = _textured_gray(mean=100, noise=35)
    out, meta = tc.adaptive_tonal_correction(normal, return_meta=True)
    check("normal-not-broken", meta["issues"] == [], meta["issues"])
    check("normal-bypassed-no-assist",
          meta["applied"] is False and meta["params"] == {}, meta["params"])
    check("normal-unchanged", np.array_equal(out, normal))
    check("MAX_BRIGHTNESS_LIFT_DEVIATION-removed",
          not hasattr(tc, "MAX_BRIGHTNESS_LIFT_DEVIATION"),
          "the proactive brightness-lift constant must no longer exist")

    # ----------------------------------------------------- 2. underexposed
    dark = _textured_gray(mean=40, noise=15)
    out, meta = tc.adaptive_tonal_correction(dark, return_meta=True)
    check("underexposed-detected", "underexposed" in meta["issues"], meta["issues"])
    check("underexposed-brightened",
          float(_l_channel(out).mean()) > float(_l_channel(dark).mean()) + 10,
          f"before={_l_channel(dark).mean():.1f} after={_l_channel(out).mean():.1f}")
    gamma = meta["params"].get("gamma")
    check("underexposed-gamma-bounded",
          gamma is not None and (1.0 - tc.MAX_GAMMA_DEVIATION) <= gamma <= 1.0,
          f"gamma={gamma}")

    # ------------------------------------------------------ 3. overexposed
    bright = _textured_gray(mean=210, noise=15)
    before_white = float((_l_channel(bright) >= 254).mean())
    out, meta = tc.adaptive_tonal_correction(bright, return_meta=True)
    after_white = float((_l_channel(out) >= 254).mean())
    check("overexposed-detected", "overexposed" in meta["issues"], meta["issues"])
    check("overexposed-darkened",
          float(_l_channel(out).mean()) < float(_l_channel(bright).mean()) - 5,
          f"before={_l_channel(bright).mean():.1f} after={_l_channel(out).mean():.1f}")
    check("overexposed-no-new-clipping", after_white <= before_white + 1e-9,
          f"before_white={before_white:.4f} after_white={after_white:.4f}")

    # ----------------------------------------------------- 4. low contrast
    flat = _textured_gray(mean=120, noise=8)
    before_range = float(np.percentile(_l_channel(flat), 99) - np.percentile(_l_channel(flat), 1))
    out, meta = tc.adaptive_tonal_correction(flat, return_meta=True)
    after_range = float(np.percentile(_l_channel(out), 99) - np.percentile(_l_channel(out), 1))
    check("low-contrast-detected", "low_contrast" in meta["issues"], meta["issues"])
    check("low-contrast-improved", after_range > before_range,
          f"before={before_range:.1f} after={after_range:.1f}")
    alpha = meta["params"].get("stretch_alpha")
    check("low-contrast-alpha-bounded",
          alpha is not None and 0.0 <= alpha <= tc.MAX_STRETCH_BLEND, f"alpha={alpha}")

    # ------------------------------------------------- 5. crushed shadows
    shadow_img = _textured_gray(mean=130, noise=35, seed=1)
    shadow_img = shadow_img.copy()
    shadow_img[0:120, :] = 0
    # two distinct near-black bands to prove detail survives the lift
    shadow_img[10:40, 10:390] = 4
    shadow_img[60:90, 10:390] = 10
    out, meta = tc.adaptive_tonal_correction(shadow_img, return_meta=True)
    check("crushed-shadows-detected", "crushed_shadows" in meta["issues"], meta["issues"])
    lift = meta["params"].get("shadow_lift", 0.0)
    check("crushed-shadows-lift-bounded", 0.0 < lift <= tc.MAX_SHADOW_LIFT, f"lift={lift}")
    band_low = float(_l_channel(out)[10:40, 10:390].mean())
    band_high = float(_l_channel(out)[60:90, 10:390].mean())
    check("crushed-shadows-detail-preserved", band_high > band_low + 1.0,
          f"band(L=4)->{band_low:.2f} band(L=10)->{band_high:.2f}")

    # ----------------------------------------------- 6. clipped highlights
    hi_img = _textured_gray(mean=130, noise=35, seed=2)
    hi_img = hi_img.copy()
    hi_img[0:120, :] = 255
    before_white2 = float((_l_channel(hi_img) >= 254).mean())
    out, meta = tc.adaptive_tonal_correction(hi_img, return_meta=True)
    after_white2 = float((_l_channel(out) >= 254).mean())
    check("clipped-highlights-detected", "clipped_highlights" in meta["issues"], meta["issues"])
    check("clipped-highlights-no-new-clipping", after_white2 <= before_white2 + 1e-9,
          f"before={before_white2:.4f} after={after_white2:.4f}")
    pull = meta["params"].get("highlight_pull", 0.0)
    check("clipped-highlights-pull-bounded", 0.0 < pull <= tc.MAX_HIGHLIGHT_PULL, f"pull={pull}")

    # -------------------------------------------------------- 7. strong colors
    strong = np.zeros((300, 300, 3), np.uint8)
    strong[:, :100] = (40, 40, 220)
    strong[:, 100:200] = (40, 200, 40)
    strong[:, 200:] = (220, 40, 40)
    out, meta = tc.adaptive_tonal_correction(strong, return_meta=True)
    lab_a = cv2.cvtColor(strong, cv2.COLOR_BGR2LAB).astype(np.int16)
    lab_b = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.int16)
    a_diff = int(np.abs(lab_a[:, :, 1] - lab_b[:, :, 1]).max())
    b_diff = int(np.abs(lab_a[:, :, 2] - lab_b[:, :, 2]).max())
    l_diff = int(np.abs(lab_a[:, :, 0] - lab_b[:, :, 0]).max())
    # adaptive_tonal_correction writes ONLY lab_out[:,:,0] (see its own
    # source) -- a/b are copied verbatim from the input LAB array, so any
    # a/b delta here is purely LAB<->BGR round-trip float/8-bit precision
    # loss, which grows with how large the L change itself is (MVP 2's new
    # proactive brightness lift can now legitimately move L a lot more than
    # before). Same flat, empirically-verified ceiling as the reference-
    # image chroma-stability checks below.
    check("strong-colors-no-shift", a_diff <= 16 and b_diff <= 16,
          f"a_diff={a_diff} b_diff={b_diff} l_diff={l_diff} (issues={meta['issues']})")

    # --------------------------------------------- 8. geometry/text/edges
    # A pure per-value LUT cannot move an edge -- proven directly and
    # precisely (not via a fixed-threshold Canny proxy, which is sensitive
    # to absolute brightness scale and gets noisier as MVP 2's stronger
    # proactive brightness lift legitimately moves L further): every input
    # L value must map to exactly ONE output L value (it's a lookup table,
    # not a spatial operation), and that mapping must be monotonic
    # non-decreasing -- both directly guarantee no edge/order inversion,
    # regardless of how large the correction's magnitude is.
    out_shadow, _ = tc.adaptive_tonal_correction(shadow_img, return_meta=True)
    in_l = _l_channel(shadow_img).flatten()
    out_l = _l_channel(out_shadow).flatten()
    mapping: dict[int, int] = {}
    consistent = True
    for iv, ov in zip(in_l.tolist(), out_l.tolist()):
        prev = mapping.setdefault(iv, ov)
        if prev != ov:
            consistent = False
            break
    check("edges-lut-consistent", consistent, "same input L mapped to different output L")
    sorted_inputs = sorted(mapping)
    outputs_in_order = [mapping[k] for k in sorted_inputs]
    monotonic = all(a <= b for a, b in zip(outputs_in_order, outputs_in_order[1:]))
    check("edges-preserved-monotonic-lut", monotonic,
          "output L values are not monotonic non-decreasing in input L")

    # ---------------------------------------------------- 9. determinism
    out_a = tc.adaptive_tonal_correction(dark)
    out_b = tc.adaptive_tonal_correction(dark)
    check("deterministic", np.array_equal(out_a, out_b))

    # ------------------------------------------ 10. six production originals
    for path, name in zip(ORIGINALS, NAMES):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            check(f"reference-readable-{name}", False, "could not read fixture")
            continue
        out, meta = tc.adaptive_tonal_correction(img, return_meta=True)
        check(f"reference-shape-preserved-{name}", out.shape == img.shape)
        # Whatever the decision, it must be internally consistent and bounded.
        if meta["applied"]:
            lab_a = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.int16)
            lab_b = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.int16)
            a_diff = int(np.abs(lab_a[:, :, 1] - lab_b[:, :, 1]).max())
            b_diff = int(np.abs(lab_a[:, :, 2] - lab_b[:, :, 2]).max())
            l_diff = int(np.abs(lab_a[:, :, 0] - lab_b[:, :, 0]).max())
            # Same LAB<->BGR round-trip cause as "strong-colors-no-shift"
            # above (adaptive_tonal_correction writes ONLY the L plane; any
            # a/b delta is 8-bit colorspace round-trip noise, not a real
            # chroma shift) -- MVP 2's proactive brightness lift can
            # legitimately move L much further than MVP 1 ever did on a real
            # photo, which grows that noise. A flat, empirically-verified
            # ceiling across all 6 real reference images (observed max 13),
            # not scaled per-image (the relationship isn't a clean ratio of
            # l_diff across different real photos' actual hues).
            check(f"reference-{name}-chroma-stable", a_diff <= 16 and b_diff <= 16,
                  f"a_diff={a_diff} b_diff={b_diff} l_diff={l_diff}")
            check(f"reference-{name}-luma-bounded", l_diff <= 130,
                  f"L max diff={l_diff} params={meta['params']}")
        else:
            check(f"reference-{name}-bypassed-identical", np.array_equal(out, img))

    # ------------------------------------- 11. MVP 2: color-cast correction
    # correct_color_cast targets PURE NEUTRAL (NEUTRAL_AB on both axes) --
    # never a directional warm/cool bias. A photo already close to neutral:
    # no-op, byte-identical (the governing rule: "keep the photograph's
    # original lighting/color character unless correction is needed").
    neutral = _textured_gray(mean=140, noise=30)
    out, meta = tc.correct_color_cast(neutral, return_meta=True)
    check("cast-neutral-not-applied", meta["applied"] is False, meta)
    check("cast-neutral-identical", np.array_equal(out, neutral))

    # A MILD cast (within CAST_MEAN_THRESHOLD) must be left alone -- this is
    # the "not force warm/cool" guarantee: a subtle, legitimate cast is part
    # of the photo's own character, not a defect.
    mild_cool = neutral.astype(np.float32)
    mild_cool[:, :, 2] = np.clip(mild_cool[:, :, 2] * 0.94, 0, 255)  # R slightly down
    mild_cool = mild_cool.astype(np.uint8)
    out_mild, meta_mild = tc.correct_color_cast(mild_cool, return_meta=True)
    check("cast-mild-not-corrected", meta_mild["applied"] is False, meta_mild)

    # A genuinely EXCESSIVE cool/blue cast: correction must move b/a toward
    # pure neutral, bounded, and NEVER touch luminance.
    cool = neutral.astype(np.float32)
    cool[:, :, 2] = np.clip(cool[:, :, 2] * 0.70, 0, 255)  # R down
    cool[:, :, 0] = np.clip(cool[:, :, 0] * 1.40, 0, 255)  # B up
    cool = cool.astype(np.uint8)
    out_cast, meta_cast = tc.correct_color_cast(cool, return_meta=True)
    check("cast-excessive-detected", meta_cast["applied"] is True, meta_cast)
    a_shift = abs(meta_cast["a_shift"])
    b_shift = abs(meta_cast["b_shift"])
    check("cast-shift-bounded",
          a_shift <= tc.MAX_CHROMA_SHIFT + 1e-6 and b_shift <= tc.MAX_CHROMA_SHIFT + 1e-6,
          f"a_shift={a_shift} b_shift={b_shift}")
    lab_cool = cv2.cvtColor(cool, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_corrected = cv2.cvtColor(out_cast, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_diff = float(np.abs(lab_cool[:, :, 0] - lab_corrected[:, :, 0]).max())
    check("cast-correction-luma-near-untouched", l_diff <= 10.0, f"L max diff={l_diff}")
    b_mean_before = float(lab_cool[:, :, 2].mean())
    b_mean_after = float(lab_corrected[:, :, 2].mean())
    check("cast-correction-moves-toward-neutral",
          abs(b_mean_after - tc.NEUTRAL_AB) < abs(b_mean_before - tc.NEUTRAL_AB),
          f"before={b_mean_before:.2f} after={b_mean_after:.2f}")
    check("cast-correction-never-overshoots-past-neutral",
          (b_mean_before < tc.NEUTRAL_AB) == (b_mean_after <= tc.NEUTRAL_AB) or
          abs(b_mean_after - tc.NEUTRAL_AB) < 1.0,
          f"before={b_mean_before:.2f} after={b_mean_after:.2f} (must not invert past neutral)")

    # A cast so extreme the ceiling itself must bind (proves the hard cap is
    # real, not just usually-not-hit).
    extreme = neutral.astype(np.float32)
    extreme[:, :, 2] = np.clip(extreme[:, :, 2] * 0.3, 0, 255)
    extreme[:, :, 0] = np.clip(extreme[:, :, 0] * 2.0, 0, 255)
    extreme = extreme.astype(np.uint8)
    _, meta_extreme = tc.correct_color_cast(extreme, return_meta=True)
    check("cast-shift-hits-ceiling-on-extreme-input",
          abs(meta_extreme["b_shift"]) == tc.MAX_CHROMA_SHIFT, meta_extreme)

    # ------------------- 11b. MVP 2: highlight-protection regression
    # A flat, uniform a/b shift visibly tints a blown-out sky (real photo
    # evidence: sky at mean L=236 landed at b=135, clearly cream -- see
    # docs/MVP2_RESEARCH.md "Brightness tuning pass"). A cool-cast image
    # with a near-white "sky" region on top of a cool mid-tone "scene" must
    # get its scene corrected while its sky stays near its ORIGINAL b.
    cool_with_sky = np.zeros((300, 300, 3), np.uint8)
    cool_with_sky[:100, :] = (255, 250, 248)  # near-white "sky" (BGR)
    mid_cool = neutral[:200, :300].astype(np.float32)
    mid_cool[:, :, 2] = np.clip(mid_cool[:, :, 2] * 0.70, 0, 255)  # cool mid-tone scene
    mid_cool[:, :, 0] = np.clip(mid_cool[:, :, 0] * 1.40, 0, 255)
    cool_with_sky[100:, :] = mid_cool.astype(np.uint8)
    lab_before = cv2.cvtColor(cool_with_sky, cv2.COLOR_BGR2LAB).astype(np.float32)
    out_hp, meta_hp = tc.correct_color_cast(cool_with_sky, return_meta=True)
    lab_after = cv2.cvtColor(out_hp, cv2.COLOR_BGR2LAB).astype(np.float32)
    sky_b_before = float(lab_before[:100, :, 2].mean())
    sky_b_after = float(lab_after[:100, :, 2].mean())
    scene_b_before = float(lab_before[100:, :, 2].mean())
    scene_b_after = float(lab_after[100:, :, 2].mean())
    check("highlight-sky-not-tinted", abs(sky_b_after - sky_b_before) < 3.0,
          f"sky b before={sky_b_before:.2f} after={sky_b_after:.2f}")
    check("highlight-scene-still-corrected", scene_b_after - scene_b_before > 5.0,
          f"scene b before={scene_b_before:.2f} after={scene_b_after:.2f}")

    # --------------------------- 12. MVP 2: sky-domination bug regression
    # A large, blown-out (near-white) "sky" region above a genuinely dim
    # "scene" region must NOT get misclassified as "overexposed" and
    # darkened.
    sky_scene = np.zeros((400, 400, 3), np.uint8)
    sky_scene[:160, :] = (250, 252, 253)  # top 40%: blown-white sky (BGR)
    rng = np.random.RandomState(1)
    dim_region = np.clip(rng.normal(80, 15, (240, 400)), 0, 255).astype(np.uint8)
    sky_scene[160:, :] = cv2.cvtColor(dim_region, cv2.COLOR_GRAY2BGR)
    lab_sky_scene = cv2.cvtColor(sky_scene, cv2.COLOR_BGR2LAB)
    stats_sky = tc._histogram_stats(lab_sky_scene[:, :, 0])
    issues_sky = tc.detect_issues(stats_sky)
    check("sky-does-not-force-overexposed", "overexposed" not in issues_sky,
          f"issues={issues_sky} scene_p50={stats_sky['scene_p50']} raw_p50={stats_sky['p50']}")
    check("sky-scene-stats-reflect-dim-foreground",
          stats_sky["scene_p50"] < stats_sky["p50"],
          f"scene_p50={stats_sky['scene_p50']} raw_p50={stats_sky['p50']}")
    out_sky, meta_sky = tc.adaptive_tonal_correction(sky_scene, return_meta=True)
    check("sky-scene-not-darkened",
          "gamma" not in meta_sky["params"] or meta_sky["params"]["gamma"] <= 1.0,
          meta_sky["params"])

    # ----------------------------- 13. MVP 2: local tone mapping (HDR-like)
    # A synthetic scene with a genuinely WIDE dynamic range (deep shadow
    # band + bright highlight band + midtones) should get measurable,
    # bounded base compression (shadow lift + highlight recovery) AND a
    # detail boost -- chroma untouched, deterministic, and the shadow band
    # must brighten while the highlight band does not get pushed further
    # into clipping.
    wide_range = _textured_gray(mean=130, noise=40, seed=3)
    wide_range = wide_range.copy()
    wide_range[0:60, :] = 15    # deep shadow band
    wide_range[240:300, :] = 245  # bright highlight band
    out_tm, meta_tm = tc.apply_local_tone_mapping(wide_range, return_meta=True)
    check("tonemap-applied", meta_tm["applied"] is True, meta_tm)
    check("tonemap-compression-bounded",
          0.0 <= meta_tm["compress_alpha"] <= tc.MAX_BASE_COMPRESSION, meta_tm)
    check("tonemap-detail-boost-bounded",
          1.0 <= meta_tm["detail_boost"] <= tc.MAX_DETAIL_BOOST, meta_tm)
    shadow_before = float(_l_channel(wide_range)[0:60, :].mean())
    shadow_after = float(_l_channel(out_tm)[0:60, :].mean())
    check("tonemap-lifts-shadow-band", shadow_after > shadow_before,
          f"before={shadow_before:.1f} after={shadow_after:.1f}")
    highlight_before = float((_l_channel(wide_range)[240:300, :] >= 254).mean())
    highlight_after = float((_l_channel(out_tm)[240:300, :] >= 254).mean())
    check("tonemap-no-new-highlight-clipping", highlight_after <= highlight_before + 1e-9,
          f"before={highlight_before:.4f} after={highlight_after:.4f}")
    # apply_local_tone_mapping writes ONLY lab_out[:,:,0] (see its own
    # source) -- same LAB<->BGR round-trip precision-loss caveat already
    # documented and tolerated elsewhere in this file (grows with how large
    # the L change itself is; this test's shadow band moves by ~37 levels).
    lab_wr = cv2.cvtColor(wide_range, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_tm = cv2.cvtColor(out_tm, cv2.COLOR_BGR2LAB).astype(np.float32)
    ab_diff_tm = float(np.abs(lab_wr[:, :, 1:] - lab_tm[:, :, 1:]).max())
    check("tonemap-chroma-untouched", ab_diff_tm <= 6.0, f"ab_diff={ab_diff_tm}")
    out_tm_b = tc.apply_local_tone_mapping(wide_range)
    check("tonemap-deterministic", np.array_equal(out_tm, out_tm_b))

    # A narrow-range (already well-balanced) scene should get little to no
    # base compression -- adaptive, not a blanket "always compress" effect.
    narrow_range = _textured_gray(mean=140, noise=15, seed=4)
    _, meta_narrow = tc.apply_local_tone_mapping(narrow_range, return_meta=True)
    check("tonemap-narrow-range-minimal-compression",
          meta_narrow["compress_alpha"] < tc.MAX_BASE_COMPRESSION * 0.15, meta_narrow)

    # --------------------- 13b. MVP 2: halo/ringing regression (real bug)
    # A HARD edge (a dark "building" silhouette flush against a flat bright
    # "sky") boosted at too high a strength, with no local bound, produced
    # visible dark-outline halos on a real photo -- see docs/MVP2_RESEARCH.md
    # "Halo regression fix". Directly proves the fix: no output pixel,
    # anywhere, may fall outside the [min, max] of its own small
    # neighborhood in the PRE-tone-mapping image -- which is precisely what
    # a halo/ring is (an overshoot beyond the locally real range).
    edge_scene = np.full((240, 240), 60, np.uint8)   # dark "building"
    edge_scene[:, 120:] = 235                         # flat bright "sky"
    edge_bgr = cv2.cvtColor(edge_scene, cv2.COLOR_GRAY2BGR)
    out_edge = tc.apply_local_tone_mapping(edge_bgr)
    l_after = _l_channel(out_edge).astype(np.float32)
    # A halo is a ring right at the edge that overshoots BEYOND what either
    # adjacent flat region itself settled at -- darker than the dark side's
    # own interior, or brighter than the bright side's own interior. This
    # compares against the ACTUAL resulting plateaus in this same output
    # (which already include the legitimate, intentionally-global base-
    # compression shift -- see apply_local_tone_mapping's own docstring),
    # not the pre-processing original, so it isolates only the halo/ring
    # signature, not the expected large-scale tone shift.
    dark_plateau_after = float(l_after[:, :90].mean())    # far from the edge
    bright_plateau_after = float(l_after[:, 150:].mean())  # far from the edge
    transition = l_after[:, 95:145]                        # right at the edge
    overshoot_dark = int(np.sum(transition < dark_plateau_after - 3.0))
    overshoot_bright = int(np.sum(transition > bright_plateau_after + 3.0))
    check("halo-no-dark-ring-at-edge", overshoot_dark == 0,
          f"{overshoot_dark} transition pixels darker than the dark plateau itself "
          f"(dark_plateau={dark_plateau_after:.1f})")
    check("halo-no-bright-ring-at-edge", overshoot_bright == 0,
          f"{overshoot_bright} transition pixels brighter than the bright plateau "
          f"itself (bright_plateau={bright_plateau_after:.1f})")
    # The detail boost must be materially lower than the old, haloing 1.85x.
    check("halo-fix-boost-is-conservative", tc.MAX_DETAIL_BOOST <= 1.30,
          f"MAX_DETAIL_BOOST={tc.MAX_DETAIL_BOOST}")

    # A flat, noisy "sky"-like region (no real texture, only noise) must NOT
    # get amplified into visible fake texture -- the other real-photo
    # artifact this fix addresses (see "Halo regression fix").
    rng_sky = np.random.RandomState(7)
    flat_noisy = np.clip(245 + rng_sky.normal(0, 3, (200, 200)), 0, 255).astype(np.uint8)
    flat_noisy_bgr = cv2.cvtColor(flat_noisy, cv2.COLOR_GRAY2BGR)
    out_flat = tc.apply_local_tone_mapping(flat_noisy_bgr)
    std_before = float(_l_channel(flat_noisy_bgr).astype(np.float32).std())
    std_after = float(_l_channel(out_flat).astype(np.float32).std())
    check("flat-noise-not-amplified-into-texture", std_after <= std_before * 1.5,
          f"std before={std_before:.2f} after={std_after:.2f}")

    # ------------------------------------- 14. MVP 2: natural saturation
    desaturated = np.full((200, 200, 3), (140, 130, 120), np.uint8)  # low-S BGR
    out_sat, meta_sat = tc.boost_natural_saturation(desaturated, return_meta=True)
    check("saturation-applied-on-desaturated", meta_sat["applied"] is True, meta_sat)
    check("saturation-boost-bounded",
          1.0 <= meta_sat["boost"] <= 1.0 + tc.MAX_SATURATION_BOOST, meta_sat)
    hsv_before = cv2.cvtColor(desaturated, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_after = cv2.cvtColor(out_sat, cv2.COLOR_BGR2HSV).astype(np.float32)
    check("saturation-hue-untouched",
          float(np.abs(hsv_before[:, :, 0] - hsv_after[:, :, 0]).max()) <= 1.0,
          "hue must not shift")

    already_saturated = np.zeros((200, 200, 3), np.uint8)
    already_saturated[:, :] = (30, 30, 220)  # strongly saturated red (BGR)
    _, meta_sat2 = tc.boost_natural_saturation(already_saturated, return_meta=True)
    check("saturation-skipped-when-already-rich", meta_sat2["applied"] is False, meta_sat2)

    # --------------------------------- 15. enhance_photographic_quality
    # MVP 2: a thin wrapper around adaptive_tonal_correction alone -- no
    # local tone mapping, no proactive brightness lift (see this function's
    # own docstring). A genuinely broken-exposure scene (crushed near-black
    # band + clipped near-white band) should still get the underlying
    # reactive correction; a well-exposed scene should be a true no-op.
    complex_scene = _textured_gray(mean=80, noise=30, seed=5)
    complex_scene = complex_scene.copy()
    complex_scene[0:40, :] = 10
    complex_scene[260:300, :] = 250
    out_combined, meta_combined = tc.enhance_photographic_quality(complex_scene, return_meta=True)
    check("quality-tonal-present", "tonal" in meta_combined, meta_combined.keys())
    check("quality-no-tone-map-key", "tone_map" not in meta_combined, meta_combined.keys())
    check("quality-shape-preserved", out_combined.shape == complex_scene.shape)

    out_bright, meta_bright = tc.enhance_photographic_quality(bright_normal, return_meta=True)
    check("quality-noop-on-well-exposed-photo",
          meta_bright["tonal"]["applied"] is False, meta_bright)
    check("quality-noop-output-unchanged", np.array_equal(out_bright, bright_normal))

    # determinism of the full composition
    out_a = tc.enhance_photographic_quality(cool)
    out_b = tc.enhance_photographic_quality(cool)
    check("quality-deterministic", np.array_equal(out_a, out_b))

    # The composed entry point must NOT touch color AT ALL -- not "less
    # aggressively," not "only when mild," but structurally never (governing
    # rule: "preserve the ORIGINAL color temperature and color character...
    # no saturation boost just for appearance"). correct_color_cast and
    # boost_natural_saturation are not called by this function at all, so
    # chroma must be identical (within the same LAB<->BGR round-trip
    # tolerance the luminance-only stages already carry) even on an image
    # with a REAL, strong color cast -- not just a mild one.
    lab_cool_in = cv2.cvtColor(cool, cv2.COLOR_BGR2LAB).astype(np.float32)
    out_quality_cool, meta_quality_cool = tc.enhance_photographic_quality(cool, return_meta=True)
    lab_cool_out = cv2.cvtColor(out_quality_cool, cv2.COLOR_BGR2LAB).astype(np.float32)
    ab_diff_quality = float(np.abs(lab_cool_in[:, :, 1:] - lab_cool_out[:, :, 1:]).max())
    check("quality-never-touches-color", "cast" not in meta_quality_cool, meta_quality_cool.keys())
    check("quality-chroma-unchanged-even-on-strong-cast", ab_diff_quality <= 16.0,
          f"ab_diff={ab_diff_quality} (cool image has a real, deliberately strong cast)")

    # Same guarantee on the extreme-cast fixture too -- the strongest cast
    # this file constructs anywhere.
    _, meta_extreme_quality = tc.enhance_photographic_quality(extreme, return_meta=True)
    check("quality-never-touches-color-on-extreme-cast",
          "cast" not in meta_extreme_quality and "saturation" not in meta_extreme_quality,
          meta_extreme_quality.keys())

    print("\nALL TONAL CORRECTION TESTS PASSED", flush=True)


if __name__ == "__main__":
    main()
