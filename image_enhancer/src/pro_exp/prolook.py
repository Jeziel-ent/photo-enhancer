"""ProLook: deterministic professional photo polish (non-generative).

Professional retoucher polish applied ONLY to existing pixels: multi-scale
luminance clarity, mild luminance unsharp-mask, gentle toning and conservative
"vibrance". Adjustments are soft-clamped (tanh) so highlights, shadows and
faces stay natural; skins tones get extra protection. No content, detail or
geometry is ever created.

All operations are pure OpenCV/numpy on the 4K input and are fully
deterministic.
"""

import cv2
import numpy as np

LITE = dict(clarity=0.35, usm=0.35, tone=1.04, vibrance=0.25)
STANDARD = dict(clarity=0.60, usm=0.55, tone=1.06, vibrance=0.35)

_CLARITY_RADII = ((41, 0.35), (81, 0.18), (161, 0.09))


def soft_clip(adj, limit):
    if limit <= 0:
        return np.zeros_like(adj)
    return limit * np.tanh(adj / limit)


def _vibrance_mult(a, b, strength, skin_weight=0.4):
    sat = np.sqrt(np.square(a - 128.0) + np.square(b - 128.0))
    satn = np.clip(sat / 90.0, 0.0, 1.0)
    f = 1.0 + strength * 0.12 * satn
    skin = (a >= 130) & (a <= 175) & (b >= 125) & (b <= 155)
    f[skin] = 1.0 + (f[skin] - 1.0) * skin_weight
    return f


def prolook(bgr, clarity=0.0, usm=0.0, usm_sigma=0.8, tone=1.0, vibrance=0.0):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)

    if clarity:
        for r, w in _CLARITY_RADII:
            lr = cv2.boxFilter(l, -1, (r, r), normalize=True)
            l = l + soft_clip((l - lr) * w, limit=clarity * 40.0)

    if usm:
        g = cv2.GaussianBlur(l, (0, 0), usm_sigma)
        l = l + soft_clip((l - g) * usm, limit=25.0)

    if abs(tone - 1.0) > 1e-6:
        m = float(np.mean(l))
        l = m + (l - m) * tone

    if vibrance:
        f = _vibrance_mult(a, b, vibrance)
        a = 128.0 + (a - 128.0) * f
        b = 128.0 + (b - 128.0) * f

    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)
    lab2 = cv2.merge([l, a, b]).astype(np.uint8)
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def lite(bgr):
    return prolook(bgr, **LITE)


def standard(bgr):
    return prolook(bgr, **STANDARD)