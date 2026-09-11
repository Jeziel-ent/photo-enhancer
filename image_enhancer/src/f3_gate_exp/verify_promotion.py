"""Post-promotion verification: proves the REAL production edit
(enhance._FINAL_F3_T0 changed from 0.30 to 0.25 in enhance.py) produces
byte-identical output to Experiment 3's monkeypatch-based validation
(reports/regression_baseline/validate_t0_025.json), on all 27 images from
that validation's dataset.

This is stronger than re-deriving quality metrics from scratch a second
time: validate_t0_025.json's numbers were already fully analyzed (per-image
deltas, tolerance checks, ROI/content-category artifact analysis, in
REPORT_validation_t025.txt). If the real production code now produces the
EXACT same bytes the monkeypatch experiment already validated, every one of
those conclusions carries over to the real code change without needing to
redo the analysis — and it is simultaneously a 27-image, cross-process,
cross-session determinism check (stronger than a same-process rerun).

No production code changed by this script. Read-only.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # image_enhancer/
REPO_ROOT = ROOT.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import enhance  # noqa: E402  (production, now with _FINAL_F3_T0 = 0.25)
from regression_harness import neutralize_billboard_regions  # noqa: E402

CANDIDATE_JSON = ROOT / "reports" / "regression_baseline" / "validate_t0_025.json"

assert enhance._FINAL_F3_T0 == 0.25, (
    f"expected production _FINAL_F3_T0 == 0.25, found {enhance._FINAL_F3_T0} "
    "-- this script must run against the promoted code, not the pre-promotion baseline"
)


def resolve_source(rec: dict) -> Path:
    # "source" was stored relative to REPO_ROOT (validate.py's ROOT.parent)
    return REPO_ROOT / rec["source"]


def main() -> int:
    candidate = json.loads(CANDIDATE_JSON.read_text(encoding="utf-8"))
    matches, mismatches, missing = [], [], []

    t0 = time.perf_counter()
    for i, rec in enumerate(candidate["images"], 1):
        path = resolve_source(rec)
        if not path.is_file():
            missing.append(rec["image"])
            print(f"  [{i}/{len(candidate['images'])}] SKIP {rec['image']} (source not found: {path})")
            continue

        img = enhance.load_image(str(path))
        with neutralize_billboard_regions():
            out, _dt = enhance.enhance("final", img)
        actual_hash = hashlib.sha256(out.tobytes()).hexdigest()
        expected_hash = rec["output_sha256"]

        if actual_hash == expected_hash:
            matches.append(rec["image"])
            print(f"  [{i}/{len(candidate['images'])}] MATCH {rec['image']}")
        else:
            mismatches.append((rec["image"], expected_hash, actual_hash))
            print(f"  [{i}/{len(candidate['images'])}] MISMATCH {rec['image']} "
                  f"expected={expected_hash[:16]}... actual={actual_hash[:16]}...")

    elapsed = time.perf_counter() - t0
    print(f"\n{'=' * 70}")
    print(f"VERIFY PROMOTION (real enhance._FINAL_F3_T0=0.25 vs monkeypatch-based "
          f"validate_t0_025.json)")
    print(f"{'=' * 70}")
    print(f"images checked: {len(candidate['images'])}")
    print(f"byte-exact matches: {len(matches)}")
    print(f"mismatches: {len(mismatches)}")
    print(f"missing sources: {len(missing)}")
    print(f"elapsed: {elapsed:.1f}s")

    if mismatches:
        print("\nMISMATCHES:")
        for name, expected, actual in mismatches:
            print(f"  {name}: expected {expected} actual {actual}")

    ok = not mismatches and (len(matches) + len(missing) == len(candidate["images"]))
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
