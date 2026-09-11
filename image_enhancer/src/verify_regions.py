"""Manual billboard-region verification for the C (billboard) pipeline.

Automatic detection is NOT used here: generic rectangle detection is
unreliable for OOH billboard photos (it latches onto roads, sky, vehicles
and unrelated rectangles). Billboards therefore use MANUALLY VERIFIED bounding
boxes stored in `regions.json`.

A single source photo may contain SEVERAL billboards. Each image entry in
`regions.json` holds a list of boxes; only the relevant Adinn board(s) are
selected manually — not every detected rectangle.

Schema (per image):
  "1.jpeg": { "boxes": [[x, y, w, h], ...], "image_size": [W, H] }

Workflow:
  1. python src\\verify_regions.py               -> review montages with the
     current boxes drawn on the originals (reports/billboard_review/).
  2. Visually check each overlay. Two ways to correct boxes:
       a. edit regions.json by hand, then re-run step 1 to re-check, or
       b. python src\\verify_regions.py --mode interactive
          (drag to draw a box; keys described in the editor window).
  3. ONLY after every box is verified, benchmark method C:
       python src\\benchmark.py --methods billboard repair

No detector is invoked anywhere in this file.
"""

import argparse
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).parent.parent
ORIGINALS = ROOT / "originals"
REPORTS = ROOT / "reports"
REVIEW = REPORTS / "billboard_review"
CONFIG = ROOT / "regions.json"

DISPLAY_MAX = 1200  # longest display side in px for review/interactive windows


def supported_image(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_config():
    if CONFIG.exists():
        text = CONFIG.read_text(encoding="utf-8-sig")
        return json.loads(text) if text.strip() else {}
    return {}


def save_config(cfg):
    CONFIG.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def _entry_boxes(entry):
    """Normalize a config entry to a list of (x, y, w, h) boxes. Prefers the
    multi-box 'boxes' key; tolerates the legacy single 'box' key."""
    if not entry:
        return []
    raw = entry.get("boxes")
    if not raw:
        raw = [entry.get("box")] if entry.get("box") else []
    boxes = []
    for b in raw:
        if not b or any(v is None for v in b):
            continue
        try:
            boxes.append(tuple(int(v) for v in b))
        except (TypeError, ValueError):
            continue
    return boxes


def find_boxes(cfg, img_p: Path, W: int, H: int):
    """Look up the manually verified boxes for an image: exact filename first,
    then a config entry whose image_size matches (single-image-set fallback)."""
    if img_p.name in cfg:
        boxes = _entry_boxes(cfg[img_p.name])
        if boxes:
            return boxes
    matches = [k for k, v in cfg.items() if v.get("image_size") == [W, H]]
    if len(matches) == 1:
        return _entry_boxes(cfg[matches[0]])
    return []


# ------------------------------------------------------------------- montage

def draw_boxes(vis, boxes):
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 3)
        label = f"#{i + 1} x={x} y={y} w={w} h={h}"
        ty = max(22, y - 12 - i * 18)
        cv2.putText(vis, label, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2)


def make_review_montage(img_p: Path, cfg):
    img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
    if img is None:
        print(f"  !! cannot read {img_p.name}, skipping")
        return
    H, W = img.shape[:2]
    boxes = find_boxes(cfg, img_p, W, H)

    vis = img.copy()
    if boxes:
        draw_boxes(vis, boxes)
    else:
        cv2.putText(vis, "NO MANUAL BOXES", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    scale = DISPLAY_MAX / max(vis.shape[1], vis.shape[0], 1)
    vis = cv2.resize(vis, (int(vis.shape[1] * scale), int(vis.shape[0] * scale)),
                     interpolation=cv2.INTER_AREA)
    out_p = REVIEW / f"review_{img_p.stem}.png"
    cv2.imwrite(str(out_p), vis)
    n = len(boxes)
    status = f"{n} box(es)" if boxes else "MISSING"
    print(f"{img_p.name}: {status}  ->  {out_p.relative_to(ROOT)}")


def montage_mode():
    REVIEW.mkdir(exist_ok=True)
    cfg = load_config()
    images = [p for p in sorted(ORIGINALS.iterdir()) if supported_image(p)]
    if not images:
        print(f"No images found in {ORIGINALS}")
        return
    for p in images:
        make_review_montage(p, cfg)
    print("\nReview montages written to", REVIEW)
    print("Edit regions.json by hand (or run --mode interactive), then re-run")
    print("this command to re-generate the overlay. Benchmark C only after")
    print("every box is verified.")


# -------------------------------------------------------------- interactive

class _Editor:
    def __init__(self, img, boxes, title):
        self.img = img
        self.H, self.W = img.shape[:2]
        scale = DISPLAY_MAX / max(self.W, self.H, 1)
        self.scale = min(scale, 1.0)
        self.show = cv2.resize(img, (int(self.W * scale), int(self.H * scale)),
                               interpolation=cv2.INTER_AREA)
        self.title = title
        self.orig_boxes = [list(b) for b in boxes]
        self.boxes = [list(b) for b in boxes]
        self.pending = None   # green new box being dragged (orig coords)
        self.sel = None       # highlighted box index (for delete/replace)
        self.drag = None      # drag anchor in display coords while drawing
        self.cursor = (0, 0)

    def to_disp(self, x, y, w, h):
        return (int(x * self.scale), int(y * self.scale),
                int(w * self.scale), int(h * self.scale))

    def to_orig(self, x, y, w, h):
        x = max(0, min(self.W - 1, int(round(x / self.scale))))
        y = max(0, min(self.H - 1, int(round(y / self.scale))))
        w = min(self.W - x, max(20, int(round(w / self.scale))))
        h = min(self.H - y, max(15, int(round(h / self.scale))))
        return (x, y, w, h)

    def render(self):
        vis = self.show.copy()
        for i, (x, y, w, h) in enumerate(self.boxes):
            color = (255, 200, 0) if i == self.sel else (0, 0, 255)
            dx, dy, dw, dh = self.to_disp(x, y, w, h)
            cv2.rectangle(vis, (dx, dy), (dx + dw, dy + dh), color, 3)
            cv2.putText(vis, f"#{i + 1}", (dx, max(22, dy - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if self.pending:
            dx, dy, dw, dh = self.to_disp(*self.pending)
            cv2.rectangle(vis, (dx, dy), (dx + dw, dy + dh), (0, 255, 0), 2)
        cv2.putText(vis, f"boxes={len(self.boxes)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return vis

    def on_mouse(self, event, posx, posy, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag = (posx, posy)
            self.cursor = (posx, posy)
        elif event == cv2.EVENT_MOUSEMOVE:
            self.cursor = (posx, posy)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drag:
                x0, y0 = self.drag
                x1, y1 = self.cursor
                l, t = min(x0, x1), min(y0, y1)
                r, b = max(x0, x1), max(y0, y1)
                self.pending = list(self.to_orig(l, t, r - l, b - t))
                self.drag = None


def edit_image(img_p: Path, cfg):
    img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
    if img is None:
        return None
    H, W = img.shape[:2]
    boxes = find_boxes(cfg, img_p, W, H)

    ed = _Editor(img, boxes, img_p.name)
    WIN = ("billboard boxes | drag=draw box  Enter=add for selected/Space "
           "Tab=select  x=delete  r=reset | s=save+next  q=quit+save  "
           "Esc=quit discard")
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1100, 700)
    cv2.setMouseCallback(WIN, ed.on_mouse)
    print(f"\n== {img_p.name}  (start from {len(boxes)} box(es) {boxes})")

    while True:
        cv2.imshow(WIN, ed.render())
        key = cv2.waitKey(40) & 0xFF
        if key in (13, 10, 32):  # Enter/Space: add pending box (or replace selected)
            if ed.pending:
                if ed.sel is not None:
                    ed.boxes[ed.sel] = list(ed.pending)
                else:
                    ed.boxes.append(list(ed.pending))
                ed.pending = None
                ed.sel = None
        elif key in (27, ord("q")):  # Esc quits discarding; q quits + saves
            if key == ord("q"):
                ed.quit_after = True
            else:
                ed.quit_after = True
                ed.discard = True
            break
        elif key == ord("s"):
            break
        elif key in (ord("x"), ord("d")):  # delete selected, else last
            if ed.pending:
                ed.pending = None
            elif ed.sel is not None:
                ed.boxes.pop(ed.sel)
                ed.sel = None
            elif ed.boxes:
                ed.boxes.pop()
        elif key == 9:  # Tab: cycle selection
            if ed.boxes:
                ed.sel = 0 if ed.sel is None else (ed.sel + 1) % len(ed.boxes)
        elif key == ord("r"):  # reset to original config boxes
            ed.boxes = [list(b) for b in ed.orig_boxes]
            ed.pending = None
            ed.sel = None

    cv2.destroyWindow(WIN)
    return ed


def interactive_mode():
    cfg = load_config()
    images = [p for p in sorted(ORIGINALS.iterdir()) if supported_image(p)]
    if not images:
        print(f"No images found in {ORIGINALS}")
        return

    editors = []
    prereqs = getattr(cv2, "__version__", "")
    print(f"Interactive box editor (OpenCV {prereqs})  --  drag to draw a box,")
    print("Enter adds it (or replaces the Tab-selected box), x deletes,")
    print("r resets to config, s saves + next, q quits + saves, Esc cancels.")
    try:
        for p in images:
            ed = edit_image(p, cfg)
            if ed is None:
                print(f"  !! cannot read {p.name}, skipping")
                continue
            editors.append((p, ed))
            if getattr(ed, "quit_after", False):
                break
    except Exception as ex:  # e.g. no GUI / headless
        print(f"\nInteractive mode unavailable: {ex}")
        print("-> edit regions.json by hand instead, then run:",
              "python src\\verify_regions.py")
        return

    dirty = False
    for p, ed in editors:
        discard = getattr(ed, "discard", False)
        if not discard:
            cfg[p.name] = {"boxes": [[int(v) for v in b] for b in ed.boxes],
                           "image_size": [ed.W, ed.H]}
            dirty = True
    if dirty:
        save_config(cfg)
        print(f"\nUpdated {CONFIG.relative_to(ROOT)}")
    else:
        print("\nNo boxes saved.")

    print("Regenerate review montages:")
    print("  python src\\verify_regions.py")


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["montage", "interactive"], default="montage",
                    help="montage: draw current config boxes for review; "
                         "interactive: adjust boxes on-screen and save them")
    args = ap.parse_args()

    if args.mode == "interactive":
        interactive_mode()
    else:
        montage_mode()


if __name__ == "__main__":
    main()