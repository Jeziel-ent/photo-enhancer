import sys
import cv2
import torch
from pathlib import Path

sys.path.insert(0, "image_enhancer/src")

from pan_arch import PAN


MODEL = "image_enhancer/models/pan/pan_4x.pt"
IMAGE = r"D:\Ai-development\image-enhancer\image_enhancer\originals\3.jpeg"

OUTPUT_DIR = Path("pan_visual_tests")
OUTPUT_DIR.mkdir(exist_ok=True)

device = torch.device("cuda")

model = PAN(
    in_nc=3,
    out_nc=3,
    nf=40,
    unf=24,
    nb=16,
    scale=4,
).to(device).eval()

state = torch.load(MODEL, map_location="cpu")
model.load_state_dict(state, strict=True)

img = cv2.imread(IMAGE)

if img is None:
    raise RuntimeError(f"Could not read image: {IMAGE}")

h, w = img.shape[:2]

crops = {
    "top_left": (0, 0),
    "center": ((w - 256) // 2, (h - 256) // 2),
    "bottom_left": (0, h - 256),
    "bottom_right": (w - 256, h - 256),
}

for name, (x, y) in crops.items():
    crop = img[y:y + 256, x:x + 256]

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    tensor = (
        torch.from_numpy(rgb)
        .permute(2, 0, 1)
        .float()
        .div(255.0)
        .unsqueeze(0)
        .to(device)
    )

    with torch.inference_mode():
        output = model(tensor).clamp(0, 1)

    result = (
        output[0]
        .permute(1, 2, 0)
        .cpu()
        .numpy()
        * 255.0
    )

    result = result.round().astype("uint8")
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(OUTPUT_DIR / f"{name}_original.png"), crop)
    cv2.imwrite(str(OUTPUT_DIR / f"{name}_pan.png"), result)

    print(
        name,
        "original:", crop.shape,
        "PAN:", result.shape,
        "finite:", torch.isfinite(output).all().item(),
    )

print("\nDONE")
print("Output folder:", OUTPUT_DIR.resolve())