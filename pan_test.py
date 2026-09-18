import sys
import cv2
import torch

sys.path.insert(0, "image_enhancer/src")

from pan_arch import PAN


MODEL = "image_enhancer/models/pan/pan_4x.pt"
IMAGE = r"D:\Ai-development\image-enhancer\image_enhancer\originals\3.jpeg"
OUTPUT = r"D:\Ai-development\image-enhancer\pan_test_3_tile.png"


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

crop = img[:256, :256]

rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

x = (
    torch.from_numpy(rgb)
    .permute(2, 0, 1)
    .float()
    .div(255.0)
    .unsqueeze(0)
    .to(device)
)

with torch.inference_mode():
    y = model(x).clamp(0, 1)

out = (
    y[0]
    .permute(1, 2, 0)
    .cpu()
    .numpy()
    * 255.0
)

out = out.round().astype("uint8")
out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

cv2.imwrite(OUTPUT, out)

print("SAVED:", OUTPUT)
print("INPUT:", crop.shape)
print("OUTPUT:", out.shape)
print("FINITE:", torch.isfinite(y).all().item())