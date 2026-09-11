"""Generate a synthetic test scene approximating an Adinn billboard photo.

Only used to smoke-test the enhancement pipeline before real reference
images are provided. NOT part of the benchmark dataset.
Content: sky+billboard (text/logo), buildings, road, a vehicle, a person.
Adds mild noise + blur to simulate a real-signage photo.
"""
import cv2
import numpy as np


def make_scene(w=1280, h=720, seed=42):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 135, np.uint8)  # mid-gray sky

    # gradient sky
    for y in range(h):
        v = int(200 - 60 * (y / h))
        img[y, :, :] = v

    # billboard (upper area) with border
    bb_x, bb_y, bb_w, bb_h = 150, 90, 720, 260
    cv2.rectangle(img, (bb_x, bb_y), (bb_x + bb_w, bb_y + bb_h), (30, 30, 30), -1)
    cv2.rectangle(img, (bb_x + 8, bb_y + 8), (bb_x + bb_w - 8, bb_y + bb_h - 8),
                  (240, 245, 250), -1)
    # billboard artwork block (colorful rectangle)
    cv2.rectangle(img, (bb_x + 60, bb_y + 40), (bb_x + 420, bb_y + 180),
                  (200, 20, 190), -1)
    cv2.rectangle(img, (bb_x + 440, bb_y + 40), (bb_x + 640, bb_y + 120),
                  (30, 120, 210), -1)
    # advertising text
    cv2.putText(img, "ADINN", (bb_x + 120, bb_y + 140),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(img, "FRESH & TASTY", (bb_x + 120, bb_y + 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    # small product/logo circle (simulating a logo)
    cv2.circle(img, (bb_x + 550, bb_y + 200), 30, (40, 180, 40), -1)

    # buildings along horizon
    for i, (bx, bw, bh) in enumerate([
            (0, 180, 120), (200, 140, 90), (950, 160, 100),
            (1130, 200, 130)]):
        cv2.rectangle(img, (bx, h - 210), (bx + bw, h - 90), (110, 110, 115), -1)
        for winx in range(bx + 14, bx + bw - 10, 22):
            for winty in range(h - 195, h - 105, 24):
                cv2.rectangle(img, (winx, winty), (winx + 10, winty + 12),
                              (70, 180, 220) if rng.random() > 0.4 else (60, 60, 70), -1)

    # road
    cv2.rectangle(img, (0, h - 90), (w, h), (60, 60, 65), -1)
    for i in range(0, w, 60):
        cv2.rectangle(img, (i, h - 26), (i + 24, h - 18), (200, 200, 205), -1)

    # vehicle (a car) on road
    car_x, car_y = 760, h - 90 - 40
    cv2.rectangle(img, (car_x, car_y), (car_x + 150, car_y + 40), (30, 40, 200), -1)
    cv2.rectangle(img, (car_x + 130, car_y - 22), (car_x + 210, car_y + 40),
                  (30, 40, 200), -1)
    for wx in (car_x + 30, car_x + 80, car_x + 140):
        cv2.rectangle(img, (wx, car_y - 2), (wx + 28, car_y + 16),
                      (210, 210, 220), -1)
    for cx in (car_x + 26, car_x + 150):
        cv2.circle(img, (cx, car_y + 40), 14, (30, 30, 30), -1)

    # pedestrian
    px, py = 300, h - 90 - 52
    cv2.circle(img, (px, py), 10, (40, 60, 160), -1)          # head
    cv2.rectangle(img, (px - 8, py + 10), (px + 8, py + 38), (60, 90, 40), -1)  # body
    cv2.line(img, (px - 8, py + 26), (px - 14, py + 46), (20, 20, 20), 3)      # leg
    cv2.line(img, (px + 8, py + 26), (px + 14, py + 46), (20, 20, 20), 3)
    cv2.line(img, (px + 8, py + 14), (px + 20, py + 20), (200, 180, 140), 3)   # arm

    # signed lamp posts (street furniture)
    for lpx in (120, 1000):
        cv2.rectangle(img, (lpx, h - 90 - 150), (lpx + 6, h - 90), (80, 80, 85), -1)
        cv2.circle(img, (lpx + 3, h - 90 - 150), 6, (80, 80, 85), -1)

    # blur + noise to simulate real-signage photo
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    noise = rng.normal(0, 12, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # slight exposure / white-balance imperfection
    img = cv2.convertScaleAbs(img, alpha=1.04, beta=-8)
    return img


if __name__ == "__main__":
    import os
    out_dir = os.environ.get("SYNTH_DIR", ".")
    os.makedirs(out_dir, exist_ok=True)
    for seed in (1, 2, 3):
        img = make_scene(seed=seed)
        cv2.imwrite(os.path.join(out_dir, f"synth_adinn_{seed}.png"), img)
    print("generated 3 synthetic test scenes")
