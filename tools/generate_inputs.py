#!/usr/bin/env python3
"""Create synthetic PGM inputs and PNG previews + CPU reference outputs."""
import os
import csv
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "data" / "input"
OUT_DIR = ROOT / "data" / "output"
ART = ROOT / "artifacts"
IN_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)


def write_pgm(path: Path, arr: np.ndarray) -> None:
    h, w = arr.shape
    with open(path, "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
        f.write(arr.astype(np.uint8).tobytes())


def box_filter(img: np.ndarray, k: int) -> np.ndarray:
    r = k // 2
    h, w = img.shape
    acc = np.zeros((h, w), dtype=np.float64)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            yy = np.clip(np.arange(h) + dy, 0, h - 1)
            xx = np.clip(np.arange(w) + dx, 0, w - 1)
            acc += img[np.ix_(yy, xx)]
    return np.clip(acc / (k * k) + 0.5, 0, 255).astype(np.uint8)


def make_image(idx: int, h: int, w: int) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w), dtype=np.float64)
    img += 40 + 80 * (x / w)
    cx, cy = w * (0.3 + 0.1 * (idx % 3)), h * (0.4 + 0.1 * ((idx // 2) % 2))
    img += 120 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * (min(h, w) / 6) ** 2))
    img += 25 * np.sin(x / 12.0 + idx) * np.cos(y / 15.0)
    rng = np.random.default_rng(idx + 7)
    img += rng.normal(0, 8, size=img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def main() -> None:
    sizes = [(256, 256), (384, 256), (512, 384), (640, 480), (320, 320), (480, 320)]
    kernel = 5
    rows = []
    for i, (h, w) in enumerate(sizes, start=1):
        name = f"scene_{i:02d}_{w}x{h}.pgm"
        img = make_image(i, h, w)
        write_pgm(IN_DIR / name, img)
        Image.fromarray(img).save(ART / f"before_{name.replace('.pgm', '.png')}")
        filt = box_filter(img, kernel)
        write_pgm(OUT_DIR / f"filtered_{name}", filt)
        Image.fromarray(filt).save(ART / f"after_{name.replace('.pgm', '.png')}")
        rows.append((name, w, h, w * h, 0.0))
        print(f"wrote {name} {w}x{h}")

    # placeholder timings; replace after GPU run
    with open(OUT_DIR / "timings.csv", "w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["file", "width", "height", "bytes", "ms"])
        for r in rows:
            wri.writerow(r)
        wri.writerow(["TOTAL", "", "", "", 0.0])

    log = OUT_DIR / "run.log"
    log.write_text(
        "NOTE: Replace this log by running ./run.sh on a machine with an NVIDIA GPU.\n"
        "GPU: (run bin/box_filter to fill device name)\n"
        f"files={len(sizes)}  kernel={kernel}  streams=4\n"
        + "\n".join(f"{r[0]}  {r[1]}x{r[2]}  (cpu reference written)" for r in rows)
        + "\n"
    )
    print("done. inputs in data/input, previews in artifacts/")


if __name__ == "__main__":
    main()
