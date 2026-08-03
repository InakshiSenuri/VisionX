"""
extract_bubbles.py
------------------
Extracts bubble crops from a comic page image using bounding boxes from the
VisionX result JSON. Saves each crop to an output folder named after the image.

Usage:
    python extract_bubbles.py <image_path> <json_path> [--padding 10] [--output_dir bubble_crops]

The JSON is expected to have:
  - "bubbles": list of dicts, each with:
      "id"         : int
      "order"      : int  (reading order)
      "panel_order": int
      "type"       : str  (speaker / thought / narrator)
      "box_pixel"  : [x_min, y_min, x_max, y_max]  absolute pixel coords
      "speaker_identity": str  (optional)
"""

import argparse
import json
import os
from pathlib import Path

from PIL import Image


def extract_bubbles(image_path: str,
                    json_path: str,
                    padding: int = 10,
                    output_dir: str = None) -> None:

    # ── Load image ────────────────────────────────────────────────────────────
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size
    print(f"[INFO] Image size: {img_w} x {img_h} px")

    # ── Load JSON ─────────────────────────────────────────────────────────────
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    bubbles = data.get("bubbles", [])
    if not bubbles:
        print("[WARN] No bubbles found in JSON.")
        return
    print(f"[INFO] Found {len(bubbles)} bubbles.")

    # ── Scale factor (JSON coords may be from a higher-res version) ───────────
    json_w = data.get("width",  img_w)
    json_h = data.get("height", img_h)
    scale_x = img_w / json_w
    scale_y = img_h / json_h
    if abs(scale_x - 1.0) > 0.01 or abs(scale_y - 1.0) > 0.01:
        print(f"[INFO] JSON reference size: {json_w}x{json_h} px  "
              f"→ scale ({scale_x:.4f}, {scale_y:.4f})")
    else:
        scale_x = scale_y = 1.0

    # ── Output folder ─────────────────────────────────────────────────────────
    if output_dir is None:
        stem = Path(image_path).stem          # e.g. "Suddi_image_001"
        output_dir = str(Path(image_path).parent / f"{stem}_bubble_crops")

    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Saving crops to: {output_dir}")

    # ── Sort bubbles by reading order ─────────────────────────────────────────
    bubbles_sorted = sorted(bubbles, key=lambda b: b.get("order", 9999))

    # ── Crop & save ───────────────────────────────────────────────────────────
    saved = 0
    for bubble in bubbles_sorted:
        bid        = bubble["id"]
        order      = bubble.get("order", "?")
        panel_ord  = bubble.get("panel_order", "?")
        btype      = bubble.get("type", "speaker")
        speaker    = bubble.get("speaker_identity", f"char_{bid:02d}")
        box        = bubble["box_pixel"]           # [x_min, y_min, x_max, y_max]

        x_min = int(round(box[0] * scale_x))
        y_min = int(round(box[1] * scale_y))
        x_max = int(round(box[2] * scale_x))
        y_max = int(round(box[3] * scale_y))

        # Apply padding, clamped to image boundaries
        x_min_p = max(0,     x_min - padding)
        y_min_p = max(0,     y_min - padding)
        x_max_p = min(img_w, x_max + padding)
        y_max_p = min(img_h, y_max + padding)

        crop = img.crop((x_min_p, y_min_p, x_max_p, y_max_p))

        # File name: order_panel_type_speaker_id.png
        filename = (f"order{order:02d}_panel{panel_ord:02d}_"
                    f"{btype}_{speaker}_id{bid:02d}.png")
        save_path = os.path.join(output_dir, filename)
        crop.save(save_path)
        saved += 1
        print(f"  [{order:>2}] Saved: {filename}  ({x_max-x_min}x{y_max-y_min} px)")

    print(f"\n[DONE] {saved}/{len(bubbles)} bubble crops saved to '{output_dir}'")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract bubble crops from VisionX JSON result.")
    parser.add_argument("image_path", help="Path to the comic page image (jpg/png)")
    parser.add_argument("json_path",  help="Path to the VisionX result JSON file")
    parser.add_argument("--padding",    type=int, default=10,
                        help="Extra pixels to add around each crop (default: 10)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output folder (default: <image_stem>_bubble_crops/)")
    args = parser.parse_args()

    extract_bubbles(
        image_path=args.image_path,
        json_path=args.json_path,
        padding=args.padding,
        output_dir=args.output_dir,
    )