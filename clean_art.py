# file: clean_art.py

import cv2
import numpy as np
import os
from pathlib import Path


def find_crop_bounds(binary, 
                     row_threshold=0.02,
                     col_threshold=0.02,
                     min_text_height_ratio=0.15):
    """
    Find tight crop bounds by scanning rows and columns
    for ink density.
    
    A row/column is considered 'text' if ink density > threshold.
    We find the contiguous block of text rows, ignoring
    isolated sparse rows that are likely art strokes.
    """
    H, W = binary.shape
    inv = cv2.bitwise_not(binary)
    
    # ink density per row and column
    row_density = np.sum(inv > 0, axis=1) / W
    col_density = np.sum(inv > 0, axis=0) / H
    
    # ── find text rows ────────────────────────────────────────
    text_rows = np.where(row_density > row_threshold)[0]
    
    if len(text_rows) == 0:
        return None
    
    # find the largest contiguous block of text rows
    # this handles cases where art rows interrupt text
    blocks = []
    start = text_rows[0]
    prev  = text_rows[0]
    
    for r in text_rows[1:]:
        if r - prev > 8:  # gap of 8+ empty rows = new block
            blocks.append((start, prev))
            start = r
        prev = r
    blocks.append((start, prev))
    
    # pick the tallest block — that's the text
    tallest = max(blocks, key=lambda b: b[1] - b[0])
    y_min, y_max = tallest
    
    # reject if text block is too small (likely just art noise)
    if (y_max - y_min) < H * min_text_height_ratio:
        # fallback: use all text rows
        y_min, y_max = text_rows[0], text_rows[-1]
    
    # ── find text columns ─────────────────────────────────────
    # only look within the text row range
    col_density_in_text = np.sum(inv[y_min:y_max, :] > 0, axis=0) / max(1, y_max - y_min)
    text_cols = np.where(col_density_in_text > col_threshold)[0]
    
    if len(text_cols) == 0:
        x_min, x_max = 0, W
    else:
        x_min, x_max = text_cols[0], text_cols[-1]
    
    return y_min, y_max, x_min, x_max


def clean_and_crop(binary, padding=10):
    H, W = binary.shape
    inv = cv2.bitwise_not(binary)
    row_density = np.sum(inv > 0, axis=1) / W

    # raised from 0.015 to 0.025 — thin tail strokes won't qualify
    text_threshold = 0.025

    text_rows = np.where(row_density > text_threshold)[0]

    cut_top    = 0
    cut_bottom = H

    if len(text_rows) > 0:
        gaps = []
        for i in range(len(text_rows) - 1):
            gap_size  = text_rows[i+1] - text_rows[i]
            gap_start = text_rows[i]
            gap_end   = text_rows[i+1]
            if gap_size > 8:
                gaps.append((gap_size, gap_start, gap_end))

        if gaps:
            biggest = max(gaps, key=lambda x: x[0])
            gap_size, gap_start, gap_end = biggest

            rows_above = np.sum(row_density[:gap_start] > text_threshold)
            rows_below = np.sum(row_density[gap_end:]   > text_threshold)

            # only cut if one side clearly dominates
            # if both sides are similar — don't cut (avoid mid-text cuts)
            ratio = max(rows_above, rows_below) / max(1, min(rows_above, rows_below))

            if ratio < 1.5:
                pass  # too ambiguous — don't cut anything
            elif rows_above >= rows_below:
                cut_bottom = gap_start
            else:
                cut_top = gap_end

    binary = binary.copy()
    if cut_top > 0:
        binary[:cut_top, :] = 255
    if cut_bottom < H:
        binary[cut_bottom:, :] = 255

    inv2   = cv2.bitwise_not(binary)
    coords = np.where(inv2 > 0)

    if len(coords[0]) == 0:
        return binary

    y_min = max(0, coords[0].min() - padding)
    y_max = min(H, coords[0].max() + padding)
    x_min = max(0, coords[1].min() - padding)
    x_max = min(W, coords[1].max() + padding)

    return binary[y_min:y_max, x_min:x_max]

def process_folder(input_folder, output_folder):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    files = sorted([f for f in os.listdir(input_folder)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"Cleaning {len(files)} masked bubbles...\n")
    success, failed = 0, []
    
    for filename in files:
        inp = os.path.join(input_folder, filename)
        out = os.path.join(output_folder, filename)
        
        try:
            binary = cv2.imread(inp, cv2.IMREAD_GRAYSCALE)
            if binary is None:
                raise FileNotFoundError(f"Cannot read: {inp}")
            
            result = clean_and_crop(binary)
            cv2.imwrite(out, result)
            success += 1
            print(f"  ✓ {filename} → {result.shape[1]}x{result.shape[0]}px")
            
        except Exception as e:
            failed.append(filename)
            print(f"  ✗ {filename} — {e}")
    
    print(f"\nDone: {success} success, {len(failed)} failed")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    process_folder(
        input_folder=r'my_preprocessed',
        output_folder=r'my_cleaned'
    )