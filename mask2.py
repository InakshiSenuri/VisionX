"""
Sinhala Comic Bubble — Preprocessing v1 + 2 fixes only
-------------------------------------------------------
Based on your ORIGINAL v1 that worked best.

Only two changes from v1:
  Fix 1 — Removed GaussianBlur before Sauvola (was causing blurry letters)
  Fix 2 — Replaced morphological open with CC area filter for noise removal
           (safer for small Sinhala diacritics like ්  ා  ෙ)

Everything else is IDENTICAL to your original working v1.
"""

import cv2
import numpy as np
import os
import sys
from skimage.filters import threshold_sauvola


def find_bubble_interior(gray):
    _, bright_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bright_mask, connectivity=8
    )
    if num_labels <= 1:
        return None

    img_area = gray.shape[0] * gray.shape[1]
    best_label = -1
    best_area  = 0

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if best_area < area < img_area * 0.8:
            best_area  = area
            best_label = label

    if best_label == -1:
        return None

    mask = np.zeros_like(gray)
    mask[labels == best_label] = 255

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        largest     = max(contours, key=cv2.contourArea)
        filled_mask = np.zeros_like(gray)
        cv2.drawContours(filled_mask, [largest], -1, 255, -1)
        return filled_mask

    return mask


def sauvola_binarize(gray):
    """FIX 1: window slightly larger (29 vs 25) to compensate for no blur."""
    thresh = threshold_sauvola(gray, window_size=29, k=0.2)
    return (gray > thresh).astype(np.uint8) * 255


def remove_noise_cc(binary, min_area=20):
    """FIX 2: CC area filter instead of morphological open."""
    inv = cv2.bitwise_not(binary)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    out = np.zeros_like(inv)
    for lbl in range(1, num):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            out[labels == lbl] = 255
    return cv2.bitwise_not(out)

def classify_bubble_type(binary):
    H, W = binary.shape
    inv = cv2.bitwise_not(binary)

    h_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(20, W // 4), 1)
    )
    h_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, h_kernel)

    v_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(20, H // 4))
    )
    v_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, v_kernel)

    h_ink = np.sum(h_lines > 0)
    v_ink = np.sum(v_lines > 0)
    total_ink = np.sum(inv > 0)

    if total_ink == 0:
        return 'organic'

    h_ratio = h_ink / total_ink
    v_ratio = v_ink / total_ink

    print(f"    h_ratio={h_ratio:.4f} v_ratio={v_ratio:.4f}", end=' ')

    # both must be present — a rectangle needs both H and V lines
    # threshold very low — just needs to be non-zero
    if h_ratio > 0.005 and v_ratio > 0.005:
        return 'bordered'
    return 'organic'


def remove_bubble_border(binary):
    """For bordered bubbles — remove rectangular frame."""
    H, W = binary.shape
    inv = cv2.bitwise_not(binary)

    h_len = max(20, W // 4)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    horizontal_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, h_kernel)

    v_len = max(20, H // 4)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    vertical_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, v_kernel)

    border_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
    dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    border_mask = cv2.dilate(border_mask, dilate_k, iterations=4)

    cleaned = inv.copy()
    cleaned[border_mask > 0] = 0
    return cv2.bitwise_not(cleaned)

def find_text_bottom(binary, density_threshold=0.008):
    """
    Scan rows top-to-bottom to find where text ends.
    Returns the y-coordinate of the last text row.
    Used to cut off art that bleeds in below text.
    """
    H, W = binary.shape
    inv = cv2.bitwise_not(binary)
    row_densities = np.sum(inv > 0, axis=1) / W

    # find last row with significant ink — that's the text bottom
    text_rows = np.where(row_densities > density_threshold)[0]

    if len(text_rows) == 0:
        return H

    last_text_row = int(text_rows[-1])

    # add small buffer for descenders
    return min(H, last_text_row + 15)


def remove_outlier_art(binary, min_char_area=15, max_char_area_ratio=0.20):
    import math
    H, W = binary.shape
    img_area = H * W
    diag = math.sqrt(H**2 + W**2)

    # ── Step A: find where text ends vertically ───────────────
    text_bottom = find_text_bottom(binary)

    # if art is clearly below text, cut it
    # only apply if cut is not too aggressive (keeps at least 40% of image)
    if text_bottom < H * 0.85:
        binary = binary.copy()
        binary[text_bottom:, :] = 255

    # ── Step B: centre-of-mass filter for remaining art ───────
    inv = cv2.bitwise_not(binary)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        inv, connectivity=8
    )
    if num <= 1:
        return binary

    valid = []
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        w    = stats[i, cv2.CC_STAT_WIDTH]
        h    = stats[i, cv2.CC_STAT_HEIGHT]
        if area < min_char_area:
            continue
        if area > img_area * max_char_area_ratio:
            continue
        aspect = w / h if h > 0 else 999
        if aspect > 10 or aspect < 0.1:
            continue
        valid.append({
            'idx': i,
            'cx': centroids[i][0],
            'cy': centroids[i][1],
            'area': area
        })

    if not valid:
        return binary

    total_area = sum(c['area'] for c in valid)
    cx_mean = sum(c['cx'] * c['area'] for c in valid) / total_area
    cy_mean = sum(c['cy'] * c['area'] for c in valid) / total_area

    # tighter threshold than before — 45% of diagonal
    threshold = diag * 0.45

    keep_labels = set()
    for c in valid:
        dist = math.sqrt((c['cx'] - cx_mean)**2 + (c['cy'] - cy_mean)**2)
        if dist <= threshold:
            keep_labels.add(c['idx'])

    out = np.zeros_like(inv)
    for lbl in keep_labels:
        out[labels == lbl] = 255

    return cv2.bitwise_not(out)


def crop_to_text(binary, padding=12):
    inv = cv2.bitwise_not(binary)
    coords = np.where(inv > 0)
    if len(coords[0]) == 0:
        return binary
    y_min = max(0, coords[0].min() - padding)
    y_max = min(binary.shape[0], coords[0].max() + padding)
    x_min = max(0, coords[1].min() - padding)
    x_max = min(binary.shape[1], coords[1].max() + padding)
    return binary[y_min:y_max, x_min:x_max]

def preprocess_dialogue_bubble(img_path):
    # ── Step 1: Load ─────────────────────────────────────────
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")

    # ── Step 2: Flatten alpha → white ────────────────────────
    if len(img.shape) == 3 and img.shape[2] == 4:
        b, g, r, alpha = cv2.split(img)
        white = np.ones_like(b) * 255
        a = alpha / 255.0
        b = (b * a + white * (1 - a)).astype(np.uint8)
        g = (g * a + white * (1 - a)).astype(np.uint8)
        r = (r * a + white * (1 - a)).astype(np.uint8)
        img = cv2.merge([b, g, r])

    # ── Step 3: Grayscale ─────────────────────────────────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── Step 4: Upscale if small ──────────────────────────────
    h, w = gray.shape
    if h < 200 or w < 200:
        scale = max(200 / h, 200 / w)
        gray  = cv2.resize(gray, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_CUBIC)

    # ── Step 5: Find bubble interior mask ────────────────────
    bubble_mask = find_bubble_interior(gray)

    # ── Step 6: Apply mask — everything outside → white ──────
    if bubble_mask is not None:
        masked = np.ones_like(gray) * 255
        masked[bubble_mask == 255] = gray[bubble_mask == 255]
        gray = masked

    # ── Step 7: [REMOVED] GaussianBlur — was causing blurry text

    # ── Step 8: Sauvola binarisation ─────────────────────────
    binary = sauvola_binarize(gray)

    # ── Step 9: Ensure black text on white ───────────────────
    if np.sum(binary == 0) > np.sum(binary == 255):
        binary = cv2.bitwise_not(binary)

    # ── Step 10: CC noise removal (replaces morphological open)
    cleaned = remove_noise_cc(binary, min_area=20)

    # ── Step 11: Style-aware art removal + crop ───────────────
    bubble_type = classify_bubble_type(cleaned)
    
    # ── print classification result ───────────────────────────
    print(f"  [{bubble_type.upper()}] {os.path.basename(img_path)}")
    
    if bubble_type == 'bordered':
        cleaned = remove_bubble_border(cleaned)
    else:
        cleaned = remove_outlier_art(cleaned)
    
    cleaned = crop_to_text(cleaned, padding=12)
    return cleaned


def process_dialogue_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    files = sorted([f for f in os.listdir(input_folder)
                    if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

    success, failed = 0, []
    print(f"Processing {len(files)} dialogue bubbles...\n")

    for filename in files:
        inp = os.path.join(input_folder, filename)
        out = os.path.join(output_folder, filename)
        try:
            result = preprocess_dialogue_bubble(inp)
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
    if len(sys.argv) == 3:
        process_dialogue_folder(sys.argv[1], sys.argv[2])
    else:
        process_dialogue_folder(
            r"my_crops",
            r"my_preprocessed"
        )

