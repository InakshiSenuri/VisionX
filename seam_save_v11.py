"""
sinhalaLineSeg_seam_v11.py
=========================
Three-projection consensus + adaptive width + v3 DP seam.

Pipeline
--------
  Image
    ↓
  Three projections (adaptive width):
    - Global  (full width)
    - Left    (min(150px, 45% of W))
    - Right   (min(150px, 45% of W))
    ↓
  Each projection finds its own valleys independently
    ↓
  Merge nearby valleys (±6 rows = same physical gap)
    ↓
  Confidence score per merged valley:
    3 = global + left + right all agree  (definitely a cut)
    2 = two projections agree            (likely a cut)
    1 = only one projection saw it       (possible cut, use cautiously)
    ↓
  Keep all valleys with confidence >= MIN_CONFIDENCE
    ↓
  v3 DP seam per valley (full width, improved cost function)
    ↓
  Line crops with confidence metadata

Why three projections
---------------------
  Global alone: misses cuts blocked by modifiers spanning full width
  Left alone:   fails when art is on the left side
  Right alone:  fails when art is on the right side
  Together:     art rarely blocks ALL three simultaneously

Adaptive width
--------------
  min(150px, 45% of W) means:
    small bubbles  → proportional strip (no info lost)
    large bubbles  → capped at 150px (focused on edge region)
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict


# ══════════════════════════════════════════════════════════════
#  PARAMETERS
# ══════════════════════════════════════════════════════════════

PROJ_FRAC      = 0.45    # fraction of width for left/right strips
PROJ_MAX_PX    = 150     # cap on strip width in pixels
MERGE_RADIUS    = 6    # valleys within this many rows = same gap
MIN_CONFIDENCE  = 2    # minimum votes to keep (2=two projections must agree)
MIN_LINE_HEIGHT = 18   # minimum rows between consecutive cuts
                       # cuts closer than this = modifier gap, not line gap
                       # tune this based on your bubble font size
SMOOTH_K       = 9       # smoothing kernel size

# DP seam weights (v3 — unchanged)
# Note: INK_WEIGHT and DIST_WEIGHT overridden in build_cost_map v8
INK_WEIGHT    = 100.0  # v8: much higher
DIST_WEIGHT   =   4.0
VALLEY_WEIGHT =   3.0  # kept for reference
BEND_WEIGHT   =   1.0
SEARCH_RADIUS = 15
MAX_STEP      =  2
PADDING       =  2


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def load_and_prepare(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    unique = np.unique(img)
    if len(unique) <= 5:
        binary = img.copy()
    else:
        _, binary = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.sum(binary == 0) < np.sum(binary == 255):
        binary = cv2.bitwise_not(binary)
    m = 5
    binary[:m,:]=0; binary[-m:,:]=0
    binary[:,:m]=0; binary[:,-m:]=0
    return binary


def smooth_1d(profile: np.ndarray, k: int = SMOOTH_K) -> np.ndarray:
    return np.convolve(profile, np.ones(k)/k, mode='same')


def adaptive_strip_width(W: int) -> int:
    """min(150px, 45% of image width) — adaptive projection width."""
    return min(PROJ_MAX_PX, int(W * PROJ_FRAC))



#  VALLEY DETECTION


def find_valleys_in(profile: np.ndarray,
                     half_w: int = 8,
                     min_depth: float = 0.40,
                     min_band_h: int = 15,
                     noise_ratio: float = 0.08) -> List[int]:
    """Find valley rows in a 1D projection profile."""
    max_ink = profile.max()
    if max_ink == 0: return []
    noise_thresh = max_ink * noise_ratio
    active = np.where(profile > noise_thresh)[0]
    if len(active) == 0: return []
    r0, r1 = int(active[0]), int(active[-1])
    if r1 - r0 < min_band_h * 2: return []

    valleys = []
    for i in range(r0+half_w, r1-half_w+1):
        val = profile[i]
        if val > profile[i-1] or val > profile[i+1]: continue
        left_pk  = profile[max(r0,i-half_w):i].max()
        right_pk = profile[i+1:min(r1+1,i+half_w+1)].max()
        local_pk = max(left_pk, right_pk)
        if local_pk <= 0: continue
        depth = (local_pk - val) / local_pk
        if depth >= min_depth:
            valleys.append((i, depth))

    if not valleys: return []
    valleys.sort(key=lambda x: x[0])
    suppressed = []
    i = 0
    while i < len(valleys):
        cluster = [valleys[i]]; j = i+1
        while j<len(valleys) and valleys[j][0]-cluster[0][0]<half_w:
            cluster.append(valleys[j]); j+=1
        suppressed.append(max(cluster, key=lambda x: x[1])[0])
        i = j
    return suppressed



#  THREE-PROJECTION CONSENSUS

@dataclass
class ValleyVote:
    row:        int
    confidence: int          # 1, 2, or 3
    sources:    List[str]    # which projections voted for this row
    depth_max:  float = 0.0  # deepest valley depth among voters


def three_projection_consensus(binary: np.ndarray,
                                 min_confidence: int = MIN_CONFIDENCE,
                                 merge_radius: int = MERGE_RADIUS
                                 ) -> Tuple[List[ValleyVote],
                                            Dict[str, List[int]]]:
    """
    Run global, left, and right projections.
    Merge nearby valleys and count how many projections agree.

    Returns
    -------
    votes      : list of ValleyVote sorted by row
    proj_valleys: dict of raw valleys per projection (for visualisation)
    """
    H, W = binary.shape
    strip_w = adaptive_strip_width(W)

    # three projections
    strips = {
        'global': binary,
        'left':   binary[:, :strip_w],
        'right':  binary[:, W-strip_w:],
    }
    # global uses stricter min_depth (wider projection = clearer valleys)
    depths = {'global': 0.50, 'left': 0.40, 'right': 0.40}

    proj_valleys: Dict[str, List[int]] = {}
    for name, strip in strips.items():
        profile = smooth_1d(strip.sum(axis=1) / 255.0)
        proj_valleys[name] = find_valleys_in(
            profile, min_depth=depths[name])

    # collect all candidate rows from all projections
    # with their source label
    all_candidates = []
    for name, rows in proj_valleys.items():
        for r in rows:
            all_candidates.append((r, name))

    if not all_candidates:
        return [], proj_valleys

    # sort by row
    all_candidates.sort(key=lambda x: x[0])

    # merge candidates within merge_radius into single votes
    votes: List[ValleyVote] = []
    used = [False] * len(all_candidates)

    for i, (row_i, src_i) in enumerate(all_candidates):
        if used[i]: continue
        cluster_rows  = [row_i]
        cluster_srcs  = [src_i]
        used[i] = True

        for j in range(i+1, len(all_candidates)):
            if used[j]: continue
            row_j, src_j = all_candidates[j]
            if abs(row_j - row_i) <= merge_radius:
                cluster_rows.append(row_j)
                cluster_srcs.append(src_j)
                used[j] = True
            elif row_j - row_i > merge_radius:
                break   # sorted, no need to look further

        # representative row = median of cluster
        rep_row = int(np.median(cluster_rows))
        # unique sources
        unique_srcs = list(dict.fromkeys(cluster_srcs))  # preserve order

        votes.append(ValleyVote(
            row=rep_row,
            confidence=len(unique_srcs),
            sources=unique_srcs,
        ))

    # ── Step 1: keep all conf >= 2 cuts ─────────────────────
    high_conf = [v for v in votes if v.confidence >= 2]
    low_conf  = [v for v in votes if v.confidence == 1]
    high_conf.sort(key=lambda v: v.row)

    # ── Step 2: estimate expected line height from high-conf cuts ──
    # use median spacing between consecutive high-conf cuts
    if len(high_conf) >= 2:
        spacings = [high_conf[i+1].row - high_conf[i].row
                    for i in range(len(high_conf)-1)]
        median_spacing = float(np.median(spacings))
    else:
        median_spacing = MIN_LINE_HEIGHT * 1.5   # fallback

    # ── Step 3: promote conf=1 cuts that fill large gaps ──────────
    # a conf=1 cut is kept if:
    #   a) it sits in a gap larger than 1.4x median spacing
    #      (meaning a line is probably missing there)
    #   AND
    #   b) it is at least MIN_LINE_HEIGHT rows from adjacent cuts
    #      (not a false modifier-gap cut)
    promoted = []
    for lv in low_conf:
        # find the gap this cut sits in among high_conf cuts
        prev_row = 0
        next_row = 99999
        for hv in high_conf:
            if hv.row < lv.row:
                prev_row = hv.row
            elif hv.row > lv.row:
                next_row = hv.row
                break
        gap_size = next_row - prev_row

        # also check distance from already-promoted cuts
        min_dist_to_any = min(
            [abs(lv.row - hv.row) for hv in high_conf + promoted]
            or [MIN_LINE_HEIGHT + 1]
        )

        if (gap_size > median_spacing * 1.4 and
                min_dist_to_any >= MIN_LINE_HEIGHT):
            promoted.append(lv)

    votes = sorted(high_conf + promoted, key=lambda v: v.row)

    # ── Step 4: remove cuts too close to each other ───────────────
    # after promotion, still remove any cuts < MIN_LINE_HEIGHT apart
    if len(votes) > 1:
        filtered = [votes[0]]
        for v in votes[1:]:
            gap = v.row - filtered[-1].row
            if gap >= MIN_LINE_HEIGHT:
                filtered.append(v)
            else:
                if v.confidence > filtered[-1].confidence:
                    filtered[-1] = v
        votes = filtered

    return votes, proj_valleys



#  V3 DP SEAM  (unchanged)

# ── Cost weights (v8) ────────────────────────────────────────
# Increased INK_WEIGHT makes crossing a character very expensive
# Quadratic valley penalty discourages large deviations from seed
# Corridor width penalises narrow vertical channels inside glyphs
INK_WEIGHT_V8       = 100.0  # was 40 — crossing ink now very costly
DIST_WEIGHT_V8      =   4.0  # gap-centre reward (unchanged)
VALLEY_WEIGHT_MIN   =   0.5  # per-column low weight when seed is ink
VALLEY_WEIGHT_MAX   =   3.0  # per-column high weight when seed is clear
VALLEY_QUAD_SCALE   =  10.0  # quadratic valley penalty scale factor
CORRIDOR_WEIGHT     =   5.0  # penalty for narrow horizontal clearance


def build_cost_map(binary: np.ndarray,
                   seed_row: int,
                   search_radius: int = SEARCH_RADIUS,
                   confidence: int = 3
                   ) -> Tuple[np.ndarray, int, int]:
    """
    Improved cost map (v8) — four cost terms:

    1. INK COST (100x) — crossing a character pixel is very expensive.
       Higher than v7 to prevent seam diving into character trunks.

    2. DISTANCE COST — reward for being in the centre of a gap
       (radial distance transform, same as before).

    3. QUADRATIC VALLEY PENALTY — discourages large deviations from seed.
       cost = scale * (|y-seed| / radius)^2
       Small adjustments cheap, large dives very expensive.
       Per-column: where seed row has ink, penalty is reduced so seam
       can still find the real gap below a modifier trunk.

    4. CORRIDOR WIDTH COST — penalises narrow horizontal channels.
       For each background pixel, measures horizontal clearance:
       (distance to nearest left ink) + (distance to nearest right ink)
       A thin vertical trunk has small clearance → high cost.
       A wide inter-line gap has large clearance → low cost.
       This is the key fix for the Sinhala modifier trunk problem.
    """
    H, W = binary.shape

    radius_scale  = {3: 1.0, 2: 1.3, 1: 1.6}.get(confidence, 1.0)
    actual_radius = min(H//3, int(search_radius * radius_scale))

    y_min = max(0,   seed_row - actual_radius)
    y_max = min(H-1, seed_row + actual_radius)
    band  = binary[y_min:y_max+1, :].copy()
    band_H= band.shape[0]

    ink_mask = (band > 0).astype(np.float32)   # 1=ink, 0=bg
    bg_mask  = (band == 0).astype(np.uint8)    # 1=bg,  0=ink

    # ── Term 1: ink cost (high penalty) ──────────────────────
    ink_cost = ink_mask * INK_WEIGHT_V8

    # ── Term 2: radial distance cost ─────────────────────────
    if np.sum(bg_mask) == 0:
        dist_cost = np.ones((band_H, W), dtype=np.float32)
    else:
        dist_from_ink = cv2.distanceTransform(bg_mask, cv2.DIST_L2, 5)
        max_d = dist_from_ink.max()
        gap_reward = dist_from_ink/max_d if max_d > 0                      else np.zeros_like(dist_from_ink)
        dist_cost = 1.0 - gap_reward
        dist_cost[ink_mask > 0] = 1.0

    # ── Term 3: quadratic valley penalty (per-column) ────────
    seed_local = max(0, min(band_H-1, seed_row - y_min))
    seed_ink   = ink_mask[seed_local, :]   # (W,) — 1 where seed row is ink

    # adaptive weight per column
    valley_weight_col = (VALLEY_WEIGHT_MAX * (1.0 - seed_ink) +
                         VALLEY_WEIGHT_MIN * seed_ink)

    rows = np.arange(y_min, y_max+1, dtype=np.float32)
    # normalised distance from seed: 0 at seed, 1 at band edge
    norm_dist = np.abs(rows - seed_row) / max(1.0, float(actual_radius))
    # QUADRATIC — small moves cheap, large dives expensive
    norm_dist_sq = norm_dist ** 2
    valley_cost = (VALLEY_QUAD_SCALE *
                   norm_dist_sq[:, np.newaxis] *
                   valley_weight_col[np.newaxis, :])

    # ── Term 4: corridor width cost ──────────────────────────
    # For each background pixel at (y, x), measure:
    #   left_dist  = pixels to the left until we hit ink (or band edge)
    #   right_dist = pixels to the right until we hit ink (or band edge)
    #   clearance  = left_dist + right_dist
    # Thin vertical trunk: clearance ~ 1-3px → high penalty
    # Wide inter-line gap: clearance ~ 10-30px → low penalty
    #
    # Computed efficiently using cumulative sums from each side.

    # left clearance: for each (y,x), how many consecutive bg pixels
    # are there to the left (including self)?
    left_clear  = np.zeros((band_H, W), dtype=np.float32)
    right_clear = np.zeros((band_H, W), dtype=np.float32)

    for y in range(band_H):
        row = bg_mask[y, :]   # 1=bg, 0=ink
        # left clearance: reset to 0 at ink pixels
        lc = 0
        for x in range(W):
            if row[x] == 1:
                lc += 1
            else:
                lc = 0
            left_clear[y, x] = lc
        # right clearance: same from right
        rc = 0
        for x in range(W-1, -1, -1):
            if row[x] == 1:
                rc += 1
            else:
                rc = 0
            right_clear[y, x] = rc

    clearance = left_clear + right_clear   # total horizontal clearance

    # normalise clearance to [0, 1] where 1 = widest gap seen
    max_clear = clearance.max()
    if max_clear > 0:
        norm_clear = clearance / max_clear
    else:
        norm_clear = np.zeros_like(clearance)

    # corridor cost: narrow clearance → high cost
    # ink pixels get maximum corridor cost
    corridor_cost = 1.0 - norm_clear
    corridor_cost[ink_mask > 0] = 1.0

    # ── Combined cost ─────────────────────────────────────────
    cost_map = (ink_cost                          +
                DIST_WEIGHT_V8 * dist_cost        +
                valley_cost                       +
                CORRIDOR_WEIGHT * corridor_cost)

    return cost_map.astype(np.float64), y_min, y_max

def find_seam_dp(cost_map: np.ndarray, seed_row: int,
                  y_min: int, max_step: int = MAX_STEP) -> np.ndarray:
    band_H, W  = cost_map.shape
    seed_local = max(0, min(band_H-1, seed_row - y_min))
    INF = float('inf')
    dp   = np.full((band_H, W), INF,  dtype=np.float64)
    prev = np.full((band_H, W), -1,   dtype=np.int32)

    start_pen = np.abs(np.arange(band_H, dtype=np.float64) - seed_local)
    dp[:, 0]  = cost_map[:, 0] + 2.0 * start_pen

    for x in range(1, W):
        for y in range(band_H):
            best_cost = INF; best_prev = -1
            for dy in range(-max_step, max_step+1):
                py = y + dy
                if py<0 or py>=band_H: continue
                if dp[py,x-1] == INF:  continue
                bend = abs(y - py)
                cand = dp[py,x-1] + cost_map[y,x] + BEND_WEIGHT*bend
                if cand < best_cost:
                    best_cost = cand; best_prev = py
            dp[y,x] = best_cost; prev[y,x] = best_prev

    end_y = int(np.argmin(dp[:, W-1]))
    sl = np.zeros(W, dtype=np.int32); sl[W-1] = end_y
    for x in range(W-2, -1, -1):
        p = prev[sl[x+1], x+1]
        sl[x] = p if p >= 0 else sl[x+1]
    return (sl + y_min).astype(np.int32)



#  SEGMENTATION


@dataclass
class SeamSegment:
    index:      int
    image:      np.ndarray
    y_start:    int
    y_end:      int
    confidence: int    # confidence of the cut ABOVE this segment
    sources:    List[str] = field(default_factory=list)


def trim_to_content(crop: np.ndarray,
                    padding: int = 3,
                    noise_ratio: float = 0.05) -> np.ndarray:
    """
    Post-seam content-aware trim.

    After the seam cuts produce a crop, remove empty rows from the
    top and bottom — rows with almost no ink. This replicates v0's
    r0/r1 detection but applied inside each crop rather than globally.

    Works on display-format image (white bg, black ink).

    padding: rows of whitespace to preserve around content (default 3)
    noise_ratio: rows with ink < this fraction of max_row_ink are empty

    Returns trimmed crop. If crop has no ink, returns original unchanged.
    """
    # work on inverted (ink=255)
    inv  = cv2.bitwise_not(crop)
    H, W = inv.shape

    # row-wise ink count
    row_ink = inv.sum(axis=1) / 255.0

    max_ink = row_ink.max()
    if max_ink == 0:
        return crop   # empty crop — return as-is

    noise_thresh = max_ink * noise_ratio
    active = np.where(row_ink > noise_thresh)[0]

    if len(active) == 0:
        return crop

    top    = max(0, int(active[0])  - padding)
    bottom = min(H, int(active[-1]) + padding + 1)

    if bottom <= top:
        return crop

    return crop[top:bottom, :]


def segment_v6(binary: np.ndarray,
               min_confidence: int = MIN_CONFIDENCE,
               search_radius:  int = SEARCH_RADIUS,
               max_step:       int = MAX_STEP,
               padding:        int = PADDING,
               ) -> Tuple[List[SeamSegment],
                          List[ValleyVote],
                          Dict[str, List[int]]]:
    """
    Full v6 pipeline.
    Returns (segments, votes, proj_valleys) for visualisation.
    """
    H, W = binary.shape

    votes, proj_valleys = three_projection_consensus(
        binary, min_confidence=min_confidence)

    if not votes:
        seg = SeamSegment(0, cv2.bitwise_not(binary), 0, H-1,
                          confidence=0, sources=[])
        return [seg], votes, proj_valleys

    seed_rows = [v.row for v in votes]

    # DP seam per seed row
    seams = []
    for v in seed_rows:
        cm, y_min, y_max = build_cost_map(binary, v, search_radius)
        seam = find_seam_dp(cm, v, y_min, max_step)
        seams.append(seam)

    # crop line regions
    segments = []
    n_cuts   = len(seams)
    for idx in range(n_cuts+1):
        top_seam = None if idx==0       else seams[idx-1]
        bot_seam = None if idx==n_cuts  else seams[idx]
        vote_above = votes[idx-1] if idx>0 else votes[0]

        y_top_arr = (np.zeros(W, dtype=np.int32) if top_seam is None
                     else np.maximum(0, top_seam - padding))
        y_bot_arr = (np.full(W, H, dtype=np.int32) if bot_seam is None
                     else np.minimum(H, bot_seam + padding))

        y_global_top = int(y_top_arr.min())
        y_global_bot = int(y_bot_arr.max())

        crop   = binary[y_global_top:y_global_bot, :].copy()
        crop_h = crop.shape[0]
        for x in range(W):
            rt = int(y_top_arr[x]) - y_global_top
            rb = int(y_bot_arr[x]) - y_global_top
            if rt > 0:       crop[:rt, x] = 0
            if rb < crop_h:  crop[rb:,  x] = 0

        # convert to display format then trim empty margins
        display = cv2.bitwise_not(crop)
        display = trim_to_content(display, padding=3)

        segments.append(SeamSegment(
            index=idx,
            image=display,
            y_start=y_global_top,
            y_end=y_global_bot,
            confidence=vote_above.confidence,
            sources=vote_above.sources))

    return segments, votes, proj_valleys


# 
#  VISUALISATION


# confidence colours: 1=yellow, 2=orange, 3=green
CONF_COLORS = {1: (0,200,255), 2: (0,140,255), 3: (0,220,80)}
SEG_COLORS  = [(0,120,255),(0,220,100),(220,0,220),(0,220,220),
               (220,120,0),(180,180,0),(120,0,220),(255,80,80),
               (80,255,80),(255,160,0)]


def test_v6(image_paths, min_confidence=MIN_CONFIDENCE,
            search_radius=SEARCH_RADIUS, max_step=MAX_STEP):
    import matplotlib.pyplot as plt

    for path in image_paths:
        if not os.path.exists(path):
            print(f"SKIP: {path}"); continue

        name   = os.path.basename(path)
        binary = load_and_prepare(path)
        H, W   = binary.shape
        strip_w= adaptive_strip_width(W)

        segments, votes, proj_valleys = segment_v6(
            binary, min_confidence, search_radius, max_step)

        # rebuild seams for drawing
        seams = []
        for v in votes:
            cm, y_min, y_max = build_cost_map(binary, v.row, search_radius, v.confidence)
            seams.append(find_seam_dp(cm, v.row, y_min, max_step))

        n = len(segments)

        print(f"\n{'='*65}")
        print(f"  {name}  (strip_w={strip_w}px)")
        print(f"  {'Source':<10} {'Valleys':>8}")
        for src, rows in proj_valleys.items():
            print(f"    {src:<10} {len(rows):>6}  {rows}")
        print(f"  Consensus cuts: {len(votes)}")
        for v in votes:
            marker = '★' if v.confidence==3 else ('◆' if v.confidence==2 else '·')
            print(f"    {marker} row {v.row:>4}  "
                  f"conf={v.confidence}  "
                  f"src={'+'.join(v.sources)}")
        print(f"  Segments: {n}")

        # ── figure ─────────────────────────────────────────────
        ncols = max(4, n)
        fig, axes = plt.subplots(2, ncols, figsize=(3*ncols, 8))
        fig.suptitle(
            f"{name}  strip_w={strip_w}px  |  "
            f"conf★3=green  conf◆2=orange  conf·1=yellow  |  "
            f"dashed=seed row  thick=DP seam",
            fontsize=8)

        # ── row 0 col 0: overlay ──────────────────────────────
        disp = cv2.cvtColor(cv2.bitwise_not(binary), cv2.COLOR_GRAY2BGR)
        for v, seam in zip(votes, seams):
            col = CONF_COLORS[v.confidence]
            # seed row — thin dashed line
            for x in range(0, W, 8):
                cv2.line(disp, (x, v.row),
                          (min(W-1, x+4), v.row), (100,100,100), 1)
            # DP seam — thick coloured line
            for x in range(W-1):
                y1, y2 = int(seam[x]), int(seam[x+1])
                if 0<=y1<H and 0<=y2<H:
                    cv2.line(disp, (x,y1), (x+1,y2), col, 2)
        axes[0][0].imshow(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
        axes[0][0].set_title(
            "Seam overlay\ngreen=conf3  orange=conf2  yellow=conf1",
            fontsize=7)
        axes[0][0].axis('off')

        # ── row 0 col 1: vote summary bar chart ───────────────
        # show which rows got votes and from how many sources
        vote_array = np.zeros(H)
        for v in votes:
            vote_array[v.row] = v.confidence
        axes[0][1].barh(np.arange(H), vote_array,
                         color=['gold','orange','limegreen'][2],
                         alpha=0.7)
        for v in votes:
            col_name = {3:'limegreen', 2:'orange', 1:'gold'}[v.confidence]
            axes[0][1].barh([v.row], [v.confidence], color=col_name)
            axes[0][1].text(v.confidence+0.05, v.row,
                             f"conf={v.confidence}",
                             va='center', fontsize=6)
        axes[0][1].invert_yaxis()
        axes[0][1].set_title("Confidence per cut\n3=all agree  1=one proj",
                              fontsize=7)
        axes[0][1].set_xlabel("confidence")

        # ── row 0 col 2: three projections overlaid ───────────
        gp = smooth_1d(binary.sum(axis=1)/255.0)
        lp = smooth_1d(binary[:,:strip_w].sum(axis=1)/255.0)
        rp = smooth_1d(binary[:,W-strip_w:].sum(axis=1)/255.0)
        # normalise each to [0,1] for easy comparison
        def norm(x): m=x.max(); return x/m if m>0 else x
        axes[0][2].barh(np.arange(H), norm(gp),
                         color='steelblue', alpha=0.4, label='global')
        axes[0][2].barh(np.arange(H), norm(lp)*0.7,
                         color='green',    alpha=0.5, label='left')
        axes[0][2].barh(np.arange(H), norm(rp)*0.5,
                         color='red',      alpha=0.5, label='right')
        for src, col, rows in [('global','navy',proj_valleys['global']),
                                ('left','darkgreen',proj_valleys['left']),
                                ('right','darkred',proj_valleys['right'])]:
            for r in rows:
                axes[0][2].axhline(r, color=col, lw=1, ls='--', alpha=0.7)
        axes[0][2].invert_yaxis()
        axes[0][2].legend(fontsize=6, loc='lower right')
        axes[0][2].set_title(
            f"3 projections\nglobal={len(proj_valleys['global'])}  "
            f"left={len(proj_valleys['left'])}  "
            f"right={len(proj_valleys['right'])}", fontsize=7)

        for j in range(3, ncols): axes[0][j].axis('off')

        # ── row 1: line crops ─────────────────────────────────
        for seg in segments:
            if seg.index < ncols:
                axes[1][seg.index].imshow(seg.image, cmap='gray')
                conf_str = '★'*seg.confidence
                axes[1][seg.index].set_title(
                    f"L{seg.index}  {conf_str}\n"
                    f"{seg.y_start}–{seg.y_end}",
                    fontsize=7)
                axes[1][seg.index].axis('off')
        for j in range(n, ncols): axes[1][j].axis('off')

        plt.tight_layout()
        plt.show()


def batch_v6(bubble_dir, min_confidence=MIN_CONFIDENCE,
              extensions=('.png','.jpg','.jpeg')):
    """Compare global valley count vs v6 segment count for all bubbles."""
    files = sorted(f for f in Path(bubble_dir).glob("*")
                   if f.suffix.lower() in extensions)
    if not files:
        print(f"No images in {bubble_dir}"); return

    print(f"\n{'Bubble':<30} {'Global':>8} {'Left':>6} "
          f"{'Right':>7} {'v6 segs':>9} {'Diff':>6}")
    print("-"*70)
    tg=tl=tr=tv=0
    for f in files:
        try:
            binary = load_and_prepare(str(f))
            _, votes, pv = segment_v6(binary, min_confidence)
            ng = len(pv['global'])+1
            nl = len(pv['left'])+1
            nr = len(pv['right'])+1
            nv = len(votes)+1
            diff = nv-ng
            mk = " ←" if diff!=0 else ""
            print(f"  {f.name:<28} {ng:>8} {nl:>6} {nr:>7} "
                  f"{nv:>9} {diff:>+6}{mk}")
            tg+=ng; tl+=nl; tr+=nr; tv+=nv
        except Exception as e:
            print(f"  {f.name:<28} ERROR: {e}")
    print("-"*70)
    print(f"  {'TOTAL':<28} {tg:>8} {tl:>6} {tr:>7} {tv:>9} {tv-tg:>+6}")
    print()
    print(f"  Confidence breakdown across all cuts:")
    print(f"  (run test mode on individual bubbles to see per-cut scores)")



#  SAVE TO DISK


import csv


def save_v6(bubble_dir, output_dir, min_confidence=MIN_CONFIDENCE,
            extensions=('.png', '.jpg', '.jpeg'),
            also_flat=False):
    """
    Run segment_v6 on every bubble in bubble_dir and write each line crop
    to disk under output_dir, plus a manifest.csv with per-line metadata
    (source bubble, line index, y-range, confidence, which projections
    agreed) so you can audit or filter lines by confidence afterward.

    Layout (default):
        output_dir/
          NeeloBubble_001/
            line_00_conf3.png
            line_01_conf2.png
            ...
          NeeloBubble_002/
            ...
          manifest.csv

    also_flat=True additionally writes every crop into output_dir/_flat/
    with a globally-unique filename (bubble__lineNN.png) — handy if your
    downstream tooling wants one directory instead of per-bubble folders.
    """
    files = sorted(f for f in Path(bubble_dir).glob("*")
                    if f.suffix.lower() in extensions)
    if not files:
        print(f"No images in {bubble_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    flat_dir = os.path.join(output_dir, "_flat")
    if also_flat:
        os.makedirs(flat_dir, exist_ok=True)

    manifest_path = os.path.join(output_dir, "manifest.csv")
    rows_written = 0
    failed = []

    with open(manifest_path, "w", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf)
        writer.writerow([
            "bubble", "line_index", "filename", "y_start", "y_end",
            "height_px", "confidence", "sources"
        ])

        for f in files:
            stem = f.stem
            try:
                binary = load_and_prepare(str(f))
                segments, votes, proj_valleys = segment_v6(
                    binary, min_confidence=min_confidence)

                bubble_out_dir = os.path.join(output_dir, stem)
                os.makedirs(bubble_out_dir, exist_ok=True)

                for seg in segments:
                    # ── art/noise filter ──────────────────────────────
                    inv2 = cv2.bitwise_not(seg.image)
                    h2, w2 = inv2.shape
                    ink_px = int(np.sum(inv2 > 0))
                    fg = ink_px / (h2 * w2) if h2 * w2 > 0 else 0
                    # adaptive density threshold
                    thresh = 0.06 if h2 < 20 else 0.12
                    # reject ultra-thin horizontal strokes
                    # (h very small, w much larger, very few ink pixels)
                    aspect_ok = h2 >= w2 * 0.12
                    min_ink_ok = ink_px >= 60
                    if fg < thresh or not aspect_ok or not min_ink_ok:
                        print(f"    DISCARD {seg.index} "
                              f"(fg={fg:.3f} h={h2} w={w2} ink={ink_px})")
                        continue
                    # ─────────────────────────────────────────────────
                    fname = f"line_{seg.index:02d}_conf{seg.confidence}.png"
                    out_path = os.path.join(bubble_out_dir, fname)
                    cv2.imwrite(out_path, seg.image)

                    if also_flat:
                        flat_name = f"{stem}__line{seg.index:02d}.png"
                        cv2.imwrite(os.path.join(flat_dir, flat_name), seg.image)

                    writer.writerow([
                        stem, seg.index, fname, seg.y_start, seg.y_end,
                        seg.y_end - seg.y_start, seg.confidence,
                        "+".join(seg.sources)
                    ])
                    rows_written += 1

                print(f"  \u2713 {f.name} \u2192 {len(segments)} line(s)")

            except Exception as e:
                failed.append(f.name)
                print(f"  \u2717 {f.name} \u2014 {e}")

    print(f"\nDone: {rows_written} line crops from {len(files)-len(failed)} "
          f"bubbles \u2192 {output_dir}")
    print(f"Manifest: {manifest_path}")
    if failed:
        print(f"Failed: {failed}")




if __name__ == "__main__":

    BUBBLE_DIR = r"D:\UOM\Semester 8\Research\Research_Pipeline\data\masked_Neelo_cleaned"

    TEST_IMAGES = [
        r"D:\UOM\Semester 8\Research\Research_Pipeline\data\masked_Neelo_cleaned\NeeloBubble_001.png",
       
    ]

    mode = sys.argv[1] if len(sys.argv) > 1 else ''

    if mode == 'test':
        test_v6(TEST_IMAGES)
    elif mode == 'batch':
        batch_v6(BUBBLE_DIR)
    elif mode == 'save':
        # python sinhalaLineSeg_seam_v11.py save <input_dir> <output_dir> [--flat]
        in_dir  = sys.argv[2] if len(sys.argv) > 2 else BUBBLE_DIR
        out_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.join(in_dir, "..", "line_segments")
        also_flat = '--flat' in sys.argv
        save_v6(in_dir, out_dir, also_flat=also_flat)
    else:
        print("Usage:")
        print("  python sinhalaLineSeg_seam_v11.py test   # visual test")
        print("  python sinhalaLineSeg_seam_v11.py batch  # all bubbles, counts only")
        print("  python sinhalaLineSeg_seam_v11.py save <input_dir> <output_dir> [--flat]")
        print("                                            # write line crops + manifest.csv")
        print()
        print(f"Parameters (top of file):")
        print(f"  PROJ_FRAC      = {PROJ_FRAC}   fraction of W for left/right strip")
        print(f"  PROJ_MAX_PX    = {PROJ_MAX_PX}   max strip width in pixels")
        print(f"  MERGE_RADIUS   = {MERGE_RADIUS}     merge valleys within ±N rows")
        print(f"  MIN_CONFIDENCE = {MIN_CONFIDENCE}     min votes to keep valley (1=all)")
        print(f"  INK_WEIGHT     = {INK_WEIGHT}  cost for cutting through ink")
        print(f"  VALLEY_WEIGHT  = {VALLEY_WEIGHT}   attraction to seed row")
        print(f"  BEND_WEIGHT    = {BEND_WEIGHT}   penalty for zig-zagging")