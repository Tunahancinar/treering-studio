from pathlib import Path
import cv2
import numpy as np
import pandas as pd

try:
    from skimage.morphology import skeletonize
except Exception:
    skeletonize = None

def make_odd(n):
    n = int(n)
    if n % 2 == 0:
        n += 1
    return max(n, 3)

def sample_band_coords(img_shape, p1, p2, band_width):
    length = int(np.linalg.norm(p2 - p1))
    length = max(length, 2)
    direction = p2 - p1
    direction = direction / (np.linalg.norm(direction) + 1e-6)
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    half = max(int(band_width // 2), 0)
    xs = np.linspace(p1[0], p2[0], length)
    ys = np.linspace(p1[1], p2[1], length)
    h, w = img_shape[:2]
    coords_main = []
    band_pixels = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        coords_main.append((x, y))
        for o in range(-half, half + 1):
            px = x + normal[0] * o
            py = y + normal[1] * o
            xi = int(round(px))
            yi = int(round(py))
            if 0 <= xi < w and 0 <= yi < h:
                band_pixels.append((xi, yi, i, o))
    return np.array(coords_main, dtype=np.float32), band_pixels

def fallback_skeleton(binary):
    skel = np.zeros(binary.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp_img = binary.copy()
    while True:
        eroded = cv2.erode(temp_img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(temp_img, temp)
        skel = cv2.bitwise_or(skel, temp)
        temp_img = eroded.copy()
        if cv2.countNonZero(temp_img) == 0:
            break
    return skel

def preprocess_ycbcr_skeleton(img, out_dir, adaptive_block=51, adaptive_c=5):
    filtered = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    ycrcb = cv2.cvtColor(filtered, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_eq = clahe.apply(y)
    cr_eq = clahe.apply(cr)
    cb_eq = clahe.apply(cb)
    block = make_odd(adaptive_block)
    thr_dark = cv2.adaptiveThreshold(y_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, adaptive_c)
    thr_cr = cv2.adaptiveThreshold(cr_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, adaptive_c)
    thr_cb = cv2.adaptiveThreshold(cb_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, adaptive_c)
    binary = cv2.bitwise_or(thr_dark, cv2.bitwise_and(thr_cr, thr_cb))
    k1 = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k1, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k1, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for lab in range(1, num_labels):
        area = stats[lab, cv2.CC_STAT_AREA]
        if area >= 12:
            cleaned[labels == lab] = 255
    if skeletonize is not None:
        skel = (skeletonize(cleaned > 0).astype(np.uint8) * 255)
    else:
        skel = fallback_skeleton(cleaned)
    out = Path(out_dir)
    cv2.imwrite(str(out / "02_filtered_bilateral.jpg"), filtered)
    cv2.imwrite(str(out / "03_ycbcr_y_clahe.jpg"), y_eq)
    cv2.imwrite(str(out / "04_adaptive_binary.jpg"), binary)
    cv2.imwrite(str(out / "05_morph_cleaned.jpg"), cleaned)
    cv2.imwrite(str(out / "06_skeleton.jpg"), skel)
    return skel, y_eq

def normalize_signal(values):
    values = np.asarray(values, dtype=np.float32)
    span = float(np.ptp(values))
    if span < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return (values - float(np.min(values))) / span

def detect_ring_points_skeleton(img, p1, p2, pixel_to_mm, out_dir, name, band_width=21, min_ring_mm=0.20, adaptive_block=51, adaptive_c=5):
    skel, y_eq = preprocess_ycbcr_skeleton(img, out_dir, adaptive_block=adaptive_block, adaptive_c=adaptive_c)
    coords_main, band_pixels = sample_band_coords(img.shape, p1, p2, band_width)
    if len(coords_main) < 20:
        raise RuntimeError("Olcum cizgisi cok kisa.")
    profile_len = len(coords_main)
    hit_score = np.zeros(profile_len, dtype=np.float32)
    hit_count = np.zeros(profile_len, dtype=np.float32)
    base_sum = np.zeros(profile_len, dtype=np.float32)
    base_count = np.zeros(profile_len, dtype=np.float32)
    for xi, yi, i, o in band_pixels:
        base_sum[i] += float(y_eq[yi, xi])
        base_count[i] += 1.0
        if skel[yi, xi] > 0:
            weight = 1.0 / (1.0 + abs(o))
            hit_score[i] += weight
            hit_count[i] += 1
    base_profile = np.divide(base_sum, np.maximum(base_count, 1.0))
    dark_profile = 255.0 - base_profile
    dark_smooth = cv2.GaussianBlur(dark_profile.reshape(-1, 1), (0, 0), sigmaX=1.0).ravel()
    score = 0.68 * normalize_signal(hit_score) + 0.32 * normalize_signal(dark_smooth)
    hit_binary = hit_score > 0
    components = []
    start = None
    for i, val in enumerate(hit_binary):
        if val and start is None:
            start = i
        if (not val or i == len(hit_binary) - 1) and start is not None:
            end = i if val and i == len(hit_binary) - 1 else i - 1
            if end >= start:
                components.append((start, end))
            start = None
    min_dist_px = max(int(min_ring_mm / pixel_to_mm), 3)
    raw_peaks = []
    raw_scores = []
    for start, end in components:
        segment = hit_score[start:end+1]
        local = int(np.argmax(segment))
        center_idx = start + local
        raw_peaks.append(center_idx)
        raw_scores.append(float(hit_score[center_idx]))
    merged = []
    merged_scores = []
    for idx, peak_score in zip(raw_peaks, raw_scores):
        if not merged:
            merged.append(idx)
            merged_scores.append(peak_score)
        elif idx - merged[-1] >= min_dist_px:
            merged.append(idx)
            merged_scores.append(peak_score)
        elif peak_score > merged_scores[-1]:
            merged[-1] = idx
            merged_scores[-1] = peak_score
    peaks = np.array(merged, dtype=int)
    peak_sources = {idx: "skeleton" for idx in peaks}
    ring_points = coords_main[peaks] if len(peaks) else np.empty((0, 2), dtype=np.float32)
    debug_df = pd.DataFrame({
        "distance_px": np.arange(profile_len),
        "distance_mm": np.arange(profile_len) * pixel_to_mm,
        "base_profile": base_profile,
        "dark_profile": dark_profile,
        "dark_smooth": dark_smooth,
        "score": score,
        "skeleton_hit_score": hit_score,
        "skeleton_hit_count": hit_count,
        "hit_binary": hit_binary.astype(int),
        "selected_peak": np.isin(np.arange(profile_len), peaks).astype(int),
        "peak_source": [peak_sources.get(i, "") for i in range(profile_len)],
    })
    debug_df.to_csv(Path(out_dir) / (name + "_skeleton_profile_debug.csv"), index=False, encoding="utf-8-sig")
    skel_vis = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
    cv2.line(skel_vis, tuple(p1.astype(int)), tuple(p2.astype(int)), (0, 255, 0), 2)
    for p in ring_points:
        cv2.circle(skel_vis, tuple(p.astype(int)), 4, (0, 0, 255), -1)
    cv2.imwrite(str(Path(out_dir) / "07_skeleton_points.jpg"), skel_vis)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(13, 4))
        plt.plot(debug_df["distance_mm"], normalize_signal(debug_df["skeleton_hit_score"]), label="skeleton_hit_score")
        plt.plot(debug_df["distance_mm"], normalize_signal(debug_df["dark_smooth"]), label="dark profile")
        plt.plot(debug_df["distance_mm"], debug_df["score"], label="combined score")
        if len(peaks):
            plt.scatter(peaks * pixel_to_mm, score[peaks], s=25, label="selected ring boundaries")
        plt.xlabel("Distance mm"); plt.ylabel("Normalized score"); plt.legend(); plt.tight_layout()
        plt.savefig(Path(out_dir) / (name + "_skeleton_profile_plot.png"), dpi=220)
        plt.close()
    except Exception as e:
        print("Skeleton profil grafigi cizilemedi:", e)
    return ring_points, debug_df, peaks
