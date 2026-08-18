from pathlib import Path
import cv2
import numpy as np
import pandas as pd

def robust_norm(x):
    x = x.astype(np.float32)
    p1 = np.percentile(x, 1)
    p99 = np.percentile(x, 99)
    if p99 - p1 < 1e-6:
        return np.zeros_like(x)
    y = (x - p1) / (p99 - p1)
    return np.clip(y, 0, 1)

def smooth_signal(x, k):
    k = int(k)
    if k % 2 == 0:
        k += 1
    k = max(k, 3)
    return cv2.GaussianBlur(x.reshape(1, -1).astype(np.float32), (k, 1), 0).reshape(-1)

def sample_band(img, p1, p2, band_width):
    length = int(np.linalg.norm(p2 - p1))
    length = max(length, 2)
    direction = p2 - p1
    direction = direction / (np.linalg.norm(direction) + 1e-6)
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)

    half = max(int(band_width // 2), 0)
    xs = np.linspace(p1[0], p2[0], length)
    ys = np.linspace(p1[1], p2[1], length)
    h, w = img.shape[:2]

    coords = []
    median_colors = []
    mean_colors = []
    std_values = []

    for x, y in zip(xs, ys):
        samples = []
        for o in range(-half, half + 1):
            px = x + normal[0] * o
            py = y + normal[1] * o
            xi, yi = int(round(px)), int(round(py))
            if 0 <= xi < w and 0 <= yi < h:
                samples.append(img[yi, xi])
        if samples:
            arr = np.array(samples, dtype=np.float32)
            coords.append((x, y))
            median_colors.append(np.median(arr, axis=0))
            mean_colors.append(np.mean(arr, axis=0))
            std_values.append(np.mean(np.std(arr, axis=0)))

    return np.array(coords, dtype=np.float32), np.array(median_colors, dtype=np.uint8), np.array(mean_colors, dtype=np.uint8), np.array(std_values, dtype=np.float32)

def local_peaks(signal, min_dist, threshold):
    peaks = []
    for i in range(2, len(signal) - 2):
        if signal[i] >= threshold and signal[i] >= signal[i-1] and signal[i] >= signal[i+1]:
            if not peaks:
                peaks.append(i)
            elif i - peaks[-1] >= min_dist:
                peaks.append(i)
            else:
                if signal[i] > signal[peaks[-1]]:
                    peaks[-1] = i
    return np.array(peaks, dtype=int)

def merge_close_peaks(peaks, score, min_dist):
    if len(peaks) == 0:
        return peaks
    peaks = np.array(sorted(set([int(p) for p in peaks])), dtype=int)
    groups = []
    cur = [peaks[0]]
    for p in peaks[1:]:
        if p - cur[-1] <= min_dist:
            cur.append(p)
        else:
            groups.append(cur)
            cur = [p]
    groups.append(cur)

    out = []
    for g in groups:
        best = max(g, key=lambda idx: score[idx] if 0 <= idx < len(score) else 0)
        out.append(best)
    return np.array(out, dtype=int)

def detect_rings_profile(img, p1, p2, pixel_to_mm, out_dir, name,
                         band_width=51, min_ring_mm=0.05, sensitivity="high"):
    """
    Skeleton yok. Ölçüm bandından 1D yoğunluk profili çıkarır.
    Dar halkalar için yerel minimum, maksimum ve gradyan birleşimi kullanır.
    """
    # Ön işlem
    filtered = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    ycrcb_full = cv2.cvtColor(filtered, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb_full)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    y_eq = clahe.apply(y)
    ycrcb_full[:, :, 0] = y_eq
    enhanced = cv2.cvtColor(ycrcb_full, cv2.COLOR_YCrCb2BGR)

    cv2.imwrite(str(Path(out_dir) / "02_bilateral.jpg"), filtered)
    cv2.imwrite(str(Path(out_dir) / "03_y_clahe.jpg"), y_eq)
    cv2.imwrite(str(Path(out_dir) / "04_enhanced_ycbcr.jpg"), enhanced)

    coords, med_bgr, mean_bgr, band_std = sample_band(enhanced, p1, p2, band_width)
    if len(coords) < 20:
        raise RuntimeError("Olcum cizgisi cok kisa.")

    line = med_bgr.reshape(1, -1, 3)
    gray = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY).reshape(-1)
    ycrcb = cv2.cvtColor(line, cv2.COLOR_BGR2YCrCb)
    lab = cv2.cvtColor(line, cv2.COLOR_BGR2Lab)

    Y = ycrcb[:, :, 0].reshape(-1)
    Cr = ycrcb[:, :, 1].reshape(-1)
    Cb = ycrcb[:, :, 2].reshape(-1)
    L = lab[:, :, 0].reshape(-1)
    A = lab[:, :, 1].reshape(-1)
    B = lab[:, :, 2].reshape(-1)

    # Koyu geç odun sınırlarını görünür yapmak için ters sinyal önemli
    p_gray = robust_norm(gray)
    p_y = robust_norm(Y)
    p_l = robust_norm(L)
    p_cr = robust_norm(Cr)
    p_cb = robust_norm(Cb)
    p_a = robust_norm(A)
    p_b = robust_norm(B)
    p_std = robust_norm(band_std)

    base = 0.30*p_y + 0.25*p_gray + 0.20*p_l + 0.10*p_cr + 0.08*p_cb + 0.04*p_b + 0.03*p_std
    dark = 1.0 - base

    if sensitivity == "high":
        smooths = [5, 9, 15, 21]
        grad_percentile = 55
        dark_percentile = 58
        valley_percentile = 57
    elif sensitivity == "low":
        smooths = [15, 23, 31]
        grad_percentile = 75
        dark_percentile = 72
        valley_percentile = 72
    else:
        smooths = [9, 15, 23]
        grad_percentile = 65
        dark_percentile = 66
        valley_percentile = 65

    min_dist_px = max(int(min_ring_mm / pixel_to_mm), 2)

    all_peaks = []
    score = np.zeros(len(base), dtype=np.float32)

    for k in smooths:
        s = smooth_signal(base, k)
        d = smooth_signal(dark, k)
        grad = smooth_signal(np.abs(np.gradient(s)), max(5, k//2))

        # gradyan pikleri
        th_g = np.percentile(grad, grad_percentile)
        pg = local_peaks(grad, min_dist_px, th_g)
        all_peaks.extend(pg.tolist())
        score += robust_norm(grad)

        # koyu geç odun pikleri
        th_d = np.percentile(d, dark_percentile)
        peaks_dark = local_peaks(d, min_dist_px, th_d)
        all_peaks.extend(peaks_dark.tolist())
        score += 0.75 * robust_norm(d)

        # açık/koyu vadiler için base tersinden ek sinyal
        valley = smooth_signal(1.0 - s, k)
        th_v = np.percentile(valley, valley_percentile)
        pv = local_peaks(valley, min_dist_px, th_v)
        all_peaks.extend(pv.tolist())
        score += 0.40 * robust_norm(valley)

    peaks = merge_close_peaks(all_peaks, score, min_dist_px)

    # Kenarları at
    margin = max(2, min_dist_px // 2)
    peaks = peaks[(peaks > margin) & (peaks < len(coords) - margin)]
    ring_points = coords[peaks] if len(peaks) else np.empty((0, 2), dtype=np.float32)

    df = pd.DataFrame({
        "distance_px": np.arange(len(base)),
        "distance_mm": np.arange(len(base)) * pixel_to_mm,
        "gray": gray,
        "YCbCr_Y": Y,
        "YCbCr_Cr": Cr,
        "YCbCr_Cb": Cb,
        "Lab_L": L,
        "base_profile": base,
        "dark_profile": dark,
        "score": score,
        "band_std": band_std
    })
    df.to_csv(Path(out_dir) / (name + "_profile_debug.csv"), index=False, encoding="utf-8-sig")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(14, 4))
        plt.plot(df["distance_mm"], df["base_profile"], label="base profile")
        plt.plot(df["distance_mm"], df["dark_profile"], label="dark profile")
        plt.plot(df["distance_mm"], robust_norm(score), label="detection score")
        if len(peaks):
            plt.scatter(peaks * pixel_to_mm, robust_norm(score)[peaks], s=25, label="detected rings")
        plt.xlabel("Mesafe (mm)")
        plt.ylabel("Sinyal")
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(out_dir) / (name + "_profile_plot.png"), dpi=220)
        plt.close()
    except Exception as e:
        print("Profil grafigi cizilemedi:", e)

    return ring_points, df, peaks
