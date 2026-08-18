import argparse
from pathlib import Path
import pandas as pd
import cv2
from utils import ensure_dir, find_images, read_image, select_two_points
from ruler import get_manual_scale
from profile_detection import detect_rings_profile
from exporter import measure, draw_output, save_all

def process_image(path, output_root, min_ring_mm, band_width, sensitivity):
    img = read_image(path)
    stem = Path(path).stem
    out_dir = Path(output_root) / stem
    ensure_dir(out_dir)

    cv2.imwrite(str(out_dir / "00_original.jpg"), img)

    print("\n" + "="*70)
    print("Islenen goruntu:", Path(path).name)

    pixel_to_mm, ruler_px, ruler_mm = get_manual_scale(img, out_dir)

    p1, p2 = select_two_points(
        img, "2_Yas_Halkasi_Cizgisi",
        "Ozden kabuga dogru yas halkalarini kesecek olcum cizgisini ciz."
    )

    ring_points, debug_df, peaks = detect_rings_profile(
        img, p1, p2, pixel_to_mm, out_dir, stem,
        band_width=band_width,
        min_ring_mm=min_ring_mm,
        sensitivity=sensitivity
    )

    df = measure(ring_points, pixel_to_mm)

    summary = {
        "image": Path(path).name,
        "method": "YCbCr_CLAHE_Bilateral_Band_Profile_Multiscale_Peak_Detection",
        "pixel_to_mm": pixel_to_mm,
        "selected_ruler_px": ruler_px,
        "selected_ruler_mm": ruler_mm,
        "min_ring_mm_parameter": min_ring_mm,
        "band_width_px": band_width,
        "sensitivity": sensitivity,
        "detected_boundaries": int(len(ring_points)),
        "measured_ring_width_count": int(len(df)),
        "mean_width_mm": float(df["width_mm"].mean()) if len(df) else None,
        "min_width_mm": float(df["width_mm"].min()) if len(df) else None,
        "max_width_mm": float(df["width_mm"].max()) if len(df) else None
    }

    save_all(df, summary, debug_df, out_dir)
    draw_output(img, p1, p2, ring_points, out_dir / "05_measurement_visual.jpg")

    print("Tespit edilen nokta:", len(ring_points))
    print("Olculen halka genisligi:", len(df))
    print("Cikti klasoru:", out_dir)

    return summary

def main():
    parser = argparse.ArgumentParser(description="TreeRingAI v6 Profile")
    parser.add_argument("--input", required=True, help="Goruntu veya goruntu klasoru")
    parser.add_argument("--output", required=True, help="Cikti klasoru")
    parser.add_argument("--min-ring-mm", type=float, default=0.05)
    parser.add_argument("--band-width", type=int, default=51)
    parser.add_argument("--sensitivity", choices=["low","normal","high"], default="high")
    args = parser.parse_args()

    ensure_dir(args.output)
    paths = find_images(args.input)
    if not paths:
        raise FileNotFoundError("Goruntu bulunamadi.")

    all_summary = []
    for p in paths:
        try:
            all_summary.append(process_image(p, args.output, args.min_ring_mm, args.band_width, args.sensitivity))
        except Exception as e:
            print("HATA:", p, e)

    if all_summary:
        pd.DataFrame(all_summary).to_csv(Path(args.output) / "ALL_summary_report.csv", index=False, encoding="utf-8-sig")

    print("\nTum islem bitti.")

if __name__ == "__main__":
    main()
