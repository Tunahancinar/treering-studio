from pathlib import Path
import cv2
import numpy as np
from utils import select_two_points

def get_manual_scale(img, out_dir):
    p1, p2 = select_two_points(
        img, "1_Cetvel_Olcegi",
        "Cetvelde bilinen araligin baslangic ve bitisini tikla. Ornek: 0-50 mm."
    )
    real_mm = float(input("Tikladigin cetvel araligi kac mm? Ornek 50: ").replace(",", "."))
    dist_px = float(np.linalg.norm(p2 - p1))
    pixel_to_mm = real_mm / dist_px

    vis = img.copy()
    cv2.line(vis, tuple(p1.astype(int)), tuple(p2.astype(int)), (0,255,0), 3)
    cv2.putText(vis, str(real_mm) + " mm", tuple(p1.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    cv2.imwrite(str(Path(out_dir) / "01_selected_ruler_scale.jpg"), vis)

    return pixel_to_mm, dist_px, real_mm
