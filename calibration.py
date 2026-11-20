"""
Camera geometric calibration for TreeRing Studio.

Implements the standard flexible camera calibration method of
Zhang, Z. (2000), "A flexible new technique for camera calibration",
IEEE Transactions on Pattern Analysis and Machine Intelligence, 22(11),
1330-1334, as provided by OpenCV (cv2.findChessboardCorners /
cv2.calibrateCamera / cv2.undistort).

Purpose
-------
TreeRing Studio converts pixel distances to millimetres using a single
scale factor derived from a ruler photographed next to the increment
core (see Eq. 1 in the manuscript). That conversion assumes a
distortion-free (pinhole) camera model. In reality every smartphone
rear camera introduces some radial and tangential lens distortion,
which is not corrected for in the current release and is not yet
quantified for the camera used in this study (an iPhone 15).

This module provides the calibration and undistortion machinery
needed to close that gap once checkerboard calibration photographs of
the camera in question are available. It performs no measurements on
its own and ships with no precomputed coefficients: a user who wants a
distortion-corrected pixel-to-millimetre conversion must first
photograph a planar checkerboard target with their own device (several
images, at different positions/orientations, following the standard
checkerboard-calibration protocol) and run calibrate_camera() on them.

Usage
-----
    from calibration import calibrate_camera, undistort_image, undistort_points

    result = calibrate_camera(
        image_paths=["chk_01.jpg", "chk_02.jpg", ...],
        checkerboard_size=(9, 6),   # inner corners, (columns, rows)
        square_size_mm=25.0,
    )
    result.save("my_iphone15_calibration.json")

    undistorted_img = undistort_image(img, result)
    undistorted_pts = undistort_points(points_xy, result)

No calibration has been carried out for this study at the time of
submission; this module is released so that calibration can be
performed and reported in a future revision or follow-up study.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class CalibrationResult:
    """Container for the outputs of a Zhang-method camera calibration."""

    camera_matrix: np.ndarray          # 3x3 intrinsic matrix (fx, fy, cx, cy)
    dist_coeffs: np.ndarray            # (k1, k2, p1, p2, k3) distortion coefficients
    image_size: Tuple[int, int]        # (width, height) in pixels, the calibration images' size
    reprojection_error_px: float       # mean reprojection error, in pixels (lower is better; <0.5 px is typical of a good calibration)
    n_images_used: int
    square_size_mm: float
    checkerboard_size: Tuple[int, int]

    def save(self, path: str) -> None:
        payload = {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.ravel().tolist(),
            "image_size": list(self.image_size),
            "reprojection_error_px": self.reprojection_error_px,
            "n_images_used": self.n_images_used,
            "square_size_mm": self.square_size_mm,
            "checkerboard_size": list(self.checkerboard_size),
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "CalibrationResult":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            camera_matrix=np.array(payload["camera_matrix"], dtype=np.float64),
            dist_coeffs=np.array(payload["dist_coeffs"], dtype=np.float64),
            image_size=tuple(payload["image_size"]),
            reprojection_error_px=payload["reprojection_error_px"],
            n_images_used=payload["n_images_used"],
            square_size_mm=payload["square_size_mm"],
            checkerboard_size=tuple(payload["checkerboard_size"]),
        )


def calibrate_camera(
    image_paths: Sequence[str],
    checkerboard_size: Tuple[int, int] = (9, 6),
    square_size_mm: float = 25.0,
) -> CalibrationResult:
    """
    Estimate the camera matrix and distortion coefficients from a set of
    checkerboard images, following Zhang (2000).

    Parameters
    ----------
    image_paths : list of str
        Paths to images of a planar checkerboard target captured with the
        same camera/lens/resolution combination used for the increment-core
        photographs. At least 10-15 images, covering different positions,
        distances, and tilt angles across the frame, are recommended for a
        stable estimate.
    checkerboard_size : (columns, rows)
        Number of INNER corners of the checkerboard pattern (i.e. one less
        than the number of squares in each direction).
    square_size_mm : float
        Physical side length of one checkerboard square, in millimetres.
        This fixes the real-world scale of the calibration; it does not
        need to match the ruler used for ring measurements.

    Returns
    -------
    CalibrationResult

    Raises
    ------
    ValueError
        If the checkerboard pattern could not be detected in enough images
        to calibrate (at least 3), or if no images were supplied.
    """
    if not image_paths:
        raise ValueError("calibrate_camera() requires at least one image path.")

    cols, rows = checkerboard_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size_mm

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    obj_points: List[np.ndarray] = []
    img_points: List[np.ndarray] = []
    image_size = None

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])  # (width, height)

        found, corners = cv2.findChessboardCorners(
            gray, (cols, rows),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            continue

        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )
        obj_points.append(objp)
        img_points.append(corners_refined)

    if len(obj_points) < 3:
        raise ValueError(
            f"Checkerboard pattern was detected in only {len(obj_points)} of "
            f"{len(image_paths)} image(s); at least 3 (ideally 10-15, covering "
            "different positions and angles) are needed for a usable calibration."
        )

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    total_error = 0.0
    for i in range(len(obj_points)):
        projected, _ = cv2.projectPoints(
            obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        observed = img_points[i].reshape(-1, 2).astype(np.float64)
        projected = projected.reshape(-1, 2).astype(np.float64)
        error = cv2.norm(observed, projected, cv2.NORM_L2) / len(projected)
        total_error += error
    mean_error = total_error / len(obj_points)

    return CalibrationResult(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=image_size,
        reprojection_error_px=float(mean_error),
        n_images_used=len(obj_points),
        square_size_mm=square_size_mm,
        checkerboard_size=checkerboard_size,
    )


def undistort_image(img: np.ndarray, calibration: CalibrationResult) -> np.ndarray:
    """Apply lens-distortion correction to a full image using a prior calibration."""
    h, w = img.shape[:2]
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        calibration.camera_matrix, calibration.dist_coeffs, (w, h), alpha=0
    )
    return cv2.undistort(
        img, calibration.camera_matrix, calibration.dist_coeffs, None, new_camera_matrix
    )


def undistort_points(
    points_xy: Sequence[Tuple[float, float]], calibration: CalibrationResult
) -> np.ndarray:
    """
    Map pixel coordinates (e.g. the ruler endpoints or detected ring
    positions used elsewhere in TreeRing Studio) from the original,
    distorted image into their undistorted positions, so that the
    pixel-to-millimetre scale factor (Eq. 1) can be applied consistently
    across the frame.
    """
    pts = np.array(points_xy, dtype=np.float64).reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(
        pts, calibration.camera_matrix, calibration.dist_coeffs,
        P=calibration.camera_matrix,
    )
    return undistorted.reshape(-1, 2)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python calibration.py <checkerboard_image1.jpg> "
            "[<checkerboard_image2.jpg> ...]\n"
            "No calibration images are bundled with this release; capture "
            "several photographs of a printed checkerboard target with the "
            "device used for core photography before running this script."
        )
        sys.exit(0)

    result = calibrate_camera(sys.argv[1:])
    print(f"Used {result.n_images_used} image(s); "
          f"mean reprojection error = {result.reprojection_error_px:.3f} px")
    print("Camera matrix:\n", result.camera_matrix)
    print("Distortion coefficients:\n", result.dist_coeffs.ravel())
    result.save("calibration_result.json")
    print("Saved calibration_result.json")
