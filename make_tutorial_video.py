from pathlib import Path
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "tutorial_video"
OUT_DIR.mkdir(exist_ok=True)

VIDEO_PATH = OUT_DIR / "TreeRing_Studio_Tutorial_EN.mp4"
SCRIPT_PATH = OUT_DIR / "TreeRing_Studio_Tutorial_EN_script.txt"
PREVIEW_PATH = OUT_DIR / "TreeRing_Studio_Tutorial_EN_preview.png"

W, H = 1920, 1080
FPS = 30

FONT_REG = "C:/Windows/Fonts/segoeui.ttf"
FONT_BOLD = "C:/Windows/Fonts/segoeuib.ttf"


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size=size)


TITLE = font(72, True)
H1 = font(48, True)
H2 = font(34, True)
BODY = font(29)
BODY_BOLD = font(29, True)
SMALL = font(23)
TINY = font(19)


COLORS = {
    "bg": "#08111f",
    "panel": "#13223a",
    "panel2": "#0f1b2e",
    "cyan": "#2dd4bf",
    "sky": "#7dd3fc",
    "amber": "#fbbf24",
    "red": "#ef4444",
    "green": "#22c55e",
    "ink": "#f8fafc",
    "muted": "#a8b3c7",
    "line": "#2b3f61",
    "white": "#ffffff",
}


ASSETS = {
    "sample": Path("C:/Users/TunahanC/Desktop/12_2-17.jpeg"),
    "ui": Path("C:/Users/TunahanC/Desktop/Ekran Alıntısı.PNG"),
    "workflow": ROOT / "article_figures" / "treeringai_workflow_photo_ui_readable_600dpi_preview.png",
}


script_lines = []


def add_script(title, bullets):
    script_lines.append(title)
    for item in bullets:
        script_lines.append(f"- {item}")
    script_lines.append("")


def new_canvas():
    img = Image.new("RGB", (W, H), COLORS["bg"])
    draw = ImageDraw.Draw(img)
    for y in range(H):
        alpha = y / H
        r = int(8 + 10 * alpha)
        g = int(17 + 12 * alpha)
        b = int(31 + 22 * alpha)
        draw.line([(0, y), (W, y)], fill=(r, g, b))
    return img


def wrap_text(draw, text, fnt, max_width):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        trial = word if not current else current + " " + word
        if draw.textbbox((0, 0), trial, font=fnt)[2] <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_text_block(draw, xy, title, bullets, width, title_color=None):
    x, y = xy
    title_color = title_color or COLORS["sky"]
    draw.text((x, y), title, font=H1, fill=title_color)
    y += 76
    for bullet in bullets:
        lines = wrap_text(draw, bullet, BODY, width - 42)
        draw.rounded_rectangle([x, y + 8, x + 18, y + 26], radius=5, fill=COLORS["cyan"])
        line_y = y
        for line in lines:
            draw.text((x + 38, line_y), line, font=BODY, fill=COLORS["ink"])
            line_y += 41
        y = line_y + 16


def panel(draw, box, fill=None, outline=None, radius=24):
    fill = fill or COLORS["panel"]
    outline = outline or COLORS["line"]
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=2)


def load_image(path):
    if path.exists():
        return Image.open(path).convert("RGB")
    img = Image.new("RGB", (900, 540), COLORS["panel2"])
    d = ImageDraw.Draw(img)
    d.text((40, 240), f"Missing asset:\n{path.name}", font=H2, fill=COLORS["muted"])
    return img


def fit_image(img, box, mode="contain"):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    iw, ih = img.size
    scale = max(bw / iw, bh / ih) if mode == "cover" else min(bw / iw, bh / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    if mode == "cover":
        left = max(0, (nw - bw) // 2)
        top = max(0, (nh - bh) // 2)
        resized = resized.crop((left, top, left + bw, top + bh))
        return resized
    canvas = Image.new("RGB", (bw, bh), COLORS["panel2"])
    canvas.paste(resized, ((bw - nw) // 2, (bh - nh) // 2))
    return canvas


def paste_framed(base, img, box, mode="contain", label=None):
    draw = ImageDraw.Draw(base)
    x0, y0, x1, y1 = box
    panel(draw, box)
    inner = (x0 + 14, y0 + 14, x1 - 14, y1 - 14)
    fitted = fit_image(img, inner, mode=mode)
    base.paste(fitted, (inner[0], inner[1]))
    if label:
        tw = draw.textbbox((0, 0), label, font=SMALL)[2]
        draw.rounded_rectangle([x0 + 24, y1 - 62, x0 + tw + 54, y1 - 22], radius=12, fill="#08111fcc")
        draw.text((x0 + 38, y1 - 55), label, font=SMALL, fill=COLORS["white"])


def draw_header(draw, step, total):
    draw.text((76, 44), "TreeRing Studio", font=H2, fill=COLORS["white"])
    progress_x = 1420
    draw.rounded_rectangle([progress_x, 58, 1818, 84], radius=13, fill="#1f3150")
    fill_w = int(398 * step / total)
    draw.rounded_rectangle([progress_x, 58, progress_x + fill_w, 84], radius=13, fill=COLORS["cyan"])
    draw.text((progress_x, 96), f"Tutorial step {step}/{total}", font=TINY, fill=COLORS["muted"])


def make_title_slide():
    img = new_canvas()
    draw = ImageDraw.Draw(img)
    draw.text((86, 92), "TreeRing Studio", font=TITLE, fill=COLORS["white"])
    draw.text((92, 176), "Tutorial video for tree-ring analysis", font=H1, fill=COLORS["sky"])
    sample = load_image(ASSETS["sample"])
    paste_framed(img, sample, (100, 330, 1160, 872), mode="cover", label="Example increment core image")
    workflow = load_image(ASSETS["workflow"])
    paste_framed(img, workflow, (1210, 330, 1820, 872), mode="contain", label="Analysis workflow")
    draw.text((100, 930), "Image loading, calibration, ring detection, manual correction, statistics and 3D modelling", font=BODY, fill=COLORS["ink"])
    add_script("Opening", ["This tutorial shows the basic workflow of TreeRing Studio."])
    return img


def make_ui_image():
    ui = load_image(ASSETS["ui"])
    ui = ui.resize((1500, 735), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(ui)
    draw.rectangle([0, 0, ui.width, 72], fill=COLORS["bg"])
    draw.text((16, 11), "TreeRing Studio", font=H2, fill=COLORS["white"])
    draw.rounded_rectangle([1300, 11, 1482, 53], radius=8, fill=COLORS["cyan"])
    draw.text((1324, 20), "Run Analysis", font=SMALL, fill="#06121f")
    return ui


def make_step_slide(step, total, title, bullets, image=None, image_label=None, image_mode="contain"):
    add_script(title, bullets)
    img = new_canvas()
    draw = ImageDraw.Draw(img)
    draw_header(draw, step, total)
    draw_text_block(draw, (92, 190), title, bullets, 760)
    if image is not None:
        paste_framed(img, image, (910, 170, 1828, 930), mode=image_mode, label=image_label)
    return img


def make_super_resolution_visual():
    sample = load_image(ASSETS["sample"])
    left = fit_image(sample, (0, 0, 780, 760), mode="cover")
    right = left.filter(ImageFilter.SHARPEN).filter(ImageFilter.DETAIL)
    img = Image.new("RGB", (1200, 760), COLORS["panel2"])
    img.paste(left.crop((0, 0, 560, 760)), (0, 0))
    img.paste(right.crop((0, 0, 560, 760)), (640, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([590, 0, 610, 760], fill=COLORS["cyan"])
    draw.text((32, 24), "Lanczos x2/x3/x4", font=H2, fill=COLORS["white"])
    draw.text((672, 24), "Enhanced SR", font=H2, fill=COLORS["white"])
    draw.text((672, 76), "CLAHE + sharpness", font=SMALL, fill=COLORS["amber"])
    return img


def make_chart_visual():
    img = Image.new("RGB", (1200, 760), "#f8fafc")
    draw = ImageDraw.Draw(img)
    draw.text((410, 34), "Ring Width Profile", font=H2, fill="#111827")
    x0, y0, x1, y1 = 90, 120, 1110, 640
    draw.rectangle([x0, y0, x1, y1], outline="#111827", width=3)
    for i in range(1, 6):
        y = y0 + i * (y1 - y0) / 6
        draw.line([x0, y, x1, y], fill="#d9dee8", width=2)
    values = [1.1, 1.8, 1.4, 2.2, 1.0, 2.8, 2.4, 1.6, 1.9, 3.4, 2.0, 4.6, 2.2, 5.1, 3.1, 4.3]
    pts = []
    for i, value in enumerate(values):
        x = x0 + 45 + i * ((x1 - x0 - 90) / (len(values) - 1))
        y = y1 - (value / 5.5) * (y1 - y0 - 40) - 20
        pts.append((x, y))
    for a, b in zip(pts[:-1], pts[1:]):
        draw.line([a, b], fill="#0f766e", width=7)
    for idx, (x, y) in enumerate(pts, start=1):
        draw.ellipse([x - 11, y - 11, x + 11, y + 11], fill="#0f766e")
        if idx in (1, 4, 8, 12, 16):
            draw.text((x - 10, y - 38), str(idx), font=SMALL, fill="#111827")
    draw.text((462, 670), "Ring no / estimated age", font=BODY, fill="#111827")
    draw.text((24, 348), "Width (mm)", font=BODY, fill="#111827")
    return img


def make_stats_visual():
    img = Image.new("RGB", (1200, 760), "#f8fafc")
    draw = ImageDraw.Draw(img)
    draw.text((300, 38), "Statistical Comparison", font=H2, fill="#111827")
    metrics = [("R²", 0.92, "#14b8a6"), ("MAE", 1.18, "#3b82f6"), ("RMSE", 1.62, "#f59e0b"), ("MAPE", 6.4, "#ef4444")]
    x = 120
    for name, value, color in metrics:
        h = int(450 * min(value / 7, 1))
        draw.rounded_rectangle([x, 610 - h, x + 170, 610], radius=16, fill=color)
        draw.text((x + 42, 630), name, font=H2, fill="#111827")
        label = f"{value:.2f}" if name != "MAPE" else f"{value:.1f}%"
        draw.text((x + 35, 570 - h), label, font=H2, fill="#111827")
        x += 260
    draw.text((108, 690), "Manual measurements are compared with analysis results.", font=BODY, fill="#334155")
    return img


def make_3d_visual():
    img = Image.new("RGB", (1200, 760), "#eef2f8")
    draw = ImageDraw.Draw(img)
    draw.text((312, 36), "Interactive 3D Ring Model", font=H2, fill="#111827")
    ox, oy = 125, 500
    scale = 8
    widths = [42, 64, 38, 72, 55, 88, 46, 76, 58, 92]

    def proj(x, y, z):
        return (ox + x * scale + y * scale * 0.38, oy - z * scale * 2.4 - y * scale * 0.24)

    x = 0
    for i, w in enumerate(widths):
        color = "#c89a55" if i % 2 == 0 else "#80552c"
        p = [proj(x, -20, 4), proj(x + w / 4, -20, 4), proj(x + w / 4, 20, 4), proj(x, 20, 4)]
        draw.polygon(p, fill=color, outline="#3b2a18")
        x += w / 4
    x = 0
    for i, w in enumerate([0] + widths):
        if i > 0:
            x += w / 4
        p = [proj(x - 0.7, -23, 4), proj(x + 0.7, -23, 4), proj(x + 0.7, 23, 14), proj(x - 0.7, 23, 14)]
        draw.polygon(p, fill="#2266e3", outline="#0b2f75")
        if i in (1, 4, 8):
            draw.text((p[0][0] - 5, p[0][1] - 64), str(i), font=SMALL, fill="#111827")
    panel(draw, (770, 172, 1130, 575), fill="#ffffff", outline="#cbd5e1", radius=18)
    draw.text((800, 206), "Selected boundary", font=H2, fill="#111827")
    for idx, line in enumerate(["Boundary no: 8", "Age marker: 7", "Position: 42.315 mm", "Prev. width: 1.820 mm", "Next width: 2.105 mm"]):
        draw.text((805, 278 + idx * 48), line, font=BODY, fill="#334155")
    draw.text((108, 680), "Blue ridges represent detected ring boundaries. Clickable HTML viewer reports mm and age data.", font=BODY, fill="#334155")
    return img


def build_slides():
    total = 9
    ui = make_ui_image()
    workflow = load_image(ASSETS["workflow"])
    sample = load_image(ASSETS["sample"])
    slides = [
        (make_title_slide(), 5.0),
        (make_step_slide(1, total, "1. Load an image", [
            "Use Select Image for a single image or Select Folder for batch image selection.",
            "Use Output Folder to choose where analysis results will be saved.",
            "Use zoom, fit and navigation buttons to inspect the image comfortably.",
        ], ui, "Main interface", "contain"), 7.0),
        (make_step_slide(2, total, "2. Define scale and measurement line", [
            "In Ruler mode, mark a known distance on the ruler in the image.",
            "Enter the real-world distance in millimeters in the Ruler distance field.",
            "In Measure mode, draw the analysis direction across the tree rings.",
        ], sample, "Calibrated sample image", "cover"), 7.0),
        (make_step_slide(3, total, "3. Improve image quality", [
            "The Super Resolution section applies Lanczos-based x2/x3/x4 resampling.",
            "CLAHE improves local contrast, while sharpness enhancement makes ring boundaries clearer.",
            "Export Super Resolution creates an enhanced image that can be reloaded for analysis.",
        ], make_super_resolution_visual(), "Super Resolution comparison", "contain"), 7.0),
        (make_step_slide(4, total, "4. Choose detection settings", [
            "Balanced, Sensitive and Conservative profiles provide starting settings for different ring densities.",
            "Auto parameter mode estimates band width, minimum ring width and adaptive block from the selected measurement line.",
            "After the Measure line is selected, Otsu-based Adaptive C is calculated automatically.",
        ], ui, "Detection settings", "contain"), 8.0),
        (make_step_slide(5, total, "5. Run the analysis", [
            "Run Analysis detects ring boundaries and orders them along the measurement direction.",
            "Points mode shows point-based detections; Boundaries mode shows detected boundary lines.",
            "The lower summary panel reports estimated age, mean ring width and measurement statistics.",
        ], make_chart_visual(), "Ring width profile", "contain"), 7.0),
        (make_step_slide(6, total, "6. Apply manual correction", [
            "Use Edit Detection Points to add missing boundaries or remove extra detections.",
            "Corrected points are aligned and reordered along the same measurement line.",
            "Save Corrected Analysis regenerates plots, tables and comparison outputs.",
        ], ui, "Manual correction workflow", "contain"), 7.0),
        (make_step_slide(7, total, "7. Run statistical comparison", [
            "Load paired data from Excel/CSV or enter two columns manually.",
            "Manual Measurement is treated as the reference and Analysis Result as the program output.",
            "R2, MAE, RMSE and MAPE quantify agreement, error magnitude and prediction performance.",
        ], make_stats_visual(), "Statistical analysis", "contain"), 7.0),
        (make_step_slide(8, total, "8. Review reports and 3D model", [
            "TreeRing Studio exports charts, Excel tables, visual reports and analysis summaries.",
            "Export 3D Model represents ring intervals as segments and ring boundaries as raised ridges.",
            "The interactive HTML viewer displays boundary number, age marker, millimeter position and neighboring ring widths.",
        ], make_3d_visual(), "Interactive 3D model", "contain"), 8.0),
        (make_step_slide(9, total, "9. Recommended workflow", [
            "First load the image and define the scale accurately.",
            "If image quality is weak, generate a Super Resolution output and analyze the enhanced image.",
            "Always visually review automatic detections and apply manual correction when needed.",
            "Finally, report statistics, charts, Excel outputs and the 3D model together.",
        ], workflow, "Complete workflow", "contain"), 9.0),
    ]
    add_script("Closing", ["TreeRing Studio combines image processing, manual validation, statistics and 3D modelling in one workflow."])
    return slides


def add_subtle_motion(frame, t, duration):
    arr = np.array(frame)
    zoom = 1.0 + 0.018 * (t / max(duration, 0.01))
    h, w = arr.shape[:2]
    nw, nh = int(w / zoom), int(h / zoom)
    x0 = (w - nw) // 2
    y0 = (h - nh) // 2
    crop = arr[y0:y0 + nh, x0:x0 + nw]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)


def main():
    slides = build_slides()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(VIDEO_PATH), fourcc, FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError("Could not open MP4 writer.")

    first_frame_saved = False
    for slide, duration in slides:
        frame_count = int(duration * FPS)
        for i in range(frame_count):
            eased = (1 - math.cos(math.pi * i / max(frame_count - 1, 1))) / 2
            frame = add_subtle_motion(slide, eased, 1.0)
            if not first_frame_saved:
                Image.fromarray(frame).save(PREVIEW_PATH)
                first_frame_saved = True
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    SCRIPT_PATH.write_text("\n".join(script_lines), encoding="utf-8-sig")
    print(VIDEO_PATH)
    print(SCRIPT_PATH)
    print(PREVIEW_PATH)


if __name__ == "__main__":
    main()
