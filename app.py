from pathlib import Path
from datetime import datetime
import json
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
import pandas as pd

from exporter import draw_boundary_output, draw_detection_mode_panel, draw_output, measure, save_all
from skeleton_detection import detect_ring_points_skeleton, sample_band_coords
from utils import ensure_dir, find_images, read_image


APP_TITLE = "TreeRing Studio"
SUPPORTED_INPUT = "Image files (*.jpg *.jpeg *.png *.tif *.tiff *.bmp)"


class ImageCanvas(tk.Canvas):
    def __init__(self, master, on_points_changed, **kwargs):
        super().__init__(master, highlightthickness=0, **kwargs)
        self.on_points_changed = on_points_changed
        self.image = None
        self.tk_image = None
        self.scale = 1.0
        self.fit_scale = 1.0
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.pan_x = 0
        self.pan_y = 0
        self.drag_start = None
        self.points = []
        self.mode = "ruler"
        self.bind("<Button-1>", self._click)
        self.bind("<ButtonPress-2>", self._start_pan)
        self.bind("<B2-Motion>", self._pan)
        self.bind("<ButtonPress-3>", self._start_pan)
        self.bind("<B3-Motion>", self._pan)
        self.bind("<MouseWheel>", self._wheel_zoom)
        self.bind("<Configure>", lambda _event: self.render())

    def set_mode(self, mode):
        self.mode = mode
        self.points = []
        self.render()
        self.on_points_changed()

    def set_image(self, image):
        self.image = image
        self.points = []
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.render()
        self.on_points_changed()

    def zoom_in(self):
        self.set_zoom(self.zoom * 1.25)

    def zoom_out(self):
        self.set_zoom(self.zoom / 1.25)

    def fit_to_view(self):
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.render()

    def pan_by(self, dx, dy):
        if self.image is None:
            return
        self.pan_x += dx
        self.pan_y += dy
        self.render()

    def set_zoom(self, value, center_x=None, center_y=None):
        if self.image is None:
            return
        old_scale = self.scale
        h, w = self.image.shape[:2]
        max_zoom_by_size = max(1.0, min(8.0, 8000 / max(w * self.fit_scale, h * self.fit_scale, 1)))
        value = max(0.2, min(float(value), max_zoom_by_size))
        if center_x is None:
            center_x = self.winfo_width() / 2
        if center_y is None:
            center_y = self.winfo_height() / 2
        img_x = (center_x - self.offset_x) / old_scale if old_scale else 0
        img_y = (center_y - self.offset_y) / old_scale if old_scale else 0
        self.zoom = value
        self.scale = self.fit_scale * self.zoom
        self.pan_x = center_x - self.winfo_width() / 2 - (img_x - w / 2) * self.scale
        self.pan_y = center_y - self.winfo_height() / 2 - (img_y - h / 2) * self.scale
        self.render()

    def reset_points(self):
        self.points = []
        self.render()
        self.on_points_changed()

    def get_points(self):
        return [np.array(p, dtype=np.float32) for p in self.points]

    def _click(self, event):
        if self.image is None or len(self.points) >= 2:
            return
        x = (event.x - self.offset_x) / self.scale
        y = (event.y - self.offset_y) / self.scale
        h, w = self.image.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            self.points.append((x, y))
            self.render()
            self.on_points_changed()

    def _start_pan(self, event):
        self.drag_start = (event.x, event.y, self.pan_x, self.pan_y)

    def _pan(self, event):
        if self.drag_start is None:
            return
        x, y, start_pan_x, start_pan_y = self.drag_start
        self.pan_x = start_pan_x + event.x - x
        self.pan_y = start_pan_y + event.y - y
        self.render()

    def _wheel_zoom(self, event):
        factor = 1.15 if event.delta > 0 else 1 / 1.15
        self.set_zoom(self.zoom * factor, event.x, event.y)

    def render(self):
        self.delete("all")
        if self.image is None:
            self._empty_state()
            return

        width = max(self.winfo_width(), 320)
        height = max(self.winfo_height(), 240)
        h, w = self.image.shape[:2]
        self.fit_scale = min(width / w, height / h, 1.0)
        self.scale = self.fit_scale * self.zoom
        display_w = max(int(w * self.scale), 1)
        display_h = max(int(h * self.scale), 1)
        self.offset_x = int((width - display_w) / 2 + self.pan_x)
        self.offset_y = int((height - display_h) / 2 + self.pan_y)

        resized = cv2.resize(self.image, (display_w, display_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        ppm = b"P6 %d %d 255 " % (display_w, display_h) + rgb.tobytes()
        self.tk_image = tk.PhotoImage(data=ppm, format="PPM")
        self.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.tk_image)

        color = "#50e3c2" if self.mode == "ruler" else "#ffcf5a"
        label = "Ruler scale" if self.mode == "ruler" else "Measurement line"
        for index, point in enumerate(self.points, start=1):
            px = self.offset_x + point[0] * self.scale
            py = self.offset_y + point[1] * self.scale
            self.create_oval(px - 6, py - 6, px + 6, py + 6, fill=color, outline="#10151f", width=2)
            self.create_text(px + 14, py - 12, text=str(index), fill="#ffffff", font=("Segoe UI", 11, "bold"))
        if len(self.points) == 2:
            a, b = self.points
            ax = self.offset_x + a[0] * self.scale
            ay = self.offset_y + a[1] * self.scale
            bx = self.offset_x + b[0] * self.scale
            by = self.offset_y + b[1] * self.scale
            self.create_line(ax, ay, bx, by, fill=color, width=3)

        self.create_rectangle(16, 16, 330, 52, fill="#10151f", outline="")
        self.create_text(28, 34, anchor="w", text=f"{label}: select 2 points | Zoom {int(self.zoom * 100)}%", fill="#f8fafc", font=("Segoe UI", 10, "bold"))

    def _empty_state(self):
        self.create_rectangle(0, 0, self.winfo_width(), self.winfo_height(), fill="#111827", outline="")
        self.create_text(
            self.winfo_width() / 2,
            self.winfo_height() / 2,
            text="Select an image",
            fill="#cbd5e1",
            font=("Segoe UI", 18, "bold"),
        )


class DetectionEditCanvas(tk.Canvas):
    def __init__(self, master, image, points, measure_points, mode_var, on_change, **kwargs):
        super().__init__(master, highlightthickness=0, **kwargs)
        self.image = image
        self.measure_points = [np.array(point, dtype=np.float32) for point in measure_points]
        self.point_records = self._sort_records([
            {"point": np.array(point, dtype=np.float32).copy(), "original_boundary_no": index + 1}
            for index, point in enumerate(points)
        ])
        self.mode_var = mode_var
        self.on_change = on_change
        self.tk_image = None
        self.scale = 1.0
        self.fit_scale = 1.0
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.pan_x = 0
        self.pan_y = 0
        self.drag_start = None
        self.bind("<Button-1>", self._click)
        self.bind("<ButtonPress-2>", self._start_pan)
        self.bind("<B2-Motion>", self._pan)
        self.bind("<ButtonPress-3>", self._start_pan)
        self.bind("<B3-Motion>", self._pan)
        self.bind("<MouseWheel>", self._wheel_zoom)
        self.bind("<Configure>", lambda _event: self.render())

    def zoom_in(self):
        self.set_zoom(self.zoom * 1.25)

    def zoom_out(self):
        self.set_zoom(self.zoom / 1.25)

    def fit_to_view(self):
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.render()

    def pan_by(self, dx, dy):
        self.pan_x += dx
        self.pan_y += dy
        self.render()

    def get_sorted_points(self):
        return [record["point"].copy() for record in self.get_sorted_point_records()]

    def get_sorted_point_records(self):
        return [
            {"point": record["point"].copy(), "original_boundary_no": record.get("original_boundary_no")}
            for record in self._sort_records(self.point_records)
        ]

    def set_zoom(self, value, center_x=None, center_y=None):
        old_scale = self.scale
        h, w = self.image.shape[:2]
        max_zoom_by_size = max(1.0, min(10.0, 9000 / max(w * self.fit_scale, h * self.fit_scale, 1)))
        value = max(0.2, min(float(value), max_zoom_by_size))
        if center_x is None:
            center_x = self.winfo_width() / 2
        if center_y is None:
            center_y = self.winfo_height() / 2
        img_x = (center_x - self.offset_x) / old_scale if old_scale else 0
        img_y = (center_y - self.offset_y) / old_scale if old_scale else 0
        self.zoom = value
        self.scale = self.fit_scale * self.zoom
        self.pan_x = center_x - self.winfo_width() / 2 - (img_x - w / 2) * self.scale
        self.pan_y = center_y - self.winfo_height() / 2 - (img_y - h / 2) * self.scale
        self.render()

    def _click(self, event):
        x = (event.x - self.offset_x) / self.scale
        y = (event.y - self.offset_y) / self.scale
        h, w = self.image.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return
        if self.mode_var.get() == "delete":
            index = self._nearest_point_index(event.x, event.y)
            if index is not None:
                self.point_records.pop(index)
        else:
            self.point_records.append({
                "point": self._project_to_measure_line(np.array([x, y], dtype=np.float32)),
                "original_boundary_no": None,
            })
            self.point_records = self._sort_records(self.point_records)
        self.render()
        self.on_change(len(self.point_records))

    def _nearest_point_index(self, screen_x, screen_y):
        if not self.point_records:
            return None
        distances = []
        for record in self.point_records:
            point = record["point"]
            px = self.offset_x + point[0] * self.scale
            py = self.offset_y + point[1] * self.scale
            distances.append(float(np.hypot(px - screen_x, py - screen_y)))
        index = int(np.argmin(distances))
        return index if distances[index] <= 12 else None

    def _project_to_measure_line(self, point):
        point = np.array(point, dtype=np.float32)
        if len(self.measure_points) != 2:
            return point
        start, end = self.measure_points
        direction = end - start
        length_sq = float(np.dot(direction, direction))
        if length_sq < 1e-6:
            return point
        t = float(np.dot(point - start, direction) / length_sq)
        t = min(max(t, 0.0), 1.0)
        return (start + t * direction).astype(np.float32)

    def _sort_points(self, points):
        if len(self.measure_points) != 2:
            return [np.array(point, dtype=np.float32) for point in points]
        start, end = self.measure_points
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length < 1e-6:
            return [np.array(point, dtype=np.float32) for point in points]
        direction = direction / length
        return sorted([np.array(point, dtype=np.float32) for point in points], key=lambda point: float(np.dot(point - start, direction)))

    def _sort_records(self, records):
        if len(self.measure_points) != 2:
            return list(records)
        start, end = self.measure_points
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length < 1e-6:
            return list(records)
        direction = direction / length
        return sorted(records, key=lambda record: float(np.dot(record["point"] - start, direction)))

    def _start_pan(self, event):
        self.drag_start = (event.x, event.y, self.pan_x, self.pan_y)

    def _pan(self, event):
        if self.drag_start is None:
            return
        x, y, start_pan_x, start_pan_y = self.drag_start
        self.pan_x = start_pan_x + event.x - x
        self.pan_y = start_pan_y + event.y - y
        self.render()

    def _wheel_zoom(self, event):
        factor = 1.15 if event.delta > 0 else 1 / 1.15
        self.set_zoom(self.zoom * factor, event.x, event.y)

    def render(self):
        self.delete("all")
        width = max(self.winfo_width(), 320)
        height = max(self.winfo_height(), 240)
        h, w = self.image.shape[:2]
        self.fit_scale = min(width / w, height / h, 1.0)
        self.scale = self.fit_scale * self.zoom
        display_w = max(int(w * self.scale), 1)
        display_h = max(int(h * self.scale), 1)
        self.offset_x = int((width - display_w) / 2 + self.pan_x)
        self.offset_y = int((height - display_h) / 2 + self.pan_y)

        resized = cv2.resize(self.image, (display_w, display_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        ppm = b"P6 %d %d 255 " % (display_w, display_h) + rgb.tobytes()
        self.tk_image = tk.PhotoImage(data=ppm, format="PPM")
        self.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.tk_image)

        if len(self.measure_points) == 2:
            a, b = self.measure_points
            ax = self.offset_x + a[0] * self.scale
            ay = self.offset_y + a[1] * self.scale
            bx = self.offset_x + b[0] * self.scale
            by = self.offset_y + b[1] * self.scale
            self.create_line(ax, ay, bx, by, fill="#22c55e", width=3)

        sorted_records = self.get_sorted_point_records()
        for index, record in enumerate(sorted_records, start=1):
            point = record["point"]
            px = self.offset_x + point[0] * self.scale
            py = self.offset_y + point[1] * self.scale
            radius = 4
            self.create_oval(px - radius, py - radius, px + radius, py + radius, fill="#ef4444", outline="#ffffff", width=1)
            if index == 1 or index == len(sorted_records) or index % 10 == 0:
                original_no = record.get("original_boundary_no")
                label = str(index) if original_no is None else f"{original_no}->{index}"
                self.create_text(px + 10, py - 9, text=label, fill="#ffffff", font=("Segoe UI", 8, "bold"))

        self.create_rectangle(16, 16, 380, 52, fill="#10151f", outline="")
        self.create_text(
            28,
            34,
            anchor="w",
            text=f"{self.mode_var.get().title()} point | {len(self.point_records)} boundaries | Zoom {int(self.zoom * 100)}%",
            fill="#f8fafc",
            font=("Segoe UI", 10, "bold"),
        )


class TreeRingStudio(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1280x780")
        self.minsize(1080, 680)

        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar(value=str(Path.cwd() / "TreeRing_Results"))
        self.real_mm = tk.StringVar(value="50")
        self.analysis_profile = tk.StringVar(value="Balanced")
        self.detection_display_mode = tk.StringVar(value="Points")
        self.min_ring_mm = tk.DoubleVar(value=0.20)
        self.band_width = tk.IntVar(value=21)
        self.parameter_mode = tk.StringVar(value="Auto")
        self.parameter_status = tk.StringVar(value="Auto: select Ruler and Measure lines")
        self.sensitivity = tk.StringVar(value="low")
        self.adaptive_block = tk.IntVar(value=51)
        self.adaptive_c = tk.IntVar(value=5)
        self.adaptive_c_mode = tk.StringVar(value="Auto (Otsu)")
        self.adaptive_c_status = tk.StringVar(value="Auto: select a Measure line")
        self.sr_scale = tk.IntVar(value=2)
        self.sr_clahe = tk.BooleanVar(value=True)
        self.sr_sharpness = tk.DoubleVar(value=0.65)
        self.status = tk.StringVar(value="Ready")
        self.mode = tk.StringVar(value="ruler")
        self.images = []
        self.current_index = 0
        self.current_image = None
        self.ruler_points = []
        self.measure_points = []
        self.stats_manual_values = []
        self.stats_analysis_values = []
        self.stats_source = ""
        self.stats_auto_remove_outliers = tk.BooleanVar(value=True)
        self.last_analysis_context = None
        self.auto_parameter_info = {}
        self.otsu_info = {}
        self.log_queue = queue.Queue()

        self._style()
        self._layout()
        self.apply_analysis_profile()
        self.after(120, self._drain_logs)

    def _style(self):
        self.configure(bg="#08111f")
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#08111f")
        style.configure("Header.TFrame", background="#08111f")
        style.configure("Panel.TFrame", background="#13223a")
        style.configure("TLabel", background="#08111f", foreground="#e5eefb", font=("Segoe UI", 10))
        style.configure("Panel.TLabel", background="#13223a", foreground="#e5eefb", font=("Segoe UI", 10))
        style.configure("Section.TLabel", background="#13223a", foreground="#7dd3fc", font=("Segoe UI", 12, "bold"))
        style.configure("Title.TLabel", background="#08111f", foreground="#ffffff", font=("Segoe UI", 24, "bold"))
        style.configure("Credit.TLabel", background="#08111f", foreground="#fbbf24", font=("Segoe UI", 10, "bold"))
        style.configure("Muted.TLabel", background="#08111f", foreground="#a8b3c7", font=("Segoe UI", 10))
        style.configure("TButton", background="#243654", foreground="#f8fafc", font=("Segoe UI", 10, "bold"), padding=(12, 8), borderwidth=0)
        style.map("TButton", background=[("active", "#334b73"), ("pressed", "#1d2d49")], foreground=[("disabled", "#94a3b8")])
        style.configure("Accent.TButton", background="#2dd4bf", foreground="#06121f")
        style.map("Accent.TButton", background=[("active", "#5eead4"), ("pressed", "#14b8a6")])
        style.configure("TEntry", fieldbackground="#0b1220", foreground="#f8fafc", insertcolor="#f8fafc")
        style.configure("TCombobox", fieldbackground="#0b1220", foreground="#f8fafc")
        style.configure("TCheckbutton", background="#13223a", foreground="#e5eefb", font=("Segoe UI", 10))
        style.map("TCheckbutton", background=[("active", "#13223a")], foreground=[("active", "#ffffff")])
        style.configure("TRadiobutton", background="#13223a", foreground="#e5eefb", font=("Segoe UI", 10))
        style.map("TRadiobutton", background=[("active", "#13223a")], foreground=[("active", "#ffffff")])
        style.configure("Horizontal.TScale", background="#13223a", troughcolor="#2a3f62")

    def _layout(self):
        root = ttk.Frame(self, padding=18)
        root.pack(fill="both", expand=True)

        header = ttk.Frame(root, style="Header.TFrame")
        header.pack(fill="x", pady=(0, 14))
        brand = ttk.Frame(header, style="Header.TFrame")
        brand.pack(side="left", fill="x", expand=True)
        ttk.Label(brand, text="TreeRing Studio", style="Title.TLabel").pack(anchor="w")
        ttk.Label(brand, text="Designed by Tunahan Çınar | Data provider: Ali Kemal Özbayram", style="Credit.TLabel").pack(anchor="w", pady=(2, 0))
        ttk.Button(header, text="Run Analysis", style="Accent.TButton", command=self.start_analysis).pack(side="right", padx=(12, 0))
        ttk.Label(header, textvariable=self.status, style="Muted.TLabel").pack(side="right", pady=(12, 0))

        body = ttk.Frame(root)
        body.pack(fill="both", expand=True)

        left_shell = ttk.Frame(body, style="Panel.TFrame")
        left_shell.pack(side="left", fill="y", padx=(0, 14))
        left_shell.configure(width=360)
        left_shell.pack_propagate(False)

        left_canvas = tk.Canvas(left_shell, bg="#13223a", highlightthickness=0, width=336)
        left_scroll = ttk.Scrollbar(left_shell, orient="vertical", command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scroll.set)
        left_canvas.pack(side="left", fill="both", expand=True)
        left_scroll.pack(side="right", fill="y")

        left = ttk.Frame(left_canvas, style="Panel.TFrame", padding=14)
        left_window = left_canvas.create_window((0, 0), window=left, anchor="nw")
        left.bind("<Configure>", lambda event: left_canvas.configure(scrollregion=left_canvas.bbox("all")))
        left_canvas.bind("<Configure>", lambda event: left_canvas.itemconfigure(left_window, width=event.width))
        left_canvas.bind("<MouseWheel>", lambda event: left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units"))

        canvas_frame = ttk.Frame(body, style="Panel.TFrame", padding=10)
        canvas_frame.pack(side="left", fill="both", expand=True)

        self._controls(left)
        self._image_toolbar(canvas_frame)
        self.canvas = ImageCanvas(canvas_frame, self._points_changed, bg="#0c1526")
        self.canvas.pack(fill="both", expand=True)

        bottom = ttk.Frame(root, style="Panel.TFrame", padding=10)
        bottom.pack(fill="x", pady=(14, 0))
        bottom_grid = ttk.Frame(bottom, style="Panel.TFrame")
        bottom_grid.pack(fill="x")

        summary_frame = ttk.Frame(bottom_grid, style="Panel.TFrame")
        summary_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        ttk.Label(summary_frame, text="Analysis Summary", style="Panel.TLabel", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 6))
        self.summary_box = tk.Text(
            summary_frame,
            height=7,
            bg="#08111f",
            fg="#e0f2fe",
            insertbackground="#e0f2fe",
            relief="flat",
            font=("Segoe UI", 10),
            wrap="word",
        )
        self.summary_box.pack(fill="both", expand=True)
        self.summary_box.insert("end", "Estimated age, measurement summary, and statistical comparison results will appear here.")
        self.summary_box.configure(state="disabled")

        log_frame = ttk.Frame(bottom_grid, style="Panel.TFrame")
        log_frame.pack(side="left", fill="both", expand=True)
        ttk.Label(log_frame, text="Process Log", style="Panel.TLabel", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 6))
        self.log = tk.Text(
            log_frame,
            height=6,
            bg="#08111f",
            fg="#dbeafe",
            insertbackground="#dbeafe",
            relief="flat",
            font=("Consolas", 10),
        )
        self.log.pack(fill="both", expand=True)

    def _image_toolbar(self, parent):
        toolbar = ttk.Frame(parent, style="Panel.TFrame")
        toolbar.pack(fill="x", pady=(0, 8))
        top = ttk.Frame(toolbar, style="Panel.TFrame")
        top.pack(fill="x")
        bottom = ttk.Frame(toolbar, style="Panel.TFrame")
        bottom.pack(fill="x", pady=(6, 0))

        ttk.Button(top, text="-", command=lambda: self.canvas.zoom_out()).pack(side="left", padx=(0, 6))
        ttk.Button(top, text="+", command=lambda: self.canvas.zoom_in()).pack(side="left", padx=(0, 6))
        ttk.Button(top, text="Fit", command=lambda: self.canvas.fit_to_view()).pack(side="left", padx=(0, 12))
        ttk.Button(top, text="Up", command=lambda: self.canvas.pan_by(0, 80)).pack(side="left", padx=(0, 6))
        ttk.Button(top, text="Down", command=lambda: self.canvas.pan_by(0, -80)).pack(side="left", padx=(0, 6))
        ttk.Button(top, text="Left", command=lambda: self.canvas.pan_by(80, 0)).pack(side="left", padx=(0, 6))
        ttk.Button(top, text="Right", command=lambda: self.canvas.pan_by(-80, 0)).pack(side="left", padx=(0, 12))

        ttk.Button(bottom, text="Previous Image", command=self.previous_image).pack(side="left", padx=(0, 6))
        ttk.Button(bottom, text="Next Image", command=self.next_image).pack(side="left", padx=(0, 12))
        ttk.Label(
            bottom,
            text="Mouse wheel zooms, right/middle drag pans",
            style="Panel.TLabel",
        ).pack(side="left")

    def _controls(self, parent):
        self._section(parent, "File")
        ttk.Button(parent, text="Select Image", command=self.pick_image).pack(fill="x", pady=4)
        ttk.Button(parent, text="Select Folder", command=self.pick_folder).pack(fill="x", pady=4)
        ttk.Entry(parent, textvariable=self.input_path).pack(fill="x", pady=(4, 12))

        self._section(parent, "Output")
        ttk.Button(parent, text="Output Folder", command=self.pick_output).pack(fill="x", pady=4)
        ttk.Entry(parent, textvariable=self.output_path).pack(fill="x", pady=(4, 12))

        self._section(parent, "Super Resolution")
        ttk.Label(parent, text="Scale", style="Panel.TLabel").pack(anchor="w", pady=(2, 2))
        ttk.Combobox(parent, textvariable=self.sr_scale, values=(2, 3, 4), state="readonly").pack(fill="x")
        ttk.Checkbutton(parent, text="CLAHE local contrast", variable=self.sr_clahe).pack(anchor="w", pady=(8, 4))
        ttk.Label(parent, text="Sharpness", style="Panel.TLabel").pack(anchor="w", pady=(4, 2))
        ttk.Spinbox(parent, from_=0.0, to=1.5, increment=0.05, textvariable=self.sr_sharpness).pack(fill="x")
        ttk.Button(parent, text="Export Super Resolution", command=self.start_super_resolution).pack(fill="x", pady=(10, 4))
        ttk.Button(parent, text="Open SR Folder", command=self.open_sr_output).pack(fill="x", pady=(0, 12))

        self._section(parent, "Selection")
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill="x", pady=4)
        ttk.Radiobutton(row, text="Ruler", value="ruler", variable=self.mode, command=self.change_mode).pack(side="left")
        ttk.Radiobutton(row, text="Measure", value="measure", variable=self.mode, command=self.change_mode).pack(side="left", padx=12)
        ttk.Button(parent, text="Clear Points", command=self.clear_points).pack(fill="x", pady=4)

        ttk.Label(parent, text="Ruler distance (mm)", style="Panel.TLabel").pack(anchor="w", pady=(10, 2))
        ttk.Entry(parent, textvariable=self.real_mm).pack(fill="x")

        self._section(parent, "Detection Settings")
        ttk.Label(parent, text="Analysis profile", style="Panel.TLabel").pack(anchor="w", pady=(2, 2))
        profile_combo = ttk.Combobox(
            parent,
            textvariable=self.analysis_profile,
            values=("Balanced", "Sensitive", "Conservative"),
            state="readonly",
        )
        profile_combo.pack(fill="x")
        profile_combo.bind("<<ComboboxSelected>>", lambda _event: self.apply_analysis_profile())
        ttk.Button(parent, text="Apply Profile", command=self.apply_analysis_profile).pack(fill="x", pady=(6, 8))

        ttk.Label(parent, text="Detection display mode", style="Panel.TLabel").pack(anchor="w", pady=(10, 2))
        ttk.Combobox(
            parent,
            textvariable=self.detection_display_mode,
            values=("Points", "Boundaries"),
            state="readonly",
        ).pack(fill="x")

        ttk.Label(parent, text="Parameter mode", style="Panel.TLabel").pack(anchor="w", pady=(10, 2))
        parameter_mode = ttk.Combobox(parent, textvariable=self.parameter_mode, values=("Auto", "Manual"), state="readonly")
        parameter_mode.pack(fill="x")
        parameter_mode.bind("<<ComboboxSelected>>", lambda _event: self._parameter_mode_changed())
        ttk.Label(parent, textvariable=self.parameter_status, style="Panel.TLabel", wraplength=300).pack(anchor="w", pady=(4, 0))

        ttk.Label(parent, text="Band width (px)", style="Panel.TLabel").pack(anchor="w", pady=(10, 2))
        ttk.Spinbox(parent, from_=11, to=151, increment=2, textvariable=self.band_width).pack(fill="x")

        ttk.Label(parent, text="Minimum ring width (mm)", style="Panel.TLabel").pack(anchor="w", pady=(10, 2))
        ttk.Spinbox(parent, from_=0.01, to=2.0, increment=0.01, textvariable=self.min_ring_mm).pack(fill="x")

        ttk.Label(parent, text="Adaptive Block", style="Panel.TLabel").pack(anchor="w", pady=(10, 2))
        ttk.Spinbox(parent, from_=15, to=151, increment=2, textvariable=self.adaptive_block).pack(fill="x")

        ttk.Label(parent, text="Adaptive C mode", style="Panel.TLabel").pack(anchor="w", pady=(10, 2))
        c_mode = ttk.Combobox(parent, textvariable=self.adaptive_c_mode, values=("Auto (Otsu)", "Manual"), state="readonly")
        c_mode.pack(fill="x")
        c_mode.bind("<<ComboboxSelected>>", lambda _event: self._adaptive_c_mode_changed())

        ttk.Label(parent, text="Adaptive C", style="Panel.TLabel").pack(anchor="w", pady=(8, 2))
        ttk.Spinbox(parent, from_=1, to=15, increment=1, textvariable=self.adaptive_c).pack(fill="x")
        ttk.Label(parent, textvariable=self.adaptive_c_status, style="Panel.TLabel", wraplength=300).pack(anchor="w", pady=(4, 0))

        ttk.Button(parent, text="Run Analysis", style="Accent.TButton", command=self.start_analysis).pack(fill="x", pady=(18, 6))
        ttk.Button(parent, text="Edit Detection Points", command=self.open_detection_editor).pack(fill="x", pady=4)
        ttk.Button(parent, text="Export 3D Model", command=self.export_current_3d_model).pack(fill="x", pady=4)
        ttk.Button(parent, text="Open Results Folder", command=self.open_output).pack(fill="x", pady=4)

        self._section(parent, "Statistical Comparison")
        ttk.Button(parent, text="Load Excel / CSV Data", command=self.load_stats_file).pack(fill="x", pady=4)
        ttk.Button(parent, text="Manual Data Entry", command=self.open_manual_stats_entry).pack(fill="x", pady=4)
        ttk.Checkbutton(parent, text="Auto remove outliers", variable=self.stats_auto_remove_outliers).pack(anchor="w", pady=(8, 2))
        ttk.Button(parent, text="Run Statistical Analysis", style="Accent.TButton", command=self.run_statistical_analysis).pack(fill="x", pady=(8, 4))
        ttk.Button(parent, text="Open Statistics Folder", command=self.open_stats_output).pack(fill="x", pady=(0, 8))

    def _section(self, parent, text):
        ttk.Label(parent, text=text, style="Section.TLabel").pack(anchor="w", pady=(12, 6))

    def apply_analysis_profile(self):
        profile = self.analysis_profile.get()
        self.sensitivity.set("low")
        self.adaptive_block.set(51)
        self.adaptive_c.set(5)
        if profile == "Sensitive":
            self.min_ring_mm.set(0.15)
            self.band_width.set(25)
            self.adaptive_block.set(41)
            self.adaptive_c.set(3)
            note = "Sensitive: higher skeleton sensitivity for narrow rings."
        elif profile == "Conservative":
            self.min_ring_mm.set(0.28)
            self.band_width.set(19)
            self.adaptive_block.set(61)
            self.adaptive_c.set(7)
            note = "Conservative: more selective detection to reduce over-counting."
        else:
            self.min_ring_mm.set(0.20)
            self.band_width.set(21)
            note = "Balanced: recommended starting point with moderate skeleton detection."
        if self.adaptive_c_mode.get() == "Auto (Otsu)":
            if len(self.measure_points) == 2 and self.current_image is not None:
                self._update_otsu_adaptive_c_from_measure_line(log_change=False)
            else:
                self.adaptive_c_status.set("Auto: select a Measure line")
        if self.parameter_mode.get() == "Auto":
            if not self._update_auto_detection_parameters(log_change=False):
                self.parameter_status.set("Auto: select Ruler and Measure lines")
        self.log_queue.put(note)
        self._set_summary(note)

    def _parameter_mode_changed(self):
        if self.parameter_mode.get() == "Auto":
            if not self._update_auto_detection_parameters(log_change=True):
                self.parameter_status.set("Auto: select Ruler and Measure lines")
        else:
            self.auto_parameter_info = {"parameter_mode": "Manual"}
            self.parameter_status.set("Manual: Band width, minimum ring width, and Adaptive Block are user controlled")

    def _odd_in_range(self, value, minimum, maximum):
        value = int(round(float(value)))
        if value % 2 == 0:
            value += 1
        minimum = int(minimum) if int(minimum) % 2 == 1 else int(minimum) + 1
        maximum = int(maximum) if int(maximum) % 2 == 1 else int(maximum) - 1
        return int(np.clip(value, minimum, maximum))

    def _pixel_to_mm_from_current_scale(self):
        if len(self.ruler_points) != 2:
            return None, None
        try:
            real_mm = float(self.real_mm.get().replace(",", "."))
            ruler_px = float(np.linalg.norm(self.ruler_points[1] - self.ruler_points[0]))
            if real_mm <= 0 or ruler_px <= 1e-6:
                return None, None
            return real_mm / ruler_px, real_mm
        except Exception:
            return None, None

    def _measure_band_profile(self, band_width):
        if self.current_image is None or len(self.measure_points) != 2:
            return None
        p1 = np.array(self.measure_points[0], dtype=np.float32)
        p2 = np.array(self.measure_points[1], dtype=np.float32)
        coords_main, band_pixels = sample_band_coords(self.current_image.shape, p1, p2, band_width)
        if len(coords_main) < 20 or len(band_pixels) < 32:
            return None

        filtered = cv2.bilateralFilter(self.current_image, d=9, sigmaColor=75, sigmaSpace=75)
        y_channel = cv2.cvtColor(filtered, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        y_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(y_channel)
        sums = np.zeros(len(coords_main), dtype=np.float32)
        counts = np.zeros(len(coords_main), dtype=np.float32)
        for xi, yi, i, _o in band_pixels:
            sums[i] += float(y_eq[yi, xi])
            counts[i] += 1.0
        valid = counts > 0
        if not np.any(valid):
            return None
        profile = sums[valid] / counts[valid]
        return profile.astype(np.float32), int(len(band_pixels)), y_eq

    def _estimate_peak_spacing_px(self, profile):
        if profile is None or len(profile) < 20:
            return None, 0
        dark_profile = 255.0 - profile.astype(np.float32)
        smooth = cv2.GaussianBlur(dark_profile.reshape(-1, 1), (0, 0), sigmaX=2.0).ravel()
        if float(np.ptp(smooth)) < 1e-6:
            return None, 0

        threshold, _binary = cv2.threshold(
            np.clip(smooth, 0, 255).astype(np.uint8).reshape(-1, 1),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        floor = max(float(threshold), float(np.mean(smooth) + 0.15 * np.std(smooth)))
        candidates = []
        for i in range(1, len(smooth) - 1):
            if smooth[i] >= floor and smooth[i] >= smooth[i - 1] and smooth[i] > smooth[i + 1]:
                candidates.append((i, float(smooth[i])))

        merged = []
        min_gap = 3
        for idx, score in candidates:
            if not merged or idx - merged[-1][0] >= min_gap:
                merged.append((idx, score))
            elif score > merged[-1][1]:
                merged[-1] = (idx, score)

        if len(merged) < 3:
            return None, len(merged)
        peaks = np.array([idx for idx, _score in merged], dtype=np.float32)
        spacings = np.diff(peaks)
        spacings = spacings[(spacings >= 3) & (spacings <= max(4, len(profile) * 0.35))]
        if len(spacings) == 0:
            return None, len(merged)
        return float(np.percentile(spacings, 35)), len(merged)

    def _update_auto_detection_parameters(self, log_change=False):
        if self.parameter_mode.get() != "Auto":
            return False
        pixel_to_mm, real_mm = self._pixel_to_mm_from_current_scale()
        if self.current_image is None or len(self.measure_points) != 2 or pixel_to_mm is None:
            return False

        nominal_band = self._odd_in_range(0.90 / pixel_to_mm, 11, 61)
        profile_result = self._measure_band_profile(nominal_band)
        if profile_result is None:
            self.parameter_status.set("Auto: Measure band is too short")
            return False
        profile, sample_count, _y_eq = profile_result
        spacing_px, preliminary_peak_count = self._estimate_peak_spacing_px(profile)
        if spacing_px is None:
            spacing_px = max(6.0, 0.35 / pixel_to_mm)

        band_width_px = self._odd_in_range(min(nominal_band, spacing_px * 1.8), 11, 61)
        min_ring_mm = float(np.clip(spacing_px * pixel_to_mm * 0.35, 0.03, 2.0))
        adaptive_block = self._odd_in_range(spacing_px * 3.0, 15, 151)

        self.band_width.set(int(band_width_px))
        self.min_ring_mm.set(round(min_ring_mm, 3))
        self.adaptive_block.set(int(adaptive_block))
        self.auto_parameter_info = {
            "parameter_mode": "Auto",
            "auto_band_width_px": int(band_width_px),
            "auto_min_ring_mm": float(round(min_ring_mm, 3)),
            "auto_adaptive_block": int(adaptive_block),
            "auto_spacing_px_robust": float(spacing_px),
            "auto_spacing_mm_robust": float(spacing_px * pixel_to_mm),
            "auto_spacing_px_median": float(spacing_px),
            "auto_spacing_mm_median": float(spacing_px * pixel_to_mm),
            "auto_preliminary_peak_count": int(preliminary_peak_count),
            "auto_profile_sample_count": int(sample_count),
            "auto_scale_mm": float(real_mm),
        }
        self.parameter_status.set(
            f"Auto: band {int(band_width_px)} px, min ring {min_ring_mm:.3f} mm, block {int(adaptive_block)}"
        )
        if self.adaptive_c_mode.get() == "Auto (Otsu)":
            self._update_otsu_adaptive_c_from_measure_line(log_change=False)
        if log_change:
            self.log_queue.put(
                f"Auto parameters estimated: band={int(band_width_px)} px, min_ring={min_ring_mm:.3f} mm, block={int(adaptive_block)}"
            )
        return True

    def _adaptive_c_mode_changed(self):
        if self.adaptive_c_mode.get() == "Auto (Otsu)":
            updated = self._update_otsu_adaptive_c_from_measure_line(log_change=True)
            if not updated:
                self.adaptive_c_status.set("Auto: select a Measure line")
        else:
            self.otsu_info = {}
            self.adaptive_c_status.set("Manual: Adaptive C is controlled by the user")

    def _update_otsu_adaptive_c_from_measure_line(self, log_change=False):
        if self.current_image is None or len(self.measure_points) != 2:
            return False
        try:
            band_width = int(self.band_width.get())
        except Exception:
            band_width = 21

        try:
            _coords_main, band_pixels = sample_band_coords(
                self.current_image.shape,
                np.array(self.measure_points[0], dtype=np.float32),
                np.array(self.measure_points[1], dtype=np.float32),
                band_width,
            )
            if len(band_pixels) < 32:
                self.adaptive_c_status.set("Auto: Measure band is too short for Otsu")
                return False

            filtered = cv2.bilateralFilter(self.current_image, d=9, sigmaColor=75, sigmaSpace=75)
            y_channel = cv2.cvtColor(filtered, cv2.COLOR_BGR2YCrCb)[:, :, 0]
            y_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(y_channel)
            values = np.array([y_eq[yi, xi] for xi, yi, _i, _o in band_pixels], dtype=np.uint8)
            otsu_threshold, _binary = cv2.threshold(
                values.reshape(-1, 1),
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )

            local_mean = float(np.mean(values))
            local_median = float(np.median(values))
            local_std = float(np.std(values))
            raw_c = local_mean - float(otsu_threshold)
            estimated_c = int(np.clip(round(raw_c * 0.45), 1, 10))
            self.otsu_info = {
                "adaptive_c_mode": self.adaptive_c_mode.get(),
                "adaptive_c_estimated": estimated_c,
                "adaptive_c_raw": float(raw_c),
                "otsu_threshold": float(otsu_threshold),
                "otsu_band_mean": local_mean,
                "otsu_band_median": local_median,
                "otsu_band_std": local_std,
                "otsu_sample_count": int(len(values)),
            }

            if self.adaptive_c_mode.get() == "Auto (Otsu)":
                self.adaptive_c.set(estimated_c)
            self.adaptive_c_status.set(
                f"Auto Otsu C: {estimated_c} | threshold {float(otsu_threshold):.1f}, band mean {local_mean:.1f}"
            )
            if log_change:
                self.log_queue.put(
                    f"Otsu Adaptive C estimated from Measure band: C={estimated_c}, threshold={float(otsu_threshold):.1f}"
                )
            return True
        except Exception as exc:
            self.adaptive_c_status.set(f"Auto Otsu unavailable: {exc}")
            return False

    def _effective_adaptive_c(self):
        if self.adaptive_c_mode.get() == "Auto (Otsu)":
            self._update_otsu_adaptive_c_from_measure_line(log_change=False)
        return int(self.adaptive_c.get())

    def pick_image(self):
        path = filedialog.askopenfilename(title="Select image", filetypes=[(SUPPORTED_INPUT, "*.jpg *.jpeg *.png *.tif *.tiff *.bmp")])
        if path:
            self._load_paths([path])

    def pick_folder(self):
        folder = filedialog.askdirectory(title="Select image folder")
        if folder:
            paths = find_images(folder)
            if not paths:
                messagebox.showwarning(APP_TITLE, "No supported images were found in this folder.")
                return
            self._load_paths(paths)

    def pick_output(self):
        folder = filedialog.askdirectory(title="Select output folder")
        if folder:
            self.output_path.set(folder)

    def _load_paths(self, paths):
        self.images = paths
        self.current_index = 0
        self.input_path.set(paths[0] if len(paths) == 1 else str(Path(paths[0]).parent))
        self._load_current_image()

    def _load_current_image(self):
        self.current_image = read_image(self.images[self.current_index])
        self.ruler_points = []
        self.measure_points = []
        self.auto_parameter_info = {}
        self.otsu_info = {}
        self.parameter_status.set("Auto: select Ruler and Measure lines")
        self.adaptive_c_status.set("Auto: select a Measure line")
        self.canvas.set_mode("ruler")
        self.mode.set("ruler")
        self.canvas.set_image(self.current_image)
        self.status.set(f"Loaded: {self.current_index + 1}/{len(self.images)} - {Path(self.images[self.current_index]).name}")

    def previous_image(self):
        if not self.images:
            messagebox.showwarning(APP_TITLE, "Select an image or folder first.")
            return
        self.current_index = (self.current_index - 1) % len(self.images)
        self._load_current_image()

    def next_image(self):
        if not self.images:
            messagebox.showwarning(APP_TITLE, "Select an image or folder first.")
            return
        self.current_index = (self.current_index + 1) % len(self.images)
        self._load_current_image()

    def change_mode(self):
        if self.canvas.mode == "ruler":
            self.ruler_points = self.canvas.get_points()
        else:
            self.measure_points = self.canvas.get_points()
        if self.parameter_mode.get() == "Auto":
            self._update_auto_detection_parameters(log_change=False)
        new_mode = self.mode.get()
        self.canvas.mode = new_mode
        self.canvas.points = [tuple(p) for p in (self.ruler_points if new_mode == "ruler" else self.measure_points)]
        self.canvas.render()

    def clear_points(self):
        if self.mode.get() == "ruler":
            self.ruler_points = []
        else:
            self.measure_points = []
            self.auto_parameter_info = {}
            self.otsu_info = {}
            if self.parameter_mode.get() == "Auto":
                self.parameter_status.set("Auto: select Ruler and Measure lines")
            if self.adaptive_c_mode.get() == "Auto (Otsu)":
                self.adaptive_c_status.set("Auto: select a Measure line")
        self.canvas.reset_points()

    def _points_changed(self):
        if not hasattr(self, "canvas"):
            return
        if self.canvas.mode == "ruler":
            self.ruler_points = self.canvas.get_points()
        else:
            self.measure_points = self.canvas.get_points()
        auto_updated = False
        if self.parameter_mode.get() == "Auto":
            if len(self.ruler_points) == 2 and len(self.measure_points) == 2:
                auto_updated = self._update_auto_detection_parameters(log_change=True)
            elif len(self.ruler_points) < 2 or len(self.measure_points) < 2:
                self.auto_parameter_info = {}
                self.parameter_status.set("Auto: select Ruler and Measure lines")
        if self.canvas.mode != "ruler":
            if len(self.measure_points) == 2 and self.adaptive_c_mode.get() == "Auto (Otsu)":
                self._update_otsu_adaptive_c_from_measure_line(log_change=not auto_updated)
            elif len(self.measure_points) < 2 and self.adaptive_c_mode.get() == "Auto (Otsu)":
                self.otsu_info = {}
                self.adaptive_c_status.set("Auto: select a Measure line")

    def start_analysis(self):
        if not self.images:
            messagebox.showwarning(APP_TITLE, "Select an image first.")
            return
        self._points_changed()
        if len(self.ruler_points) != 2 or len(self.measure_points) != 2:
            messagebox.showwarning(APP_TITLE, "Select two points for the ruler and two points for the measurement line.")
            return
        try:
            real_mm = float(self.real_mm.get().replace(",", "."))
            if real_mm <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning(APP_TITLE, "Ruler distance must be a positive number.")
            return

        if self.parameter_mode.get() == "Auto":
            self._update_auto_detection_parameters(log_change=False)
        self._effective_adaptive_c()
        self.status.set("Analysis running...")
        self.log_queue.put("Analysis started.")
        worker = threading.Thread(target=self._process_current, args=(real_mm,), daemon=True)
        worker.start()

    def start_super_resolution(self):
        if not self.images or self.current_image is None:
            messagebox.showwarning(APP_TITLE, "Select an image first.")
            return
        try:
            scale = int(self.sr_scale.get())
            sharpness = float(self.sr_sharpness.get())
            if scale not in (2, 3, 4) or not (0 <= sharpness <= 1.5):
                raise ValueError
        except ValueError:
            messagebox.showwarning(APP_TITLE, "SR scale must be 2/3/4 and sharpness must be between 0.0 and 1.5.")
            return

        self.status.set("Super Resolution running...")
        self.log_queue.put("Super Resolution started.")
        worker = threading.Thread(target=self._process_super_resolution, args=(scale, sharpness, bool(self.sr_clahe.get())), daemon=True)
        worker.start()

    def _process_super_resolution(self, scale, sharpness, use_clahe):
        try:
            path = Path(self.images[self.current_index])
            out_dir = Path(self.output_path.get()) / "SuperResolution" / path.stem
            ensure_dir(out_dir)
            result_path, comparison_path = self._create_super_resolution_outputs(
                self.current_image.copy(),
                out_dir,
                path.stem,
                scale,
                sharpness,
                use_clahe,
            )
            method_text = self._super_resolution_method_text(path.name, scale, sharpness, use_clahe, result_path)
            (out_dir / "super_resolution_method.txt").write_text(method_text, encoding="utf-8")
            self.log_queue.put(f"SR output: {result_path}")
            self.log_queue.put(f"SR comparison: {comparison_path}")
            self.after(0, lambda text=method_text: self._set_summary(text))
            self.after(0, lambda: self.status.set("Super Resolution completed"))
        except Exception as exc:
            self.log_queue.put(f"SR ERROR: {exc}")
            self.after(0, lambda: self.status.set("Super Resolution error"))

    def _create_super_resolution_outputs(self, img, out_dir, stem, scale, sharpness, use_clahe):
        upscaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(str(out_dir / f"{stem}_SR_x{scale}_lanczos.png"), upscaled)

        work = cv2.bilateralFilter(upscaled, d=5, sigmaColor=35, sigmaSpace=35)
        if use_clahe:
            lab = cv2.cvtColor(work, cv2.COLOR_BGR2Lab)
            l_channel, a_channel, b_channel = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
            work = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_Lab2BGR)

        blur = cv2.GaussianBlur(work, (0, 0), sigmaX=1.0)
        enhanced = cv2.addWeighted(work, 1.0 + sharpness, blur, -sharpness, 0)
        result_path = out_dir / f"{stem}_SR_x{scale}_enhanced.png"
        cv2.imwrite(str(result_path), enhanced)

        comparison = self._make_sr_comparison(upscaled, enhanced)
        comparison_path = out_dir / f"{stem}_SR_x{scale}_comparison.png"
        cv2.imwrite(str(comparison_path), comparison)
        return result_path, comparison_path

    def _make_sr_comparison(self, left, right):
        max_w = 1100
        each_w = max_w // 2
        scale = min(each_w / left.shape[1], 1.0)
        display_h = max(int(left.shape[0] * scale), 1)
        display_w = max(int(left.shape[1] * scale), 1)
        left_small = cv2.resize(left, (display_w, display_h), interpolation=cv2.INTER_AREA)
        right_small = cv2.resize(right, (display_w, display_h), interpolation=cv2.INTER_AREA)
        cv2.putText(left_small, "Lanczos x SR", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(left_small, "Lanczos x SR", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(right_small, "Enhanced SR", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(right_small, "Enhanced SR", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
        return cv2.hconcat([left_small, right_small])

    def _super_resolution_method_text(self, image_name, scale, sharpness, use_clahe, result_path):
        clahe_text = "enabled" if use_clahe else "disabled"
        return (
            f"Super Resolution Output\n"
            f"Image: {image_name}\n"
            f"Output: {result_path}\n"
            f"Scale: {scale}x\n"
            f"CLAHE: {clahe_text}\n"
            f"Sharpness: {sharpness:.2f}\n\n"
            "Method: Lanczos windowed-sinc interpolation, optional CLAHE local contrast enhancement, bilateral "
            "smoothing, and controlled unsharp-mask sharpening.\n\n"
            "Scientific basis: this is a reproducible classical preprocessing workflow, not a deep model that "
            "hallucinates new anatomical details. Lanczos/sinc resampling is used in image sampling literature, "
            "CLAHE is common for local contrast enhancement, and unsharp masking is widely used for microscopy "
            "and edge-visibility preprocessing. Visually inspect the SR image before analysis, then reload the "
            "new PNG file and run detection on it."
        )

    def load_stats_file(self):
        path = filedialog.askopenfilename(
            title="Load manual and analysis result data",
            filetypes=[
                ("Excel / CSV files", "*.xlsx *.xls *.csv"),
                ("Excel files", "*.xlsx *.xls"),
                ("CSV files", "*.csv"),
            ],
        )
        if not path:
            return
        try:
            manual, analysis = self._read_stats_table(path)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Could not read comparison data:\n{exc}")
            return
        self.stats_manual_values = manual
        self.stats_analysis_values = analysis
        self.stats_source = str(path)
        self.log_queue.put(f"Loaded statistical data: {len(manual)} paired rows from {path}")
        self._set_summary(
            "Statistical data loaded.\n"
            f"Source: {path}\n"
            f"Paired rows: {len(manual)}\n\n"
            "Click Run Statistical Analysis to create the validation report."
        )

    def open_manual_stats_entry(self):
        window = tk.Toplevel(self)
        window.title("Manual Statistical Data Entry")
        window.geometry("760x620")
        window.configure(bg="#0f172a")
        window.transient(self)

        frame = ttk.Frame(window, padding=14)
        frame.pack(fill="both", expand=True)
        ttk.Label(
            frame,
            text="Use one row per sample. Fill both columns, or paste a two-column table from Excel.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(0, 10))

        table_shell = ttk.Frame(frame)
        table_shell.pack(fill="both", expand=True)
        table_canvas = tk.Canvas(table_shell, bg="#0b1220", highlightthickness=0)
        table_scroll = ttk.Scrollbar(table_shell, orient="vertical", command=table_canvas.yview)
        table_canvas.configure(yscrollcommand=table_scroll.set)
        table_canvas.pack(side="left", fill="both", expand=True)
        table_scroll.pack(side="right", fill="y")

        table = ttk.Frame(table_canvas, padding=10)
        table_window = table_canvas.create_window((0, 0), window=table, anchor="nw")
        table.bind("<Configure>", lambda _event: table_canvas.configure(scrollregion=table_canvas.bbox("all")))
        table_canvas.bind("<Configure>", lambda event: table_canvas.itemconfigure(table_window, width=event.width))
        table_canvas.bind("<MouseWheel>", lambda event: table_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units"))

        ttk.Label(table, text="#", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(0, 6))
        ttk.Label(table, text="Manual Measurement", font=("Segoe UI", 10, "bold")).grid(row=0, column=1, sticky="ew", padx=6, pady=(0, 6))
        ttk.Label(table, text="Analysis Result", font=("Segoe UI", 10, "bold")).grid(row=0, column=2, sticky="ew", padx=6, pady=(0, 6))
        table.columnconfigure(1, weight=1)
        table.columnconfigure(2, weight=1)

        row_entries = []

        def add_rows(count, manual_values=None, analysis_values=None):
            manual_values = manual_values or []
            analysis_values = analysis_values or []
            start = len(row_entries)
            for offset in range(count):
                row_index = start + offset
                ttk.Label(table, text=str(row_index + 1)).grid(row=row_index + 1, column=0, sticky="w", padx=(0, 8), pady=3)
                manual_entry = ttk.Entry(table)
                analysis_entry = ttk.Entry(table)
                manual_entry.grid(row=row_index + 1, column=1, sticky="ew", padx=6, pady=3)
                analysis_entry.grid(row=row_index + 1, column=2, sticky="ew", padx=6, pady=3)
                if offset < len(manual_values):
                    manual_entry.insert(0, f"{manual_values[offset]:g}")
                if offset < len(analysis_values):
                    analysis_entry.insert(0, f"{analysis_values[offset]:g}")
                row_entries.append((manual_entry, analysis_entry))

        initial_count = max(30, len(self.stats_manual_values), len(self.stats_analysis_values))
        add_rows(initial_count, self.stats_manual_values, self.stats_analysis_values)

        def parse_entry_value(value):
            value = value.strip().replace(",", ".")
            return float(value) if value else None

        def paste_table():
            try:
                text = window.clipboard_get()
            except tk.TclError:
                messagebox.showwarning(APP_TITLE, "Clipboard is empty.")
                return
            rows = []
            for line in text.splitlines():
                numbers = self._parse_number_series(line)
                if len(numbers) >= 2:
                    rows.append((numbers[0], numbers[1]))
            if not rows:
                messagebox.showwarning(APP_TITLE, "Clipboard must contain at least two numeric columns.")
                return
            while len(row_entries) < len(rows):
                add_rows(10)
            for index, (manual_value, analysis_value) in enumerate(rows):
                manual_entry, analysis_entry = row_entries[index]
                manual_entry.delete(0, "end")
                analysis_entry.delete(0, "end")
                manual_entry.insert(0, f"{manual_value:g}")
                analysis_entry.insert(0, f"{analysis_value:g}")

        def accept():
            manual = []
            analysis = []
            for manual_entry, analysis_entry in row_entries:
                manual_value = parse_entry_value(manual_entry.get())
                analysis_value = parse_entry_value(analysis_entry.get())
                if manual_value is None and analysis_value is None:
                    continue
                if manual_value is None or analysis_value is None:
                    messagebox.showwarning(APP_TITLE, "Every used row must contain both Manual Measurement and Analysis Result.")
                    return
                manual.append(manual_value)
                analysis.append(analysis_value)
            if not manual or not analysis:
                messagebox.showwarning(APP_TITLE, "Enter at least one value in both columns.")
                return
            if len(manual) != len(analysis):
                messagebox.showwarning(APP_TITLE, "Manual Measurement and Analysis Result must contain the same number of values.")
                return
            self.stats_manual_values = manual
            self.stats_analysis_values = analysis
            self.stats_source = "Manual entry"
            self.log_queue.put(f"Manual statistical data entered: {len(manual)} paired rows")
            window.destroy()
            self.run_statistical_analysis()

        actions = ttk.Frame(frame)
        actions.pack(fill="x", pady=(12, 0))
        ttk.Button(actions, text="Add 10 Rows", command=lambda: add_rows(10)).pack(side="left")
        ttk.Button(actions, text="Paste Table", command=paste_table).pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="Run Statistical Analysis", style="Accent.TButton", command=accept).pack(side="right")
        ttk.Button(actions, text="Close", command=window.destroy).pack(side="right", padx=(0, 8))

    def _parse_number_series(self, text):
        matches = re.findall(r"[-+]?\d+(?:[.,]\d+)?", text)
        return [float(value.replace(",", ".")) for value in matches]

    def _normalize_column_name(self, name):
        text = str(name).strip().casefold()
        text = text.translate(str.maketrans({
            "ı": "i", "ğ": "g", "ü": "u", "ş": "s", "ö": "o", "ç": "c",
            "İ": "i", "Ğ": "g", "Ü": "u", "Ş": "s", "Ö": "o", "Ç": "c",
        }))
        return re.sub(r"[^a-z0-9]+", "", text)

    def _read_stats_table(self, path):
        path = Path(path)
        if path.suffix.lower() == ".csv":
            data = pd.read_csv(path)
        else:
            data = pd.read_excel(path)
        if data.empty:
            raise ValueError("The selected file is empty.")

        normalized = {self._normalize_column_name(col): col for col in data.columns}
        manual_candidates = {
            "manualmeasurement", "manualmeasurements", "manual", "manuelyolcum", "manuelolcum",
            "manualage", "reference", "groundtruth", "usermeasurement", "userage",
        }
        analysis_candidates = {
            "analysisresult", "analysisresults", "analysis", "analizsonucu", "detectedage",
            "detected", "airesult", "softwareage", "treeringai", "estimatedage",
        }

        manual_col = next((normalized[key] for key in normalized if key in manual_candidates), None)
        analysis_col = next((normalized[key] for key in normalized if key in analysis_candidates), None)
        if manual_col is None or analysis_col is None:
            available = ", ".join(str(col) for col in data.columns)
            raise ValueError(
                "Expected columns named Manual Measurement and Analysis Result "
                "(also accepts manuel olcum / analiz sonucu). "
                f"Found columns: {available}"
            )

        paired = pd.DataFrame({
            "manual_measurement": pd.to_numeric(data[manual_col], errors="coerce"),
            "analysis_result": pd.to_numeric(data[analysis_col], errors="coerce"),
        }).dropna()
        if paired.empty:
            raise ValueError("No numeric paired rows were found in the selected columns.")
        return paired["manual_measurement"].astype(float).tolist(), paired["analysis_result"].astype(float).tolist()

    def run_statistical_analysis(self):
        if not self.stats_manual_values or not self.stats_analysis_values:
            messagebox.showwarning(APP_TITLE, "Load Excel/CSV data or enter values manually first.")
            return
        if len(self.stats_manual_values) != len(self.stats_analysis_values):
            messagebox.showwarning(APP_TITLE, "Manual Measurement and Analysis Result must contain the same number of values.")
            return
        if len(self.stats_manual_values) < 2:
            messagebox.showwarning(APP_TITLE, "At least two paired values are required for statistical analysis.")
            return

        self.status.set("Statistical analysis running...")
        worker = threading.Thread(target=self._process_statistical_analysis, daemon=True)
        worker.start()

    def _process_statistical_analysis(self):
        try:
            manual = np.asarray(self.stats_manual_values, dtype=float)
            analysis = np.asarray(self.stats_analysis_values, dtype=float)
            out_dir = Path(self.output_path.get()) / "StatisticalAnalysis" / datetime.now().strftime("%Y%m%d_%H%M%S")
            ensure_dir(out_dir)

            raw = pd.DataFrame({
                "sample_no": np.arange(1, len(manual) + 1),
                "manual_measurement": manual,
                "analysis_result": analysis,
            })
            raw["difference_analysis_minus_manual"] = raw["analysis_result"] - raw["manual_measurement"]
            raw["absolute_error"] = raw["difference_analysis_minus_manual"].abs()
            raw["absolute_percentage_error"] = np.where(
                np.abs(raw["manual_measurement"]) > 1e-12,
                raw["absolute_error"] / np.abs(raw["manual_measurement"]) * 100,
                np.nan,
            )
            raw, used, removed, correction = self._apply_outlier_correction(raw)

            manual = used["manual_measurement"].to_numpy(dtype=float)
            analysis = used["analysis_result"].to_numpy(dtype=float)
            diff = used["difference_analysis_minus_manual"].to_numpy(dtype=float)
            abs_error = used["absolute_error"].to_numpy(dtype=float)
            percent_error = used["absolute_percentage_error"].dropna().to_numpy(dtype=float)
            if np.std(manual) > 1e-12 and np.std(analysis) > 1e-12:
                r2 = float(np.corrcoef(manual, analysis)[0, 1] ** 2)
                r2 = min(max(r2, 0.0), 1.0)
            else:
                r2 = 0.0

            metrics = {
                "total_samples": int(len(raw)),
                "paired_samples": int(len(used)),
                "excluded_samples": int(len(removed)),
                "outlier_correction": "Enabled" if correction["enabled"] else "Disabled",
                "outlier_method": correction["method"],
                "outlier_threshold": correction["threshold"],
                "r2": float(r2),
                "mae": float(np.mean(abs_error)),
                "rmse": float(np.sqrt(np.mean(diff ** 2))),
                "mape_percent": float(np.mean(percent_error)) if len(percent_error) else np.nan,
                "source": self.stats_source,
            }

            summary_text = self._format_stats_summary(metrics, out_dir)
            (out_dir / "statistical_summary.txt").write_text(summary_text, encoding="utf-8")
            raw.to_csv(out_dir / "statistical_comparison_raw.csv", index=False, encoding="utf-8-sig")
            used.to_csv(out_dir / "statistical_comparison_used.csv", index=False, encoding="utf-8-sig")
            if len(removed):
                removed.to_csv(out_dir / "statistical_comparison_removed_outliers.csv", index=False, encoding="utf-8-sig")
            with pd.ExcelWriter(out_dir / "statistical_comparison.xlsx", engine="openpyxl") as writer:
                used.rename(columns={
                    "sample_no": "Sample No",
                    "manual_measurement": "Manual Measurement",
                    "analysis_result": "Analysis Result",
                    "difference_analysis_minus_manual": "Difference (Analysis - Manual)",
                    "absolute_error": "Absolute Error",
                    "absolute_percentage_error": "Absolute Percentage Error (%)",
                    "used_for_statistics": "Used For Statistics",
                    "outlier_reason": "Outlier Reason",
                }).to_excel(writer, sheet_name="paired_data_used", index=False)
                removed.rename(columns={
                    "sample_no": "Sample No",
                    "manual_measurement": "Manual Measurement",
                    "analysis_result": "Analysis Result",
                    "difference_analysis_minus_manual": "Difference (Analysis - Manual)",
                    "absolute_error": "Absolute Error",
                    "absolute_percentage_error": "Absolute Percentage Error (%)",
                    "used_for_statistics": "Used For Statistics",
                    "outlier_reason": "Outlier Reason",
                }).to_excel(writer, sheet_name="removed_outliers", index=False)
                raw.rename(columns={
                    "sample_no": "Sample No",
                    "manual_measurement": "Manual Measurement",
                    "analysis_result": "Analysis Result",
                    "difference_analysis_minus_manual": "Difference (Analysis - Manual)",
                    "absolute_error": "Absolute Error",
                    "absolute_percentage_error": "Absolute Percentage Error (%)",
                    "used_for_statistics": "Used For Statistics",
                    "outlier_reason": "Outlier Reason",
                }).to_excel(writer, sheet_name="all_paired_data", index=False)
                self._stats_summary_table(metrics).to_excel(writer, sheet_name="summary_metrics", index=False)
                self._format_excel_workbook(writer.book)

            self._create_statistical_charts(used, metrics, out_dir)
            self.log_queue.put(f"Statistical report created: {out_dir}")
            self.after(0, lambda text=summary_text: self._set_summary(text))
            self.after(0, lambda: self.status.set("Statistical analysis completed"))
        except Exception as exc:
            self.log_queue.put(f"STATISTICS ERROR: {exc}")
            self.after(0, lambda: self.status.set("Statistical analysis error"))

    def _apply_outlier_correction(self, raw):
        raw = raw.copy()
        raw["used_for_statistics"] = True
        raw["outlier_reason"] = ""
        correction = {
            "enabled": bool(self.stats_auto_remove_outliers.get()),
            "method": "None",
            "threshold": np.nan,
        }
        if not correction["enabled"] or len(raw) < 3:
            return raw, raw.copy(), raw.iloc[0:0].copy(), correction

        multiplier = 3.0

        errors = raw["absolute_error"].to_numpy(dtype=float)
        median_error = float(np.median(errors))
        mad = float(np.median(np.abs(errors - median_error)))
        if mad > 1e-12:
            robust_sigma = 1.4826 * mad
            threshold = median_error + multiplier * robust_sigma
            method = f"absolute error <= median + {multiplier:g} x MAD"
        else:
            higher_errors = errors[errors > median_error + 1e-12]
            max_zero_mad_outliers = max(1, int(np.ceil(len(errors) * 0.25)))
            if 0 < len(higher_errors) <= max_zero_mad_outliers:
                threshold = median_error
                method = "absolute error <= median error (zero MAD)"
            else:
                std_error = float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0
                if std_error <= 1e-12:
                    correction["method"] = "No error variation; no outliers removed"
                    correction["threshold"] = float(np.max(errors)) if len(errors) else np.nan
                    return raw, raw.copy(), raw.iloc[0:0].copy(), correction
                mean_error = float(np.mean(errors))
                threshold = mean_error + multiplier * std_error
                method = f"absolute error <= mean + {multiplier:g} x SD"

        outlier_mask = raw["absolute_error"] > threshold
        if int(outlier_mask.sum()) == 0 or int((~outlier_mask).sum()) < 2:
            correction["method"] = method + " (no safe removal)"
            correction["threshold"] = float(threshold)
            return raw, raw.copy(), raw.iloc[0:0].copy(), correction

        raw.loc[outlier_mask, "used_for_statistics"] = False
        raw.loc[outlier_mask, "outlier_reason"] = f"absolute_error > {threshold:.4f}"
        used = raw[raw["used_for_statistics"]].copy()
        removed = raw[~raw["used_for_statistics"]].copy()
        correction["method"] = method
        correction["threshold"] = float(threshold)
        return raw, used, removed, correction

    def _format_stats_summary(self, metrics, out_dir):
        def fmt(value, decimals=3):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return "-"
            return f"{value:.{decimals}f}"

        return (
            "Statistical Comparison Report\n"
            f"Source: {metrics.get('source') or '-'}\n"
            f"Output folder: {out_dir}\n\n"
            f"Total samples: {metrics['total_samples']}\n"
            f"Used samples: {metrics['paired_samples']}\n"
            f"Removed outliers: {metrics['excluded_samples']}\n"
            f"Outlier correction: {metrics['outlier_correction']}\n"
            f"Outlier method: {metrics['outlier_method']}\n"
            f"Automatic outlier threshold: {fmt(metrics['outlier_threshold'])}\n"
            f"R2: {fmt(metrics['r2'], 4)}\n"
            f"MAE: {fmt(metrics['mae'])}\n"
            f"RMSE: {fmt(metrics['rmse'])}\n"
            f"MAPE: {fmt(metrics['mape_percent'])}%\n\n"
            "Generated files: statistical_comparison.xlsx, used/removed CSV files, statistical_summary.txt, metrics dashboard, scatter plot, and error distribution."
        )

    def _stats_summary_table(self, metrics):
        labels = {
            "total_samples": "Total Samples",
            "paired_samples": "Paired Samples",
            "excluded_samples": "Removed Outliers",
            "outlier_correction": "Outlier Correction",
            "outlier_method": "Outlier Method",
            "outlier_threshold": "Automatic Outlier Threshold",
            "r2": "R2",
            "mae": "MAE",
            "rmse": "RMSE",
            "mape_percent": "MAPE (%)",
            "source": "Source",
        }
        return pd.DataFrame([
            {"Metric": label, "Value": metrics.get(key)}
            for key, label in labels.items()
            if key in metrics
        ])

    def _format_excel_workbook(self, workbook):
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter

        header_fill = PatternFill("solid", fgColor="0F766E")
        header_font = Font(color="FFFFFF", bold=True)
        for sheet in workbook.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for cell in sheet[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            for row in sheet.iter_rows(min_row=2):
                for cell in row:
                    cell.alignment = Alignment(vertical="center", wrap_text=True)
                    if isinstance(cell.value, float):
                        cell.number_format = "0.0000"
            for column_cells in sheet.columns:
                max_length = 0
                column_letter = get_column_letter(column_cells[0].column)
                for cell in column_cells:
                    value = "" if cell.value is None else str(cell.value)
                    max_length = max(max_length, len(value))
                sheet.column_dimensions[column_letter].width = min(max(max_length + 3, 14), 42)

    def _create_statistical_charts(self, raw, metrics, out_dir):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        manual = raw["manual_measurement"].to_numpy(dtype=float)
        analysis = raw["analysis_result"].to_numpy(dtype=float)
        diff = raw["difference_analysis_minus_manual"].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(9, 5.5))
        labels = ["R2", "MAE", "RMSE", "MAPE"]
        values = [metrics["r2"], metrics["mae"], metrics["rmse"], metrics["mape_percent"]]
        colors = ["#0f766e", "#2563eb", "#f59e0b", "#ef4444"]
        bars = ax.bar(labels, values, color=colors, alpha=0.9)
        ax.set_title("Core Statistical Metrics")
        ax.set_ylabel("Metric value")
        ax.grid(True, axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            label = "-" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:.3f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha="center", va="bottom", fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / "01_core_metrics_dashboard.png", dpi=220)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(manual, analysis, s=58, color="#0f766e", edgecolor="#083344", alpha=0.88)
        min_v = float(min(np.min(manual), np.min(analysis)))
        max_v = float(max(np.max(manual), np.max(analysis)))
        pad = max((max_v - min_v) * 0.08, 1.0)
        ax.plot([min_v - pad, max_v + pad], [min_v - pad, max_v + pad], color="#ef4444", linewidth=2, label="Perfect agreement")
        if len(raw) > 1:
            coef = np.polyfit(manual, analysis, 1)
            fit_x = np.linspace(min_v - pad, max_v + pad, 100)
            ax.plot(fit_x, coef[0] * fit_x + coef[1], color="#2563eb", linewidth=2, label="Linear fit")
        ax.set_title("Manual Measurement vs Analysis Result")
        ax.set_xlabel("Manual Measurement")
        ax.set_ylabel("Analysis Result")
        ax.text(
            0.04,
            0.96,
            f"R2 = {metrics['r2']:.4f}\nMAE = {metrics['mae']:.3f}\nRMSE = {metrics['rmse']:.3f}\nMAPE = {metrics['mape_percent']:.3f}%",
            transform=ax.transAxes,
            va="top",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#ecfeff", "edgecolor": "#0891b2"},
        )
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "02_manual_vs_analysis_scatter.png", dpi=220)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5.5))
        bins = min(18, max(5, len(diff) // 2))
        ax.hist(diff, bins=bins, color="#f59e0b", edgecolor="#78350f", alpha=0.86)
        ax.axvline(0, color="#0f172a", linewidth=2, label="Zero error")
        ax.axvline(float(np.mean(diff)), color="#ef4444", linewidth=2, label="Mean error")
        ax.set_title("Error Distribution")
        ax.set_xlabel("Analysis - Manual")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "03_error_distribution.png", dpi=220)
        plt.close(fig)

    def _metric_block(self, original_values, corrected_values):
        original_values = np.asarray(original_values, dtype=float)
        corrected_values = np.asarray(corrected_values, dtype=float)
        n = min(len(original_values), len(corrected_values))
        result = {"n": int(n), "r2": np.nan, "mae": np.nan, "rmse": np.nan, "mape_percent": np.nan}
        if n == 0:
            return result

        original_values = original_values[:n]
        corrected_values = corrected_values[:n]
        diff = corrected_values - original_values
        abs_diff = np.abs(diff)
        with np.errstate(divide="ignore", invalid="ignore"):
            ape = np.where(np.abs(original_values) > 1e-12, abs_diff / np.abs(original_values) * 100, np.nan)

        if n > 1 and np.std(original_values) > 1e-12 and np.std(corrected_values) > 1e-12:
            result["r2"] = float(np.corrcoef(original_values, corrected_values)[0, 1] ** 2)
        result["mae"] = float(np.mean(abs_diff))
        result["rmse"] = float(np.sqrt(np.mean(diff ** 2)))
        result["mape_percent"] = float(np.nanmean(ape)) if not np.all(np.isnan(ape)) else np.nan
        return result

    def _write_box_obj(self, lines, vertices, x0, x1, y0, y1, z0, z1, material, name):
        index = len(vertices) + 1
        box_vertices = [
            (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
        ]
        vertices.extend(box_vertices)
        lines.append(f"g {name}")
        lines.append(f"usemtl {material}")
        faces = [
            (1, 2, 3, 4),
            (5, 8, 7, 6),
            (1, 5, 6, 2),
            (2, 6, 7, 3),
            (3, 7, 8, 4),
            (4, 8, 5, 1),
        ]
        for face in faces:
            lines.append("f " + " ".join(str(index + value - 1) for value in face))

    def _write_3d_model_viewer(self, viewer_path, model_data):
        model_json = json.dumps(model_data, ensure_ascii=True)
        html = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TreeRing Studio Interactive 3D Model</title>
<style>
  :root {
    --bg: #f7f8fb;
    --ink: #172033;
    --muted: #5b6475;
    --panel: #ffffff;
    --line: #d9dee8;
    --ring-a: #c89a55;
    --ring-b: #80552c;
    --boundary: #2266e3;
    --accent: #0f766e;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--ink);
    font-family: Arial, Helvetica, sans-serif;
  }
  header {
    padding: 18px 24px 10px;
    border-bottom: 1px solid var(--line);
    background: var(--panel);
  }
  h1 {
    margin: 0 0 6px;
    font-size: 24px;
    font-weight: 800;
    letter-spacing: 0;
  }
  .subtitle {
    color: var(--muted);
    font-size: 14px;
    font-weight: 600;
  }
  main {
    display: grid;
    grid-template-columns: minmax(520px, 1fr) 330px;
    gap: 16px;
    padding: 16px;
  }
  .stage, .panel {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 8px;
    box-shadow: 0 10px 24px rgba(19, 32, 54, 0.08);
  }
  .stage {
    min-height: 620px;
    position: relative;
    overflow: hidden;
  }
  canvas {
    width: 100%;
    height: 100%;
    display: block;
    cursor: pointer;
  }
  .hint {
    position: absolute;
    left: 18px;
    bottom: 14px;
    padding: 8px 10px;
    border-radius: 6px;
    background: rgba(255,255,255,0.86);
    border: 1px solid var(--line);
    color: var(--muted);
    font-size: 13px;
    font-weight: 700;
  }
  .panel {
    padding: 16px;
  }
  h2 {
    margin: 0 0 10px;
    font-size: 17px;
    font-weight: 800;
  }
  .readout {
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 16px;
    background: #fbfcff;
  }
  .readout-title {
    font-size: 18px;
    font-weight: 800;
    margin-bottom: 8px;
  }
  .row {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    padding: 6px 0;
    border-bottom: 1px solid #edf0f5;
    font-size: 14px;
  }
  .row:last-child { border-bottom: 0; }
  .key {
    color: var(--muted);
    font-weight: 700;
  }
  .value {
    text-align: right;
    font-weight: 800;
  }
  .legend-item {
    display: grid;
    grid-template-columns: 28px 1fr;
    gap: 10px;
    align-items: center;
    margin: 10px 0;
    color: var(--muted);
    font-size: 14px;
    font-weight: 700;
  }
  .swatch {
    width: 28px;
    height: 18px;
    border-radius: 4px;
    border: 1px solid rgba(0,0,0,0.18);
  }
  .swatch.ring-a { background: var(--ring-a); }
  .swatch.ring-b { background: var(--ring-b); }
  .swatch.boundary { background: var(--boundary); }
  .stats {
    margin-top: 16px;
    padding-top: 12px;
    border-top: 1px solid var(--line);
    color: var(--muted);
    font-size: 13px;
    font-weight: 700;
    line-height: 1.5;
  }
  @media (max-width: 900px) {
    main { grid-template-columns: 1fr; }
    .stage { min-height: 520px; }
  }
</style>
</head>
<body>
<header>
  <h1>TreeRing Studio Interactive 3D Model</h1>
  <div class="subtitle">Click a blue boundary ridge or a ring segment to inspect age and millimeter measurements.</div>
</header>
<main>
  <section class="stage">
    <canvas id="viewer"></canvas>
    <div class="hint">Drag is not required. Click model elements to inspect values.</div>
  </section>
  <aside class="panel">
    <h2>Selected Element</h2>
    <div class="readout" id="readout">
      <div class="readout-title">No selection</div>
      <div class="row"><span class="key">Action</span><span class="value">Click a boundary</span></div>
    </div>
    <h2>Legend</h2>
    <div class="legend-item"><span class="swatch ring-a"></span><span>Light segment: annual ring interval.</span></div>
    <div class="legend-item"><span class="swatch ring-b"></span><span>Dark segment: alternating annual ring interval.</span></div>
    <div class="legend-item"><span class="swatch boundary"></span><span>Blue ridge: detected ring boundary / age marker.</span></div>
    <div class="stats" id="stats"></div>
  </aside>
</main>
<script>
const model = __MODEL_JSON__;
const canvas = document.getElementById("viewer");
const ctx = canvas.getContext("2d");
const readout = document.getElementById("readout");
const stats = document.getElementById("stats");
let hitRegions = [];
let selected = null;

function fmt(value, decimals = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(decimals);
}

function row(key, value) {
  return `<div class="row"><span class="key">${key}</span><span class="value">${value}</span></div>`;
}

function showSelection(item) {
  selected = item;
  if (!item) {
    readout.innerHTML = `<div class="readout-title">No selection</div>${row("Action", "Click a boundary")}`;
    draw();
    return;
  }
  if (item.type === "boundary") {
    const d = item.data;
    readout.innerHTML = `
      <div class="readout-title">Boundary ${d.boundary_no}</div>
      ${row("Estimated age marker", d.estimated_age_marker)}
      ${row("Position", fmt(d.position_mm) + " mm")}
      ${row("Previous ring width", fmt(d.previous_ring_width_mm) + " mm")}
      ${row("Next ring width", fmt(d.next_ring_width_mm) + " mm")}
    `;
  } else {
    const d = item.data;
    readout.innerHTML = `
      <div class="readout-title">Ring interval ${d.ring_no}</div>
      ${row("Width", fmt(d.width_mm) + " mm")}
      ${row("Start boundary", d.start_boundary_no)}
      ${row("End boundary", d.end_boundary_no)}
      ${row("Start position", fmt(d.start_mm) + " mm")}
      ${row("End position", fmt(d.end_mm) + " mm")}
    `;
  }
  draw();
}

function resizeCanvas() {
  const rect = canvas.parentElement.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(800, Math.floor(rect.width * dpr));
  canvas.height = Math.max(520, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}

function project(x, y, z, scale, ox, oy) {
  return {
    x: ox + x * scale + y * scale * 0.38,
    y: oy - z * scale * 2.4 - y * scale * 0.24
  };
}

function makePath(points) {
  const path = new Path2D();
  points.forEach((p, i) => {
    if (i === 0) path.moveTo(p.x, p.y);
    else path.lineTo(p.x, p.y);
  });
  path.closePath();
  return path;
}

function polygon(points, fill, stroke = "rgba(0,0,0,0.25)", lineWidth = 1) {
  const path = makePath(points);
  ctx.fillStyle = fill;
  ctx.strokeStyle = stroke;
  ctx.lineWidth = lineWidth;
  ctx.fill(path);
  ctx.stroke(path);
  return path;
}

function drawBox(x0, x1, y0, y1, z0, z1, fill, edge, scale, ox, oy, payload) {
  const p = (x, y, z) => project(x, y, z, scale, ox, oy);
  const top = [p(x0,y0,z1), p(x1,y0,z1), p(x1,y1,z1), p(x0,y1,z1)];
  const front = [p(x0,y1,z0), p(x1,y1,z0), p(x1,y1,z1), p(x0,y1,z1)];
  const side = [p(x1,y0,z0), p(x1,y1,z0), p(x1,y1,z1), p(x1,y0,z1)];
  polygon(front, shade(fill, -20), edge);
  polygon(side, shade(fill, -35), edge);
  const topPath = polygon(top, fill, edge, payload && selected && selected.type === payload.type && selected.data.id === payload.data.id ? 3 : 1);
  if (payload) hitRegions.push({ path: topPath, payload });
}

function shade(hex, amount) {
  const clean = hex.replace("#", "");
  const n = parseInt(clean, 16);
  let r = (n >> 16) + amount;
  let g = ((n >> 8) & 255) + amount;
  let b = (n & 255) + amount;
  r = Math.max(0, Math.min(255, r));
  g = Math.max(0, Math.min(255, g));
  b = Math.max(0, Math.min(255, b));
  return `rgb(${r},${g},${b})`;
}

function draw() {
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  hitRegions = [];
  const marginX = 70;
  const totalLength = Math.max(model.total_length_mm, 1);
  const visualDepth = model.strip_width_mm * 0.75;
  const scale = Math.min((rect.width - marginX * 2) / totalLength, (rect.height - 140) / Math.max(visualDepth * 2.4, 10));
  const ox = marginX;
  const oy = rect.height * 0.66;

  ctx.fillStyle = "#eef2f8";
  ctx.fillRect(0, 0, rect.width, rect.height);

  model.rings.forEach((ring, index) => {
    const fill = index % 2 === 0 ? "#c89a55" : "#80552c";
    const payload = { type: "ring", data: ring };
    drawBox(ring.start_mm, ring.end_mm, -model.strip_width_mm / 2, model.strip_width_mm / 2, 0, model.base_height_mm, fill, "rgba(56,38,20,0.46)", scale, ox, oy, payload);
  });

  model.boundaries.forEach((boundary) => {
    const x0 = Math.max(0, boundary.position_mm - model.ridge_width_mm / 2);
    const x1 = Math.min(model.total_length_mm, boundary.position_mm + model.ridge_width_mm / 2);
    const payload = { type: "boundary", data: boundary };
    drawBox(x0, x1, -model.strip_width_mm * 0.56, model.strip_width_mm * 0.56, model.base_height_mm, model.ridge_height_mm, "#2266e3", "rgba(8,35,100,0.70)", scale, ox, oy, payload);
  });

  const baseline0 = project(0, model.strip_width_mm * 0.72, 0, scale, ox, oy);
  const baseline1 = project(model.total_length_mm, model.strip_width_mm * 0.72, 0, scale, ox, oy);
  ctx.strokeStyle = "#172033";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(baseline0.x, baseline0.y + 22);
  ctx.lineTo(baseline1.x, baseline1.y + 22);
  ctx.stroke();
  ctx.fillStyle = "#172033";
  ctx.font = "700 13px Arial";
  ctx.fillText("0 mm", baseline0.x, baseline0.y + 42);
  ctx.fillText(fmt(model.total_length_mm) + " mm", baseline1.x - 78, baseline1.y + 42);

  stats.innerHTML = `
    Rings: ${model.ring_count}<br>
    Boundary points: ${model.boundary_count}<br>
    Total measured length: ${fmt(model.total_length_mm)} mm<br>
    Strip width: ${fmt(model.strip_width_mm)} mm
  `;
}

canvas.addEventListener("click", (event) => {
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  for (let i = hitRegions.length - 1; i >= 0; i -= 1) {
    if (ctx.isPointInPath(hitRegions[i].path, x, y)) {
      showSelection(hitRegions[i].payload);
      return;
    }
  }
  showSelection(null);
});

window.addEventListener("resize", resizeCanvas);
resizeCanvas();
</script>
</body>
</html>
"""
        viewer_path.write_text(html.replace("__MODEL_JSON__", model_json), encoding="utf-8")

    def _export_ring_3d_model(self, ring_points, pixel_to_mm, out_dir, stem="tree_ring_model"):
        out_dir = Path(out_dir)
        ensure_dir(out_dir)
        ring_points = np.asarray(ring_points, dtype=np.float32)
        df = measure(ring_points, pixel_to_mm)
        if len(df) == 0:
            raise RuntimeError("At least two boundary points are required for 3D export.")

        widths = df["width_mm"].to_numpy(dtype=float)
        total_length = float(np.sum(widths))
        mean_width = float(np.mean(widths)) if len(widths) else 1.0
        strip_width = max(4.0, min(14.0, float(self.band_width.get()) * float(pixel_to_mm)))
        base_height = 0.35
        ridge_height = 1.25
        ridge_width = min(max(mean_width * 0.10, 0.035), 0.22)
        half_width = strip_width / 2.0

        obj_path = out_dir / "08_3d_ring_model.obj"
        mtl_path = out_dir / "08_3d_ring_model.mtl"
        note_path = out_dir / "08_3d_ring_model_readme.txt"
        viewer_path = out_dir / "08_3d_ring_model_viewer.html"
        rings_csv_path = out_dir / "08_3d_ring_model_rings.csv"
        boundaries_csv_path = out_dir / "08_3d_ring_model_boundaries.csv"

        vertices = []
        lines = [
            "# TreeRing Studio 3D ring model",
            "# Units: millimeters",
            f"mtllib {mtl_path.name}",
            "",
        ]

        x0 = 0.0
        for index, width in enumerate(widths, start=1):
            x1 = x0 + float(width)
            material = "ring_light" if index % 2 else "ring_dark"
            self._write_box_obj(
                lines,
                vertices,
                x0,
                x1,
                -half_width,
                half_width,
                0.0,
                base_height,
                material,
                f"ring_interval_{index:03d}",
            )
            x0 = x1

        boundary_positions = [0.0]
        cumulative = 0.0
        for width in widths:
            cumulative += float(width)
            boundary_positions.append(cumulative)

        ring_records = []
        x0 = 0.0
        for index, width in enumerate(widths, start=1):
            x1 = x0 + float(width)
            ring_records.append({
                "id": f"ring-{index}",
                "ring_no": int(index),
                "start_boundary_no": int(index),
                "end_boundary_no": int(index + 1),
                "start_mm": float(x0),
                "end_mm": float(x1),
                "width_mm": float(width),
            })
            x0 = x1

        boundary_records = []
        for index, position in enumerate(boundary_positions, start=1):
            previous_width = float(widths[index - 2]) if index >= 2 else None
            next_width = float(widths[index - 1]) if index - 1 < len(widths) else None
            boundary_records.append({
                "id": f"boundary-{index}",
                "boundary_no": int(index),
                "estimated_age_marker": int(index - 1),
                "position_mm": float(position),
                "previous_ring_width_mm": previous_width,
                "next_ring_width_mm": next_width,
            })

        for index, position in enumerate(boundary_positions, start=1):
            x0 = max(0.0, position - ridge_width / 2.0)
            x1 = min(total_length, position + ridge_width / 2.0)
            if x1 <= x0:
                x1 = x0 + ridge_width
            self._write_box_obj(
                lines,
                vertices,
                x0,
                x1,
                -half_width * 1.08,
                half_width * 1.08,
                base_height,
                ridge_height,
                "boundary_ridge",
                f"boundary_{index:03d}",
            )

        with obj_path.open("w", encoding="utf-8") as handle:
            for vertex in vertices:
                handle.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            for line in lines:
                handle.write(line + "\n")

        mtl_path.write_text(
            "\n".join([
                "newmtl ring_light",
                "Kd 0.74 0.55 0.30",
                "Ka 0.20 0.15 0.08",
                "Ks 0.08 0.06 0.04",
                "",
                "newmtl ring_dark",
                "Kd 0.47 0.30 0.14",
                "Ka 0.16 0.10 0.05",
                "Ks 0.08 0.06 0.04",
                "",
                "newmtl boundary_ridge",
                "Kd 0.05 0.20 0.80",
                "Ka 0.02 0.05 0.20",
                "Ks 0.30 0.30 0.35",
                "",
            ]),
            encoding="utf-8",
        )

        pd.DataFrame(ring_records).to_csv(rings_csv_path, index=False)
        pd.DataFrame(boundary_records).to_csv(boundaries_csv_path, index=False)

        self._write_3d_model_viewer(
            viewer_path,
            {
                "ring_count": int(len(widths)),
                "boundary_count": int(len(boundary_positions)),
                "total_length_mm": float(total_length),
                "strip_width_mm": float(strip_width),
                "base_height_mm": float(base_height),
                "ridge_height_mm": float(ridge_height),
                "ridge_width_mm": float(ridge_width),
                "rings": ring_records,
                "boundaries": boundary_records,
            },
        )

        note_path.write_text(
            "\n".join([
                "TreeRing Studio 3D Model",
                "Format: OBJ + MTL",
                f"Interactive viewer: {viewer_path.name}",
                f"Ring data table: {rings_csv_path.name}",
                f"Boundary data table: {boundaries_csv_path.name}",
                "Units: millimeters",
                f"Ring intervals: {len(widths)}",
                f"Boundary points: {len(ring_points)}",
                f"Total modeled length: {total_length:.3f} mm",
                f"Strip width: {strip_width:.3f} mm",
                "Geometry: each annual interval is a scaled strip segment; each detected boundary is a raised ridge.",
                "Viewer: click a blue boundary ridge to see estimated age marker, position in mm, and neighboring ring widths.",
            ]),
            encoding="utf-8",
        )
        return {
            "obj": obj_path,
            "mtl": mtl_path,
            "viewer": viewer_path,
            "rings_csv": rings_csv_path,
            "boundaries_csv": boundaries_csv_path,
            "readme": note_path,
        }

    def export_current_3d_model(self):
        if not self.last_analysis_context:
            messagebox.showwarning(APP_TITLE, "Run an image analysis before exporting a 3D model.")
            return
        context = self.last_analysis_context
        ring_points = np.asarray(context.get("corrected_ring_points", context.get("ring_points", [])), dtype=np.float32)
        if len(ring_points) < 2:
            messagebox.showwarning(APP_TITLE, "At least two boundary points are required for 3D export.")
            return
        out_dir = Path(context.get("last_correction_dir", context.get("out_dir", self.output_path.get())))
        try:
            model_paths = self._export_ring_3d_model(
                ring_points,
                float(context["pixel_to_mm"]),
                out_dir,
                context.get("stem", "tree_ring_model"),
            )
            self.log_queue.put(f"3D model exported: {model_paths['viewer']}")
            self.status.set("3D model exported")
            messagebox.showinfo(
                APP_TITLE,
                "3D model exported:\n"
                f"Interactive viewer: {model_paths['viewer']}\n"
                f"OBJ model: {model_paths['obj']}",
            )
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Could not export 3D model:\n{exc}")

    def _match_boundary_points(self, original_points, corrected_points, measure_points, pixel_to_mm):
        original_points = np.asarray(original_points, dtype=np.float32)
        corrected_points = np.asarray(corrected_points, dtype=np.float32)
        if len(original_points) == 0 or len(corrected_points) == 0:
            return pd.DataFrame()

        measure_points = [np.array(point, dtype=np.float32) for point in measure_points]
        if len(measure_points) != 2:
            limit = min(len(original_points), len(corrected_points))
            shift_px = np.linalg.norm(corrected_points[:limit] - original_points[:limit], axis=1)
            return pd.DataFrame({
                "Boundary No": np.arange(1, limit + 1),
                "Original Boundary No": np.arange(1, limit + 1),
                "Corrected Boundary No": np.arange(1, limit + 1),
                "Original X (px)": original_points[:limit, 0],
                "Original Y (px)": original_points[:limit, 1],
                "Corrected X (px)": corrected_points[:limit, 0],
                "Corrected Y (px)": corrected_points[:limit, 1],
                "Axis Difference (px)": np.zeros(limit, dtype=float),
                "Shift (px)": shift_px,
                "Shift (mm)": shift_px * pixel_to_mm,
            })

        start, end = measure_points
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length < 1e-6:
            return pd.DataFrame()
        direction = direction / length
        original_axis = np.array([float(np.dot(point - start, direction)) for point in original_points], dtype=float)
        corrected_axis = np.array([float(np.dot(point - start, direction)) for point in corrected_points], dtype=float)

        spacing = np.diff(np.sort(original_axis))
        spacing = spacing[spacing > 1e-6]
        median_spacing = float(np.median(spacing)) if len(spacing) else 8.0
        tolerance_px = max(3.0, median_spacing * 0.45)

        candidates = []
        for original_index, original_pos in enumerate(original_axis):
            for corrected_index, corrected_pos in enumerate(corrected_axis):
                axis_diff = abs(corrected_pos - original_pos)
                if axis_diff <= tolerance_px:
                    candidates.append((axis_diff, original_index, corrected_index))
        candidates.sort(key=lambda item: item[0])

        used_original = set()
        used_corrected = set()
        rows = []
        for axis_diff, original_index, corrected_index in candidates:
            if original_index in used_original or corrected_index in used_corrected:
                continue
            used_original.add(original_index)
            used_corrected.add(corrected_index)
            shift_px = float(np.linalg.norm(corrected_points[corrected_index] - original_points[original_index]))
            rows.append({
                "Boundary No": int(original_index + 1),
                "Original Boundary No": int(original_index + 1),
                "Corrected Boundary No": int(corrected_index + 1),
                "Original Axis Position (mm)": float(original_axis[original_index] * pixel_to_mm),
                "Corrected Axis Position (mm)": float(corrected_axis[corrected_index] * pixel_to_mm),
                "Original X (px)": float(original_points[original_index, 0]),
                "Original Y (px)": float(original_points[original_index, 1]),
                "Corrected X (px)": float(corrected_points[corrected_index, 0]),
                "Corrected Y (px)": float(corrected_points[corrected_index, 1]),
                "Axis Difference (px)": float(axis_diff),
                "Shift (px)": shift_px,
                "Shift (mm)": shift_px * pixel_to_mm,
            })

        return pd.DataFrame(rows).sort_values("Original Boundary No").reset_index(drop=True) if rows else pd.DataFrame()

    def _paired_boundaries_from_editor_records(self, original_points, corrected_records, measure_points, pixel_to_mm):
        original_points = np.asarray(original_points, dtype=np.float32)
        if len(original_points) == 0 or not corrected_records:
            return pd.DataFrame()

        measure_points = [np.array(point, dtype=np.float32) for point in measure_points]
        if len(measure_points) == 2:
            start, end = measure_points
            direction = end - start
            length = float(np.linalg.norm(direction))
            direction = direction / length if length >= 1e-6 else np.array([1.0, 0.0], dtype=np.float32)
        else:
            start = np.array([0.0, 0.0], dtype=np.float32)
            direction = np.array([1.0, 0.0], dtype=np.float32)

        rows = []
        for corrected_index, record in enumerate(corrected_records, start=1):
            original_no = record.get("original_boundary_no")
            if original_no is None:
                continue
            original_no = int(original_no)
            if original_no < 1 or original_no > len(original_points):
                continue
            original_point = original_points[original_no - 1]
            corrected_point = np.array(record["point"], dtype=np.float32)
            original_axis = float(np.dot(original_point - start, direction))
            corrected_axis = float(np.dot(corrected_point - start, direction))
            axis_diff = abs(corrected_axis - original_axis)
            shift_px = float(np.linalg.norm(corrected_point - original_point))
            rows.append({
                "Boundary No": original_no,
                "Original Boundary No": original_no,
                "Corrected Boundary No": corrected_index,
                "Original Axis Position (mm)": original_axis * pixel_to_mm,
                "Corrected Axis Position (mm)": corrected_axis * pixel_to_mm,
                "Original X (px)": float(original_point[0]),
                "Original Y (px)": float(original_point[1]),
                "Corrected X (px)": float(corrected_point[0]),
                "Corrected Y (px)": float(corrected_point[1]),
                "Axis Difference (px)": axis_diff,
                "Shift (px)": shift_px,
                "Shift (mm)": shift_px * pixel_to_mm,
            })

        return pd.DataFrame(rows).sort_values("Original Boundary No").reset_index(drop=True) if rows else pd.DataFrame()

    def _build_matched_cumulative_distances(self, paired_boundaries):
        if paired_boundaries.empty:
            return pd.DataFrame()
        required = {"Original Axis Position (mm)", "Corrected Axis Position (mm)"}
        if not required.issubset(set(paired_boundaries.columns)):
            return pd.DataFrame()

        original_values = paired_boundaries["Original Axis Position (mm)"].to_numpy(dtype=float)
        corrected_values = paired_boundaries["Corrected Axis Position (mm)"].to_numpy(dtype=float)
        diff = corrected_values - original_values
        with np.errstate(divide="ignore", invalid="ignore"):
            ape = np.where(np.abs(original_values) > 1e-12, np.abs(diff) / np.abs(original_values) * 100, np.nan)
        return pd.DataFrame({
            "Original Boundary No": paired_boundaries["Original Boundary No"].to_numpy(dtype=int),
            "Corrected Boundary No": paired_boundaries["Corrected Boundary No"].to_numpy(dtype=int),
            "Original Cumulative Boundary Position (mm)": original_values,
            "Corrected Cumulative Boundary Position (mm)": corrected_values,
            "Difference Corrected - Original (mm)": diff,
            "Absolute Difference (mm)": np.abs(diff),
            "Absolute Percentage Difference (%)": ape,
        })

    def _build_matched_interval_widths(self, original_df, corrected_df, paired_boundaries):
        if paired_boundaries.empty or original_df.empty or corrected_df.empty:
            return pd.DataFrame()

        boundary_map = {
            int(row["Original Boundary No"]): int(row["Corrected Boundary No"])
            for _, row in paired_boundaries.iterrows()
        }
        corrected_by_ring = {int(row["ring_no"]): row for _, row in corrected_df.iterrows()}
        rows = []
        for _, original_row in original_df.iterrows():
            original_ring = int(original_row["ring_no"])
            left_boundary = original_ring
            right_boundary = original_ring + 1
            if left_boundary not in boundary_map or right_boundary not in boundary_map:
                continue
            corrected_left = boundary_map[left_boundary]
            corrected_right = boundary_map[right_boundary]
            if corrected_right - corrected_left != 1:
                continue
            corrected_ring = corrected_left
            if corrected_ring not in corrected_by_ring:
                continue

            corrected_row = corrected_by_ring[corrected_ring]
            original_width = float(original_row["width_mm"])
            corrected_width = float(corrected_row["width_mm"])
            diff = corrected_width - original_width
            ape = abs(diff) / abs(original_width) * 100 if abs(original_width) > 1e-12 else np.nan
            rows.append({
                "Ring No": original_ring,
                "Original Ring No": original_ring,
                "Corrected Ring No": corrected_ring,
                "Original Boundary Pair": f"{left_boundary}-{right_boundary}",
                "Corrected Boundary Pair": f"{corrected_left}-{corrected_right}",
                "Original Distance Between Rings (mm)": original_width,
                "Corrected Distance Between Rings (mm)": corrected_width,
                "Difference Corrected - Original (mm)": diff,
                "Absolute Difference (mm)": abs(diff),
                "Absolute Percentage Difference (%)": ape,
            })

        return pd.DataFrame(rows)

    def _create_correction_comparison(self, original_df, corrected_df, original_points, corrected_points, pixel_to_mm, measure_points, out_dir, corrected_records=None):
        ensure_dir(out_dir)
        out_dir = Path(out_dir)
        for stale_name in ("07_matched_original_vs_corrected_widths.png", "08_matched_original_vs_corrected_boundary_spacing.png"):
            stale_path = out_dir / stale_name
            if stale_path.exists():
                stale_path.unlink()
        width_n = min(len(original_df), len(corrected_df))

        original_age = int(len(original_df))
        corrected_age = int(len(corrected_df))
        age_percent_change = (corrected_age - original_age) / original_age * 100 if original_age else np.nan
        paired_widths = pd.DataFrame()
        paired_cumulative = pd.DataFrame()
        paired_boundaries = pd.DataFrame()

        width_metrics = self._metric_block(
            original_df["width_mm"].to_numpy(dtype=float) if len(original_df) else [],
            corrected_df["width_mm"].to_numpy(dtype=float) if len(corrected_df) else [],
        )
        cumulative_metrics = self._metric_block(
            original_df["cumulative_distance_mm"].to_numpy(dtype=float) if len(original_df) else [],
            corrected_df["cumulative_distance_mm"].to_numpy(dtype=float) if len(corrected_df) else [],
        )

        if width_n:
            original_widths = original_df["width_mm"].iloc[:width_n].to_numpy(dtype=float)
            corrected_widths = corrected_df["width_mm"].iloc[:width_n].to_numpy(dtype=float)
            width_diff = corrected_widths - original_widths
            with np.errstate(divide="ignore", invalid="ignore"):
                width_ape = np.where(np.abs(original_widths) > 1e-12, np.abs(width_diff) / np.abs(original_widths) * 100, np.nan)
            paired_widths = pd.DataFrame({
                "Ring No": np.arange(1, width_n + 1),
                "Original Distance Between Rings (mm)": original_widths,
                "Corrected Distance Between Rings (mm)": corrected_widths,
                "Difference Corrected - Original (mm)": width_diff,
                "Absolute Difference (mm)": np.abs(width_diff),
                "Absolute Percentage Difference (%)": width_ape,
            })

            original_cumulative = original_df["cumulative_distance_mm"].iloc[:width_n].to_numpy(dtype=float)
            corrected_cumulative = corrected_df["cumulative_distance_mm"].iloc[:width_n].to_numpy(dtype=float)
            cumulative_diff = corrected_cumulative - original_cumulative
            with np.errstate(divide="ignore", invalid="ignore"):
                cumulative_ape = np.where(np.abs(original_cumulative) > 1e-12, np.abs(cumulative_diff) / np.abs(original_cumulative) * 100, np.nan)
            paired_cumulative = pd.DataFrame({
                "Ring No": np.arange(1, width_n + 1),
                "Original Cumulative Distance (mm)": original_cumulative,
                "Corrected Cumulative Distance (mm)": corrected_cumulative,
                "Difference Corrected - Original (mm)": cumulative_diff,
                "Absolute Difference (mm)": np.abs(cumulative_diff),
                "Absolute Percentage Difference (%)": cumulative_ape,
            })

        original_points = np.asarray(original_points, dtype=np.float32)
        corrected_points = np.asarray(corrected_points, dtype=np.float32)
        if corrected_records:
            paired_boundaries = self._paired_boundaries_from_editor_records(original_points, corrected_records, measure_points, pixel_to_mm)
        else:
            paired_boundaries = self._match_boundary_points(original_points, corrected_points, measure_points, pixel_to_mm)
        point_n = len(paired_boundaries)
        boundary_metrics = {
            "Compared Boundary Points": int(point_n),
            "Unmatched Original Boundary Points": int(max(len(original_points) - point_n, 0)),
            "Unmatched Corrected Boundary Points": int(max(len(corrected_points) - point_n, 0)),
            "Mean Shift (mm)": np.nan,
            "Median Shift (mm)": np.nan,
            "Max Shift (mm)": np.nan,
            "RMSE Shift (mm)": np.nan,
        }
        if point_n:
            shift_mm = paired_boundaries["Shift (mm)"].to_numpy(dtype=float)
            boundary_metrics.update({
                "Mean Shift (mm)": float(np.mean(shift_mm)),
                "Median Shift (mm)": float(np.median(shift_mm)),
                "Max Shift (mm)": float(np.max(shift_mm)),
                "RMSE Shift (mm)": float(np.sqrt(np.mean(shift_mm ** 2))),
            })

        matched_widths = self._build_matched_interval_widths(original_df, corrected_df, paired_boundaries)
        paired_widths = matched_widths
        width_metrics = self._metric_block(
            matched_widths["Original Distance Between Rings (mm)"].to_numpy(dtype=float) if not matched_widths.empty else [],
            matched_widths["Corrected Distance Between Rings (mm)"].to_numpy(dtype=float) if not matched_widths.empty else [],
        )

        paired_cumulative = self._build_matched_cumulative_distances(paired_boundaries)
        cumulative_metrics = self._metric_block(
            paired_cumulative["Original Cumulative Boundary Position (mm)"].to_numpy(dtype=float) if not paired_cumulative.empty else [],
            paired_cumulative["Corrected Cumulative Boundary Position (mm)"].to_numpy(dtype=float) if not paired_cumulative.empty else [],
        )

        summary_rows = [
            {"Group": "Age", "Metric": "Original Estimated Age", "Value": original_age, "Unit": "years/rings"},
            {"Group": "Age", "Metric": "Corrected Estimated Age", "Value": corrected_age, "Unit": "years/rings"},
            {"Group": "Age", "Metric": "Age Difference", "Value": corrected_age - original_age, "Unit": "years/rings"},
            {"Group": "Age", "Metric": "Age Percent Change", "Value": age_percent_change, "Unit": "%"},
            {"Group": "Ring Width", "Metric": "Compared Matched Intervals", "Value": width_metrics["n"], "Unit": "rings"},
            {"Group": "Ring Width", "Metric": "R2", "Value": width_metrics["r2"], "Unit": "0-1"},
            {"Group": "Ring Width", "Metric": "MAE", "Value": width_metrics["mae"], "Unit": "mm"},
            {"Group": "Ring Width", "Metric": "RMSE", "Value": width_metrics["rmse"], "Unit": "mm"},
            {"Group": "Ring Width", "Metric": "MAPE", "Value": width_metrics["mape_percent"], "Unit": "%"},
            {"Group": "Cumulative Distance", "Metric": "Compared Matched Boundaries", "Value": cumulative_metrics["n"], "Unit": "points"},
            {"Group": "Cumulative Distance", "Metric": "R2", "Value": cumulative_metrics["r2"], "Unit": "0-1"},
            {"Group": "Cumulative Distance", "Metric": "MAE", "Value": cumulative_metrics["mae"], "Unit": "mm"},
            {"Group": "Cumulative Distance", "Metric": "RMSE", "Value": cumulative_metrics["rmse"], "Unit": "mm"},
            {"Group": "Cumulative Distance", "Metric": "MAPE", "Value": cumulative_metrics["mape_percent"], "Unit": "%"},
            {"Group": "Boundary Shift", "Metric": "Compared Boundary Points", "Value": boundary_metrics["Compared Boundary Points"], "Unit": "points"},
            {"Group": "Boundary Shift", "Metric": "Unmatched Original Boundary Points", "Value": boundary_metrics["Unmatched Original Boundary Points"], "Unit": "points"},
            {"Group": "Boundary Shift", "Metric": "Unmatched Corrected Boundary Points", "Value": boundary_metrics["Unmatched Corrected Boundary Points"], "Unit": "points"},
            {"Group": "Boundary Shift", "Metric": "Mean Shift", "Value": boundary_metrics["Mean Shift (mm)"], "Unit": "mm"},
            {"Group": "Boundary Shift", "Metric": "Median Shift", "Value": boundary_metrics["Median Shift (mm)"], "Unit": "mm"},
            {"Group": "Boundary Shift", "Metric": "Max Shift", "Value": boundary_metrics["Max Shift (mm)"], "Unit": "mm"},
            {"Group": "Boundary Shift", "Metric": "RMSE Shift", "Value": boundary_metrics["RMSE Shift (mm)"], "Unit": "mm"},
        ]
        summary_df = pd.DataFrame(summary_rows)
        excel_path = out_dir / "correction_comparison.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="summary", index=False)
            paired_widths.to_excel(writer, sheet_name="ring_width_statistics", index=False)
            paired_cumulative.to_excel(writer, sheet_name="cumulative_statistics", index=False)
            paired_boundaries.to_excel(writer, sheet_name="boundary_shift_statistics", index=False)
            original_df.to_excel(writer, sheet_name="original_widths", index=False)
            corrected_df.to_excel(writer, sheet_name="corrected_widths", index=False)
            self._format_excel_workbook(writer.book)

        def fmt(value, decimals=3):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return "-"
            return f"{value:.{decimals}f}"

        summary_text = (
            "Original vs Corrected Comparison\n"
            f"Original estimated age: {original_age}\n"
            f"Corrected estimated age: {corrected_age}\n"
            f"Age difference: {corrected_age - original_age}\n"
            f"Age percent change: {fmt(age_percent_change)}%\n\n"
            f"Matched ring width R2: {fmt(width_metrics['r2'], 4)}\n"
            f"Ring width MAE/RMSE/MAPE: {fmt(width_metrics['mae'])} mm / {fmt(width_metrics['rmse'])} mm / {fmt(width_metrics['mape_percent'])}%\n"
            f"Cumulative distance R2: {fmt(cumulative_metrics['r2'], 4)}\n"
            f"Cumulative distance MAE/RMSE/MAPE: {fmt(cumulative_metrics['mae'])} mm / {fmt(cumulative_metrics['rmse'])} mm / {fmt(cumulative_metrics['mape_percent'])}%\n"
            f"Boundary shift mean/median/max/RMSE: {fmt(boundary_metrics['Mean Shift (mm)'])} / {fmt(boundary_metrics['Median Shift (mm)'])} / {fmt(boundary_metrics['Max Shift (mm)'])} / {fmt(boundary_metrics['RMSE Shift (mm)'])} mm\n"
            "Full corrected boundary spacing chart: 06_original_vs_corrected_boundary_spacing.png\n"
            f"Output folder: {out_dir}"
        )
        (out_dir / "correction_comparison_summary.txt").write_text(summary_text, encoding="utf-8")
        self._create_correction_comparison_charts(
            paired_widths,
            paired_cumulative,
            paired_boundaries,
            width_metrics,
            cumulative_metrics,
            boundary_metrics,
            original_df,
            corrected_df,
            out_dir,
        )
        return summary_text

    def _create_correction_comparison_charts(self, paired_widths, paired_cumulative, paired_boundaries, width_metrics, cumulative_metrics, boundary_metrics, original_df, corrected_df, out_dir):
        if original_df.empty and corrected_df.empty:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for stale_name in (
            "02_difference_by_ring.png",
            "03_original_vs_corrected_cumulative_distance.png",
            "04_boundary_shift_by_point.png",
        ):
            stale_path = out_dir / stale_name
            if stale_path.exists():
                stale_path.unlink()

        if not corrected_df.empty:
            fig, ax = plt.subplots(figsize=(13, 5.5))
            corrected_full_x = corrected_df["ring_no"].to_numpy(dtype=int)
            corrected_full_y = corrected_df["width_mm"].to_numpy(dtype=float)
            ax.bar(corrected_full_x, corrected_full_y, width=0.62, color="#0f766e", alpha=0.88, label="Corrected full analysis")
            label_step = max(1, int(np.ceil(len(corrected_full_x) / 35)))
            for index, ring in enumerate(corrected_full_x):
                ax.text(ring, corrected_full_y[index], f"{corrected_full_y[index]:.2f}", ha="center", va="bottom", fontsize=6.5, rotation=90 if len(corrected_full_x) > 18 else 0)
            ax.set_title("Corrected Boundary Spacing Profile")
            ax.set_xlabel("Boundary interval / estimated annual ring")
            ax.set_ylabel("Distance between consecutive boundaries (mm)")
            ax.legend(framealpha=0.35, facecolor="white", edgecolor="#94a3b8")
            ax.grid(True, axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(out_dir / "06_original_vs_corrected_boundary_spacing.png", dpi=220)
            plt.close(fig)

        if not paired_widths.empty:
            ring_no = paired_widths["Ring No"].to_numpy(dtype=int)
            corrected_ring_no = paired_widths["Corrected Ring No"].to_numpy(dtype=int) if "Corrected Ring No" in paired_widths.columns else ring_no
            original = paired_widths["Original Distance Between Rings (mm)"].to_numpy(dtype=float)
            diff = paired_widths["Difference Corrected - Original (mm)"].to_numpy(dtype=float)
            label_step = max(1, int(np.ceil(max(len(paired_widths), len(corrected_df)) / 35)))

            fig, ax = plt.subplots(figsize=(12, 5.5))
            if not corrected_df.empty:
                corrected_full_x = corrected_df["ring_no"].to_numpy(dtype=int)
                corrected_full_y = corrected_df["width_mm"].to_numpy(dtype=float)
                ax.plot(corrected_full_x, corrected_full_y, marker="s", linewidth=2.1, color="#f97316", label="Corrected full profile")
            ax.plot(corrected_ring_no, original, marker="o", linewidth=2.1, color="#1d4ed8", label="Original matched at corrected position")
            for index, corrected_ring in enumerate(corrected_ring_no):
                original_ring = ring_no[index]
                if index == 0 or index == len(corrected_ring_no) - 1 or corrected_ring % label_step == 0 or original_ring != corrected_ring:
                    label = str(corrected_ring) if original_ring == corrected_ring else f"{original_ring}->{corrected_ring}"
                    ax.annotate(label, (corrected_ring, original[index]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7.5, color="#1d4ed8")
            ax.set_title("Original vs Corrected Ring Widths (Matched by Boundary Identity)")
            ax.set_xlabel("Corrected ring no / estimated age")
            ax.set_ylabel("Width (mm)")
            ax.legend(framealpha=0.35, facecolor="white", edgecolor="#94a3b8")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(out_dir / "01_original_vs_corrected_widths.png", dpi=220)
            plt.close(fig)
        else:
            for stale_name in ("01_original_vs_corrected_widths.png",):
                stale_path = out_dir / stale_name
                if stale_path.exists():
                    stale_path.unlink()

        original_age = int(len(original_df))
        corrected_age = int(len(corrected_df))
        age_difference = corrected_age - original_age
        age_percent = age_difference / original_age * 100 if original_age else np.nan
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        labels = ["Original age", "Corrected age"]
        values = [original_age, corrected_age]
        bars = ax.bar(labels, values, color=["#1d4ed8", "#f97316"], alpha=0.92)
        ax.set_title("Age Comparison After Manual Correction")
        ax.set_ylabel("Estimated age / ring count")
        ax.grid(True, axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value}", ha="center", va="bottom", fontweight="bold", fontsize=13)
        percent_text = "-" if np.isnan(age_percent) else f"{age_percent:+.2f}%"
        ax.text(
            0.5,
            0.92,
            f"Age difference: {age_difference:+d} rings\nPercent change: {percent_text}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f8fafc", "edgecolor": "#94a3b8", "alpha": 0.82},
        )
        fig.tight_layout()
        fig.savefig(out_dir / "05_correction_metrics_dashboard.png", dpi=220)
        plt.close(fig)

    def open_detection_editor(self):
        if not self.last_analysis_context:
            messagebox.showwarning(APP_TITLE, "Run an image analysis before editing detection points.")
            return

        context = self.last_analysis_context
        window = tk.Toplevel(self)
        window.title("Detection Point Editor")
        window.geometry("1180x760")
        window.configure(bg="#0f172a")
        window.transient(self)

        mode_var = tk.StringVar(value="add")
        count_var = tk.StringVar()

        header = ttk.Frame(window, padding=10)
        header.pack(fill="x")
        ttk.Label(header, text="Detection Point Editor", style="Title.TLabel", font=("Segoe UI", 16, "bold")).pack(side="left")
        ttk.Label(header, textvariable=count_var, style="Muted.TLabel").pack(side="right", padx=(12, 0))

        tools = ttk.Frame(window, padding=(10, 0, 10, 8))
        tools.pack(fill="x")
        ttk.Radiobutton(tools, text="Add point", value="add", variable=mode_var).pack(side="left", padx=(0, 10))
        ttk.Radiobutton(tools, text="Delete nearest", value="delete", variable=mode_var).pack(side="left", padx=(0, 18))
        ttk.Button(tools, text="-", command=lambda: editor.zoom_out()).pack(side="left", padx=(0, 6))
        ttk.Button(tools, text="+", command=lambda: editor.zoom_in()).pack(side="left", padx=(0, 6))
        ttk.Button(tools, text="Fit", command=lambda: editor.fit_to_view()).pack(side="left", padx=(0, 10))
        ttk.Button(tools, text="Up", command=lambda: editor.pan_by(0, 100)).pack(side="left", padx=(0, 6))
        ttk.Button(tools, text="Down", command=lambda: editor.pan_by(0, -100)).pack(side="left", padx=(0, 6))
        ttk.Button(tools, text="Left", command=lambda: editor.pan_by(100, 0)).pack(side="left", padx=(0, 6))
        ttk.Button(tools, text="Right", command=lambda: editor.pan_by(-100, 0)).pack(side="left", padx=(0, 18))
        ttk.Label(
            tools,
            text="Left click adds/deletes points. Added points snap to the measurement line.",
            style="Muted.TLabel",
        ).pack(side="left")

        def update_count(count):
            count_var.set(f"Boundary points: {count} | Estimated age: {max(count - 1, 0)}")

        editor = DetectionEditCanvas(
            window,
            context["image"].copy(),
            context["ring_points"],
            context["measure_points"],
            mode_var,
            update_count,
            bg="#111827",
        )
        editor.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        update_count(len(context["ring_points"]))

        actions = ttk.Frame(window, padding=(10, 0, 10, 10))
        actions.pack(fill="x")
        ttk.Button(actions, text="Save Corrected Analysis", style="Accent.TButton", command=lambda: self._save_corrected_detection(editor, window)).pack(side="right")
        ttk.Button(actions, text="Close", command=window.destroy).pack(side="right", padx=(0, 8))

    def _save_corrected_detection(self, editor, window):
        context = self.last_analysis_context
        corrected_records = editor.get_sorted_point_records()
        ring_points = np.array([record["point"] for record in corrected_records], dtype=np.float32)
        if len(ring_points) < 2:
            messagebox.showwarning(APP_TITLE, "At least two boundary points are required.")
            return
        try:
            base_out_dir = Path(context["out_dir"])
            correction_dir = base_out_dir / "ManualCorrection"
            ensure_dir(correction_dir)
            summary_text = self._write_detection_outputs(
                context["image"].copy(),
                context["image_name"],
                context["stem"] + "_manual_correction",
                correction_dir,
                ring_points,
                context["pixel_to_mm"],
                context["ruler_px"],
                context["real_mm"],
                context["measure_points"],
                method="ManualCorrection",
                debug_df=context["debug_df"],
                adaptive_c_value=context.get("adaptive_c_value"),
                otsu_info=context.get("otsu_info"),
                auto_parameter_info=context.get("auto_parameter_info"),
            )
            original_points = np.array(context.get("original_ring_points", context["ring_points"]), dtype=np.float32)
            original_df = measure(original_points, context["pixel_to_mm"])
            corrected_df = measure(ring_points, context["pixel_to_mm"])
            comparison_dir = correction_dir / "Original_vs_Corrected_Comparison"
            comparison_text = self._create_correction_comparison(
                original_df,
                corrected_df,
                original_points,
                ring_points,
                context["pixel_to_mm"],
                context["measure_points"],
                comparison_dir,
                corrected_records=corrected_records,
            )
            self.last_analysis_context["ring_points"] = ring_points
            self.last_analysis_context["corrected_ring_points"] = ring_points
            self.last_analysis_context["last_correction_dir"] = correction_dir
            self.last_analysis_context["last_comparison_dir"] = comparison_dir
            self.log_queue.put(f"Manual correction saved: {correction_dir}")
            self.log_queue.put(f"Original vs corrected comparison: {comparison_dir}")
            combined_summary = summary_text + "\n\n" + comparison_text
            self.after(0, lambda text=combined_summary: self._set_summary(text))
            self.status.set("Manual correction saved")
            window.destroy()
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Could not save manual correction:\n{exc}")

    def _process_current(self, real_mm):
        try:
            path = self.images[self.current_index]
            img = self.current_image.copy()
            stem = Path(path).stem
            out_dir = Path(self.output_path.get()) / stem
            ensure_dir(out_dir)

            cv2.imwrite(str(out_dir / "00_original.jpg"), img)
            ruler_px = float(np.linalg.norm(self.ruler_points[1] - self.ruler_points[0]))
            pixel_to_mm = real_mm / ruler_px

            scale_vis = img.copy()
            cv2.line(scale_vis, tuple(self.ruler_points[0].astype(int)), tuple(self.ruler_points[1].astype(int)), (0, 255, 0), 3)
            cv2.putText(scale_vis, f"{real_mm:g} mm", tuple(self.ruler_points[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imwrite(str(out_dir / "01_selected_ruler_scale.jpg"), scale_vis)

            if self.parameter_mode.get() == "Auto":
                self._update_auto_detection_parameters(log_change=False)
            effective_adaptive_c = int(self.adaptive_c.get())
            otsu_info = dict(self.otsu_info) if self.adaptive_c_mode.get() == "Auto (Otsu)" else {"adaptive_c_mode": "Manual"}
            auto_parameter_info = (
                dict(self.auto_parameter_info)
                if self.parameter_mode.get() == "Auto"
                else {"parameter_mode": "Manual"}
            )
            ring_points, debug_df, _peaks = detect_ring_points_skeleton(
                img,
                self.measure_points[0],
                self.measure_points[1],
                pixel_to_mm,
                out_dir,
                stem,
                band_width=int(self.band_width.get()),
                min_ring_mm=float(self.min_ring_mm.get()),
                adaptive_block=int(self.adaptive_block.get()),
                adaptive_c=effective_adaptive_c,
            )
            summary_text = self._write_detection_outputs(
                img,
                Path(path).name,
                stem,
                out_dir,
                ring_points,
                pixel_to_mm,
                ruler_px,
                real_mm,
                self.measure_points,
                method="TreeRing_Studio_Skeleton_ConnectedComponent",
                debug_df=debug_df,
                adaptive_c_value=effective_adaptive_c,
                otsu_info=otsu_info,
                auto_parameter_info=auto_parameter_info,
            )
            self.last_analysis_context = {
                "image": img.copy(),
                "image_name": Path(path).name,
                "stem": stem,
                "out_dir": out_dir,
                "ring_points": np.array(ring_points, dtype=np.float32),
                "original_ring_points": np.array(ring_points, dtype=np.float32),
                "pixel_to_mm": pixel_to_mm,
                "ruler_px": ruler_px,
                "real_mm": real_mm,
                "measure_points": [np.array(point, dtype=np.float32) for point in self.measure_points],
                "debug_df": debug_df.copy() if hasattr(debug_df, "copy") else debug_df,
                "adaptive_c_value": effective_adaptive_c,
                "otsu_info": otsu_info,
                "auto_parameter_info": auto_parameter_info,
            }
            self._write_all_summary(self._last_detection_summary)
            self.log_queue.put(f"Completed: {len(ring_points)} boundary points, {max(len(ring_points) - 1, 0)} ring widths.")
            self.log_queue.put(f"Output: {out_dir}")
            self.after(0, lambda text=summary_text: self._set_summary(text))
            self.after(0, lambda: self.status.set("Analysis completed"))
        except Exception as exc:
            self.log_queue.put(f"ERROR: {exc}")
            self.after(0, lambda: self.status.set("Analysis error"))

    def _write_detection_outputs(
        self,
        img,
        image_name,
        stem,
        out_dir,
        ring_points,
        pixel_to_mm,
        ruler_px,
        real_mm,
        measure_points,
        method,
        debug_df,
        adaptive_c_value=None,
        otsu_info=None,
        auto_parameter_info=None,
    ):
        ensure_dir(out_dir)
        ring_points = np.array(ring_points, dtype=np.float32)
        df = measure(ring_points, pixel_to_mm)
        otsu_info = otsu_info or {}
        auto_parameter_info = auto_parameter_info or {}
        if adaptive_c_value is None:
            adaptive_c_value = int(self.adaptive_c.get())
        estimated_age = int(len(df))
        total_distance = float(df["width_mm"].sum()) if len(df) else 0.0
        median_width = float(df["width_mm"].median()) if len(df) else None
        std_width = float(df["width_mm"].std()) if len(df) > 1 else 0.0
        mean_width = float(df["width_mm"].mean()) if len(df) else None
        cv_width = float(std_width / mean_width) if mean_width else None
        summary = {
            "image": image_name,
            "method": method,
            "pixel_to_mm": pixel_to_mm,
            "selected_ruler_px": ruler_px,
            "selected_ruler_mm": real_mm,
            "analysis_profile": self.analysis_profile.get(),
            "detection_display_mode": self.detection_display_mode.get(),
            "parameter_mode": auto_parameter_info.get("parameter_mode", self.parameter_mode.get()),
            "min_ring_mm_parameter": float(self.min_ring_mm.get()),
            "band_width_px": int(self.band_width.get()),
            "sensitivity": self.sensitivity.get(),
            "adaptive_block": int(self.adaptive_block.get()),
            "auto_band_width_px": auto_parameter_info.get("auto_band_width_px"),
            "auto_min_ring_mm": auto_parameter_info.get("auto_min_ring_mm"),
            "auto_adaptive_block": auto_parameter_info.get("auto_adaptive_block"),
            "auto_spacing_px_median": auto_parameter_info.get("auto_spacing_px_median"),
            "auto_spacing_mm_median": auto_parameter_info.get("auto_spacing_mm_median"),
            "auto_spacing_px_robust": auto_parameter_info.get("auto_spacing_px_robust"),
            "auto_spacing_mm_robust": auto_parameter_info.get("auto_spacing_mm_robust"),
            "auto_preliminary_peak_count": auto_parameter_info.get("auto_preliminary_peak_count"),
            "auto_profile_sample_count": auto_parameter_info.get("auto_profile_sample_count"),
            "adaptive_c": int(adaptive_c_value),
            "adaptive_c_mode": otsu_info.get("adaptive_c_mode", self.adaptive_c_mode.get()),
            "adaptive_c_estimated": otsu_info.get("adaptive_c_estimated"),
            "adaptive_c_raw": otsu_info.get("adaptive_c_raw"),
            "otsu_threshold": otsu_info.get("otsu_threshold"),
            "otsu_band_mean": otsu_info.get("otsu_band_mean"),
            "otsu_band_median": otsu_info.get("otsu_band_median"),
            "otsu_band_std": otsu_info.get("otsu_band_std"),
            "otsu_sample_count": otsu_info.get("otsu_sample_count"),
            "detected_boundaries": int(len(ring_points)),
            "measured_ring_width_count": int(len(df)),
            "estimated_age_years": estimated_age,
            "total_measured_distance_mm": total_distance,
            "mean_width_mm": mean_width,
            "median_width_mm": median_width,
            "std_width_mm": std_width,
            "variation_coefficient": cv_width,
            "min_width_mm": float(df["width_mm"].min()) if len(df) else None,
            "max_width_mm": float(df["width_mm"].max()) if len(df) else None,
        }
        save_all(df, summary, debug_df, out_dir)
        point_visual = Path(out_dir) / "05_measurement_visual_points.jpg"
        boundary_visual = Path(out_dir) / "05_measurement_visual_boundaries.jpg"
        main_visual = Path(out_dir) / "05_measurement_visual.jpg"
        draw_output(img, measure_points[0], measure_points[1], ring_points, point_visual)
        draw_boundary_output(
            img,
            measure_points[0],
            measure_points[1],
            ring_points,
            boundary_visual,
            band_width=int(self.band_width.get()),
        )
        if self.detection_display_mode.get() == "Boundaries":
            draw_boundary_output(
                img,
                measure_points[0],
                measure_points[1],
                ring_points,
                main_visual,
                band_width=int(self.band_width.get()),
            )
        else:
            draw_output(img, measure_points[0], measure_points[1], ring_points, main_visual)
        draw_detection_mode_panel(point_visual, boundary_visual, Path(out_dir) / "05_detection_display_panel.jpg")
        self._create_report_charts(df, summary, debug_df, out_dir, stem)
        try:
            model_paths = self._export_ring_3d_model(ring_points, pixel_to_mm, out_dir, stem)
            summary["3d_model_obj"] = str(model_paths["obj"])
            summary["3d_model_viewer"] = str(model_paths["viewer"])
        except Exception as exc:
            summary["3d_model_obj"] = f"3D export skipped: {exc}"
            summary["3d_model_viewer"] = f"3D viewer skipped: {exc}"
        summary_text = self._format_summary_text(summary)
        (Path(out_dir) / "analysis_summary.txt").write_text(summary_text, encoding="utf-8")
        self._last_detection_summary = summary
        return summary_text

    def _format_summary_text(self, summary):
        age = summary.get("estimated_age_years", 0)
        mean_width = summary.get("mean_width_mm")
        median_width = summary.get("median_width_mm")
        total = summary.get("total_measured_distance_mm", 0)
        min_width = summary.get("min_width_mm")
        max_width = summary.get("max_width_mm")
        cv_width = summary.get("variation_coefficient")
        boundaries = summary.get("detected_boundaries", 0)
        profile = summary.get("analysis_profile", "-")
        display_mode = summary.get("detection_display_mode", "Points")
        parameter_mode = summary.get("parameter_mode", "Manual")
        adaptive_c_mode = summary.get("adaptive_c_mode", "Manual")
        adaptive_c = summary.get("adaptive_c")
        otsu_threshold = summary.get("otsu_threshold")

        if not age:
            quality = "No reliable ring spacing was found; check the measurement line and scale."
        elif cv_width is not None and cv_width > 0.45:
            quality = "Ring widths are highly variable; visually review the points and detection settings."
        elif boundaries > age + 8:
            quality = "Boundary count looks high; increase minimum ring width or use the Conservative profile if needed."
        else:
            quality = "Detection looks balanced; the target is one boundary point per ring."

        def fmt(value):
            return "-" if value is None else f"{value:.3f}"

        if adaptive_c_mode == "Auto (Otsu)" and otsu_threshold is not None:
            adaptive_line = f"Adaptive C: Auto Otsu = {adaptive_c} (threshold {float(otsu_threshold):.1f})"
        else:
            adaptive_line = f"Adaptive C: Manual = {adaptive_c}"

        if parameter_mode == "Auto":
            parameter_line = (
                f"Parameters: Auto | band {summary.get('band_width_px')} px, "
                f"min ring {summary.get('min_ring_mm_parameter'):.3f} mm, "
                f"block {summary.get('adaptive_block')}"
            )
        else:
            parameter_line = (
                f"Parameters: Manual | band {summary.get('band_width_px')} px, "
                f"min ring {summary.get('min_ring_mm_parameter'):.3f} mm, "
                f"block {summary.get('adaptive_block')}"
            )

        return (
            f"Image: {summary.get('image')}\n"
            f"Analysis profile: {profile}\n"
            f"Detection display mode: {display_mode}\n"
            f"{parameter_line}\n"
            f"{adaptive_line}\n"
            f"Estimated age / ring count: {age}\n"
            f"Detected boundary points: {boundaries}\n"
            f"Total measured distance: {total:.3f} mm\n"
            f"Mean ring width: {fmt(mean_width)} mm\n"
            f"Median ring width: {fmt(median_width)} mm\n"
            f"Min / Max ring width: {fmt(min_width)} / {fmt(max_width)} mm\n"
            f"Boundary spacing chart: 06_boundary_spacing_profile.png\n"
            f"3D model: {summary.get('3d_model_obj', '08_3d_ring_model.obj')}\n"
            f"Interactive 3D viewer: {summary.get('3d_model_viewer', '08_3d_ring_model_viewer.html')}\n"
            f"Interpretation: {quality}"
        )

    def _set_summary(self, text):
        self.summary_box.configure(state="normal")
        self.summary_box.delete("1.0", "end")
        self.summary_box.insert("end", text)
        self.summary_box.configure(state="disabled")

    def _create_report_charts(self, df, summary, debug_df, out_dir, stem):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            out_dir = Path(out_dir)
            if len(df):
                label_step = max(1, int(np.ceil(len(df) / 35)))

                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(df["ring_no"], df["width_mm"], marker="o", linewidth=2.2, color="#0f766e")
                ax.fill_between(df["ring_no"], df["width_mm"], color="#99f6e4", alpha=0.35)
                for index, row in df.iterrows():
                    ring_no = int(row["ring_no"])
                    if index == 0 or index == len(df) - 1 or ring_no % label_step == 0:
                        ax.annotate(
                            str(ring_no),
                            (row["ring_no"], row["width_mm"]),
                            textcoords="offset points",
                            xytext=(0, 7),
                            ha="center",
                            fontsize=8,
                            color="#0f172a",
                        )
                ax.set_title("Ring Width Profile")
                ax.set_xlabel("Ring no / estimated age")
                ax.set_ylabel("Width (mm)")
                ax.grid(True, alpha=0.25)
                fig.tight_layout()
                fig.savefig(out_dir / "06_ring_width_profile.png", dpi=220)
                plt.close(fig)

                fig, ax = plt.subplots(figsize=(13, 5.5))
                colors = ["#2563eb" if value >= (summary.get("mean_width_mm") or 0) else "#14b8a6" for value in df["width_mm"]]
                bars = ax.bar(df["ring_no"], df["width_mm"], color=colors, alpha=0.88)
                mean_value = summary.get("mean_width_mm")
                if mean_value is not None:
                    ax.axhline(mean_value, color="#ef4444", linestyle="--", linewidth=1.8, label=f"Mean {mean_value:.3f} mm")
                    ax.legend(loc="upper right", framealpha=0.28, facecolor="white", edgecolor="#94a3b8")
                for index, row in df.iterrows():
                    label = f"{row['width_mm']:.2f}"
                    ax.text(
                        row["ring_no"],
                        row["width_mm"],
                        label,
                        ha="center",
                        va="bottom",
                        fontsize=6.5,
                        color="#0f172a",
                        rotation=90 if len(df) > 18 else 0,
                    )
                ax.set_title("Boundary Spacing Profile")
                ax.set_xlabel("Boundary interval / estimated annual ring")
                ax.set_ylabel("Distance between consecutive boundaries (mm)")
                ax.set_xticks(df["ring_no"])
                if len(df) > 55:
                    ax.set_xticklabels([str(int(x)) if int(x) % label_step == 0 else "" for x in df["ring_no"]])
                ax.grid(True, axis="y", alpha=0.25)
                ax.margins(x=0.01)
                fig.tight_layout()
                fig.savefig(out_dir / "06_boundary_spacing_profile.png", dpi=220)
                plt.close(fig)

                fig, axes = plt.subplots(2, 2, figsize=(14, 9))
                fig.suptitle("TreeRing Studio Analysis Dashboard", fontsize=18, fontweight="bold")
                axes[0, 0].bar(df["ring_no"], df["width_mm"], color="#14b8a6")
                for index, row in df.iterrows():
                    ring_no = int(row["ring_no"])
                    if index == 0 or index == len(df) - 1 or ring_no % label_step == 0:
                        axes[0, 0].text(
                            row["ring_no"],
                            row["width_mm"],
                            str(ring_no),
                            ha="center",
                            va="bottom",
                            fontsize=7,
                            color="#0f172a",
                            rotation=90 if len(df) > 25 else 0,
                        )
                axes[0, 0].set_title("Annual Ring Widths")
                axes[0, 0].set_xlabel("Ring no")
                axes[0, 0].set_ylabel("mm")
                axes[0, 0].grid(True, axis="y", alpha=0.25)

                axes[0, 1].hist(df["width_mm"], bins=min(18, max(5, len(df) // 2)), color="#3b82f6", alpha=0.85)
                axes[0, 1].axvline(summary.get("mean_width_mm") or 0, color="#ef4444", linewidth=2, label="Mean")
                axes[0, 1].set_title("Width Distribution")
                axes[0, 1].set_xlabel("mm")
                axes[0, 1].legend()

                axes[1, 0].plot(df["ring_no"], df["cumulative_distance_mm"], color="#8b5cf6", linewidth=2.2)
                axes[1, 0].scatter(df["ring_no"], df["cumulative_distance_mm"], color="#6d28d9", s=18)
                for index, row in df.iterrows():
                    ring_no = int(row["ring_no"])
                    if index == 0 or index == len(df) - 1 or ring_no % label_step == 0:
                        axes[1, 0].annotate(
                            str(ring_no),
                            (row["ring_no"], row["cumulative_distance_mm"]),
                            textcoords="offset points",
                            xytext=(0, 6),
                            ha="center",
                            fontsize=7,
                            color="#312e81",
                        )
                axes[1, 0].set_title("Cumulative Growth Distance")
                axes[1, 0].set_xlabel("Ring no")
                axes[1, 0].set_ylabel("mm")
                axes[1, 0].grid(True, alpha=0.25)

                stats_text = self._format_summary_text(summary)
                axes[1, 1].axis("off")
                axes[1, 1].text(
                    0.02,
                    0.96,
                    stats_text,
                    va="top",
                    ha="left",
                    fontsize=11,
                    linespacing=1.5,
                    bbox={"boxstyle": "round,pad=0.6", "facecolor": "#ecfeff", "edgecolor": "#0891b2"},
                )
                fig.tight_layout(rect=(0, 0, 1, 0.95))
                fig.savefig(out_dir / "07_analysis_dashboard.png", dpi=220)
                plt.close(fig)

            if len(debug_df):
                fig, ax = plt.subplots(figsize=(14, 5))
                ax.plot(debug_df["distance_mm"], debug_df["base_profile"], label="base", color="#64748b", linewidth=1.6)
                ax.plot(debug_df["distance_mm"], debug_df["dark_profile"], label="dark", color="#0f766e", linewidth=1.6)
                score = debug_df["score"].to_numpy(dtype=float)
                score_range = np.ptp(score)
                score_norm = (score - np.min(score)) / score_range if score_range > 1e-9 else score * 0
                ax.plot(debug_df["distance_mm"], score_norm, label="detection score", color="#ef4444", linewidth=1.8)
                ax.set_title("Profile Signal and Detection Score")
                ax.set_xlabel("Distance (mm)")
                ax.set_ylabel("Normalized signal")
                ax.legend()
                ax.grid(True, alpha=0.22)
                fig.tight_layout()
                fig.savefig(out_dir / "08_signal_quality_report.png", dpi=220)
                plt.close(fig)
        except Exception as exc:
            self.log_queue.put(f"Could not create chart report: {exc}")

    def _write_all_summary(self, summary):
        out_root = Path(self.output_path.get())
        all_path = out_root / "ALL_summary_report.csv"
        if all_path.exists():
            old = pd.read_csv(all_path)
            data = pd.concat([old, pd.DataFrame([summary])], ignore_index=True)
        else:
            data = pd.DataFrame([summary])
        data.to_csv(all_path, index=False, encoding="utf-8-sig")

    def open_output(self):
        ensure_dir(self.output_path.get())
        subprocess.Popen(["explorer", str(Path(self.output_path.get()))])

    def open_sr_output(self):
        out_dir = Path(self.output_path.get()) / "SuperResolution"
        ensure_dir(out_dir)
        subprocess.Popen(["explorer", str(out_dir)])

    def open_stats_output(self):
        out_dir = Path(self.output_path.get()) / "StatisticalAnalysis"
        ensure_dir(out_dir)
        subprocess.Popen(["explorer", str(out_dir)])

    def _drain_logs(self):
        while True:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.log.insert("end", line + "\n")
            self.log.see("end")
        self.after(120, self._drain_logs)


if __name__ == "__main__":
    TreeRingStudio().mainloop()
