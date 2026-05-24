#!/usr/bin/env python3
"""Split a 3D mesh into N equal radial wedges around a central axis.

Standalone Python equivalent of the FusionSplitter Fusion 360 add-in. Reads
an STL/OBJ/PLY/3MF file, splits the geometry into N equal pie-slice wedges
around a chosen axis, optionally drills matching cylindrical alignment-pin
holes on every joint, and writes one file per piece.

Dependencies: trimesh, manifold3d, shapely, numpy. Install with:
    pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import Polygon


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def align_axis_to_z(origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """4x4 transform mapping the line (origin, direction) onto +Z through 0."""
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    origin = np.asarray(origin, dtype=float)

    T = np.eye(4)
    T[:3, 3] = -origin

    z = np.array([0.0, 0.0, 1.0])
    if np.allclose(direction, z):
        R = np.eye(4)
    elif np.allclose(direction, -z):
        R = trimesh.transformations.rotation_matrix(math.pi, [1.0, 0.0, 0.0])
    else:
        axis = np.cross(direction, z)
        axis /= np.linalg.norm(axis)
        angle = math.acos(float(np.clip(np.dot(direction, z), -1.0, 1.0)))
        R = trimesh.transformations.rotation_matrix(angle, axis)

    return R @ T


def wedge_polygon(theta_a: float, theta_b: float, radius: float,
                  arc_segments: int = 24) -> Polygon:
    """Sector polygon from theta_a to theta_b at the given radius.

    Apex is at (0,0); the arc is approximated by straight segments. Because
    `radius` is chosen to exceed the body's radial extent, the arc never
    touches the body and the approximation is geometrically irrelevant — but
    using a sector (rather than a bare triangle) keeps the polygon valid for
    N = 2, where a triangle would collapse to a line.
    """
    angles = np.linspace(theta_a, theta_b, max(2, arc_segments + 1))
    pts = [(0.0, 0.0)]
    pts.extend((radius * math.cos(a), radius * math.sin(a)) for a in angles)
    return Polygon(pts)


def build_wedge_prism(theta_a: float, theta_b: float, radius: float,
                      z_lo: float, z_hi: float) -> trimesh.Trimesh:
    """Wedge cutter as a Z-aligned prism extending from z_lo to z_hi."""
    poly = wedge_polygon(theta_a, theta_b, radius)
    prism = trimesh.creation.extrude_polygon(poly, height=z_hi - z_lo)
    prism.apply_translation([0.0, 0.0, z_lo])
    return prism


def build_pin_cylinder(center: np.ndarray, axis: np.ndarray,
                       radius: float, length: float,
                       sections: int = 24) -> trimesh.Trimesh:
    """Cylinder of given radius and total length, centered at `center` with
    its long axis along `axis` (need not be unit-length)."""
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    cyl = trimesh.creation.cylinder(radius=radius, height=length,
                                    sections=sections)
    # default cylinder is along +Z centered at origin
    R = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], axis)
    cyl.apply_transform(R)
    cyl.apply_translation(center)
    return cyl


# ---------------------------------------------------------------------------
# Core split
# ---------------------------------------------------------------------------

def split_radial(mesh: trimesh.Trimesh,
                 n: int,
                 axis_origin: np.ndarray,
                 axis_direction: np.ndarray,
                 pin_diameter: float = 0.0,
                 pin_depth: float = 0.0,
                 pin_count: int = 0) -> list[trimesh.Trimesh]:
    """Split `mesh` into N radial wedge pieces around the given axis.

    All lengths are in the mesh's native units (typically millimetres for
    STL). Returns the list of piece meshes, in original world coordinates.
    """
    if n < 2:
        raise ValueError("n must be >= 2")

    T = align_axis_to_z(axis_origin, axis_direction)
    T_inv = np.linalg.inv(T)

    local = mesh.copy()
    local.apply_transform(T)

    bb_min, bb_max = local.bounds
    # Max radial distance in XY of any bounding-box corner
    xy_corners = np.array([[bb_min[0], bb_min[1]],
                           [bb_max[0], bb_min[1]],
                           [bb_min[0], bb_max[1]],
                           [bb_max[0], bb_max[1]]])
    max_r = float(np.max(np.linalg.norm(xy_corners, axis=1)))
    big_r = max_r * 2.5 + 1.0

    z_lo_raw, z_hi_raw = float(bb_min[2]), float(bb_max[2])
    margin = max(0.5, (z_hi_raw - z_lo_raw) * 0.05)
    z_lo, z_hi = z_lo_raw - margin, z_hi_raw + margin

    pieces: list[trimesh.Trimesh] = []
    for i in range(n):
        theta_a = 2.0 * math.pi * i / n
        theta_b = 2.0 * math.pi * (i + 1) / n
        cutter = build_wedge_prism(theta_a, theta_b, big_r, z_lo, z_hi)
        piece = local.intersection(cutter)
        if piece.is_empty or len(piece.vertices) == 0:
            print(f"  wedge {i + 1}: empty (skipped)", file=sys.stderr)
            continue
        pieces.append(piece)

    # Pin holes ---------------------------------------------------------------
    if pin_diameter > 0 and pin_depth > 0 and pin_count > 0:
        cylinders: list[trimesh.Trimesh] = []
        for j in range(n):
            theta = 2.0 * math.pi * (j + 1) / n  # boundary between j and j+1
            radial = np.array([math.cos(theta), math.sin(theta), 0.0])
            # cylinder axis is perpendicular to the cutting plane (which is
            # spanned by Z and `radial`), so it lies in XY perpendicular to
            # `radial`.
            cyl_axis = np.array([-math.sin(theta), math.cos(theta), 0.0])
            for k in range(pin_count):
                t = (k + 1) / (pin_count + 1)
                z = z_lo + (z_hi - z_lo) * t
                center = radial * (max_r * 0.5) + np.array([0.0, 0.0, z])
                cylinders.append(build_pin_cylinder(
                    center, cyl_axis,
                    radius=pin_diameter / 2.0,
                    length=2.0 * pin_depth))

        if cylinders:
            all_pins = trimesh.util.concatenate(cylinders)
            holed: list[trimesh.Trimesh] = []
            for piece in pieces:
                drilled = piece.difference(all_pins)
                if drilled.is_empty or len(drilled.vertices) == 0:
                    # fall back to the un-drilled piece if subtract failed
                    holed.append(piece)
                else:
                    holed.append(drilled)
            pieces = holed

    for piece in pieces:
        piece.apply_transform(T_inv)

    return pieces


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_vec3(text: str) -> np.ndarray:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"expected 'x,y,z', got {text!r}")
    try:
        return np.array([float(p) for p in parts], dtype=float)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc))


def resolve_axis(mesh: trimesh.Trimesh,
                 named: str | None,
                 explicit_origin: np.ndarray | None,
                 explicit_direction: np.ndarray | None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Pick an axis origin + direction. Explicit values win; otherwise the
    named axis ('X'/'Y'/'Z') passes through the mesh centroid."""
    if explicit_direction is not None:
        direction = explicit_direction
        origin = (explicit_origin if explicit_origin is not None
                  else np.asarray(mesh.centroid, dtype=float))
        return origin, direction

    axis_map = {
        "X": np.array([1.0, 0.0, 0.0]),
        "Y": np.array([0.0, 1.0, 0.0]),
        "Z": np.array([0.0, 0.0, 1.0]),
    }
    direction = axis_map[(named or "Z").upper()]
    origin = (explicit_origin if explicit_origin is not None
              else np.asarray(mesh.centroid, dtype=float))
    return origin, direction


def run_split(input_path: Path, pieces: int, axis: str,
              axis_origin: np.ndarray | None,
              axis_direction: np.ndarray | None,
              pins: bool, pin_diameter: float, pin_depth: float, pin_count: int,
              output_dir: Path | None, prefix: str | None,
              fmt: str | None,
              log=print) -> list[Path]:
    """Shared implementation used by both CLI and GUI paths.

    Returns the list of written file paths. Raises on failure.
    """
    if pieces < 2:
        raise ValueError("pieces must be >= 2")
    if not input_path.is_file():
        raise FileNotFoundError(f"input file not found: {input_path}")

    mesh = trimesh.load(input_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(
            f"input did not load as a single mesh: {type(mesh).__name__}")
    if not mesh.is_watertight:
        log("warning: input mesh is not watertight; boolean ops may fail "
            "or produce odd results")

    origin, direction = resolve_axis(mesh, axis, axis_origin, axis_direction)

    def _fmt(v):
        return "(" + ", ".join(f"{x:.3f}" for x in v) + ")"
    log(f"splitting {input_path.name} into {pieces} wedges "
        f"around axis origin={_fmt(origin)} dir={_fmt(direction)}")

    piece_meshes = split_radial(
        mesh,
        n=pieces,
        axis_origin=origin,
        axis_direction=direction,
        pin_diameter=(pin_diameter if pins else 0.0),
        pin_depth=(pin_depth if pins else 0.0),
        pin_count=(pin_count if pins else 0),
    )

    if not piece_meshes:
        raise RuntimeError("no pieces produced")

    out_dir = output_dir or input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    name_prefix = prefix or f"{input_path.stem}_wedge"
    ext = (fmt or input_path.suffix.lstrip(".") or "stl").lower()

    written: list[Path] = []
    for idx, piece in enumerate(piece_meshes, start=1):
        out_path = out_dir / f"{name_prefix}_{idx:02d}.{ext}"
        piece.export(out_path)
        log(f"  wrote {out_path}  ({len(piece.vertices)} verts, "
            f"{len(piece.faces)} faces)")
        written.append(out_path)
    return written


# ---------------------------------------------------------------------------
# GUI (tkinter, stdlib)
# ---------------------------------------------------------------------------

def run_gui() -> int:
    import tkinter as tk
    from tkinter import filedialog, messagebox

    # Use plain tk widgets with explicit colors. ttk widgets render
    # invisibly on macOS dark mode with some Python/Tk builds, so we avoid
    # them entirely here.
    BG = "#f4f4f4"
    FG = "#000000"
    ENTRY_BG = "#ffffff"
    BTN_BG = "#e0e0e0"

    root = tk.Tk()
    root.title("FusionSplitter — Radial Wedge Splitter")
    root.geometry("640x600")
    root.configure(bg=BG)

    def lbl(parent, text, **kw):
        return tk.Label(parent, text=text, bg=BG, fg=FG,
                        anchor="w", **kw)

    def ent(parent, var, **kw):
        return tk.Entry(parent, textvariable=var, bg=ENTRY_BG, fg=FG,
                        insertbackground=FG, highlightthickness=1,
                        highlightbackground="#888888", relief="flat", **kw)

    def btn(parent, text, command, **kw):
        return tk.Button(parent, text=text, command=command,
                         bg=BTN_BG, fg=FG, activebackground="#cccccc",
                         activeforeground=FG, highlightbackground=BG,
                         relief="raised", **kw)

    pad = {"padx": 8, "pady": 4}

    input_var = tk.StringVar()
    out_var = tk.StringVar()

    def pick_input():
        path = filedialog.askopenfilename(
            title="Select mesh to split",
            filetypes=[("Mesh files", "*.stl *.obj *.ply *.3mf"),
                       ("All files", "*.*")])
        if path:
            input_var.set(path)
            if not out_var.get():
                out_var.set(str(Path(path).parent))

    def pick_output():
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            out_var.set(path)

    row = 0
    lbl(root, "Input mesh:").grid(row=row, column=0, sticky="w", **pad)
    ent(root, input_var, width=50).grid(row=row, column=1, sticky="we", **pad)
    btn(root, "Browse…", pick_input).grid(row=row, column=2, **pad)

    row += 1
    lbl(root, "Output dir:").grid(row=row, column=0, sticky="w", **pad)
    ent(root, out_var, width=50).grid(row=row, column=1, sticky="we", **pad)
    btn(root, "Browse…", pick_output).grid(row=row, column=2, **pad)

    row += 1
    tk.Frame(root, bg="#cccccc", height=1).grid(
        row=row, column=0, columnspan=3, sticky="we", padx=8, pady=8)

    pieces_var = tk.IntVar(value=4)
    axis_var = tk.StringVar(value="Z")

    row += 1
    lbl(root, "Number of pieces (>=2):").grid(row=row, column=0, sticky="w", **pad)
    tk.Spinbox(root, from_=2, to=64, textvariable=pieces_var, width=8,
               bg=ENTRY_BG, fg=FG, buttonbackground=BTN_BG,
               highlightthickness=1, highlightbackground="#888888",
               relief="flat").grid(row=row, column=1, sticky="w", **pad)

    row += 1
    lbl(root, "Center axis:").grid(row=row, column=0, sticky="w", **pad)
    axis_frame = tk.Frame(root, bg=BG)
    axis_frame.grid(row=row, column=1, sticky="w", **pad)
    for opt in ("X", "Y", "Z"):
        tk.Radiobutton(axis_frame, text=opt, variable=axis_var, value=opt,
                       bg=BG, fg=FG, selectcolor=ENTRY_BG,
                       activebackground=BG, activeforeground=FG).pack(
            side="left", padx=4)
    lbl(root, "(through mesh centroid)").grid(
        row=row, column=2, sticky="w", **pad)

    row += 1
    tk.Frame(root, bg="#cccccc", height=1).grid(
        row=row, column=0, columnspan=3, sticky="we", padx=8, pady=8)

    pins_var = tk.BooleanVar(value=True)
    pin_dia_var = tk.DoubleVar(value=4.0)
    pin_depth_var = tk.DoubleVar(value=10.0)
    pin_count_var = tk.IntVar(value=2)

    row += 1
    tk.Checkbutton(root, text="Add alignment pin holes",
                   variable=pins_var, bg=BG, fg=FG,
                   selectcolor=ENTRY_BG, activebackground=BG,
                   activeforeground=FG).grid(
        row=row, column=0, columnspan=2, sticky="w", **pad)

    row += 1
    lbl(root, "Pin diameter (mm):").grid(row=row, column=0, sticky="w", **pad)
    ent(root, pin_dia_var, width=10).grid(row=row, column=1, sticky="w", **pad)

    row += 1
    lbl(root, "Pin depth each side (mm):").grid(
        row=row, column=0, sticky="w", **pad)
    ent(root, pin_depth_var, width=10).grid(row=row, column=1, sticky="w", **pad)

    row += 1
    lbl(root, "Pins per joint:").grid(row=row, column=0, sticky="w", **pad)
    tk.Spinbox(root, from_=1, to=10, textvariable=pin_count_var, width=8,
               bg=ENTRY_BG, fg=FG, buttonbackground=BTN_BG,
               highlightthickness=1, highlightbackground="#888888",
               relief="flat").grid(row=row, column=1, sticky="w", **pad)

    row += 1
    tk.Frame(root, bg="#cccccc", height=1).grid(
        row=row, column=0, columnspan=3, sticky="we", padx=8, pady=8)

    row += 1
    log_text = tk.Text(root, height=10, width=72, wrap="word",
                       bg=ENTRY_BG, fg=FG, insertbackground=FG,
                       relief="flat", highlightthickness=1,
                       highlightbackground="#888888")
    log_text.grid(row=row, column=0, columnspan=3, sticky="we", **pad)

    def log(msg: str):
        log_text.insert("end", msg + "\n")
        log_text.see("end")
        root.update_idletasks()

    def do_split():
        try:
            if not input_var.get():
                messagebox.showerror("Missing input", "Please pick an input mesh.")
                return
            input_path = Path(input_var.get())
            out_dir = Path(out_var.get()) if out_var.get() else None
            log_text.delete("1.0", "end")
            written = run_split(
                input_path=input_path,
                pieces=pieces_var.get(),
                axis=axis_var.get(),
                axis_origin=None,
                axis_direction=None,
                pins=pins_var.get(),
                pin_diameter=pin_dia_var.get(),
                pin_depth=pin_depth_var.get(),
                pin_count=pin_count_var.get(),
                output_dir=out_dir,
                prefix=None,
                fmt=None,
                log=log,
            )
            messagebox.showinfo("Done", f"Wrote {len(written)} piece(s) to:\n"
                                        f"{written[0].parent}")
        except Exception as exc:
            log(f"ERROR: {exc}")
            messagebox.showerror("Split failed", str(exc))

    row += 1
    btn(root, "Split", do_split, font=("Helvetica", 14, "bold"),
        padx=20, pady=8).grid(row=row, column=0, columnspan=3, pady=14)

    root.columnconfigure(1, weight=1)
    root.mainloop()
    return 0


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    # Launch GUI if no args were passed (e.g. running from PyCharm Run button).
    if argv is None and len(sys.argv) == 1:
        return run_gui()

    p = argparse.ArgumentParser(
        description="Split a mesh into N equal radial wedges around a central axis. "
                    "Run with no arguments to open the GUI.")
    p.add_argument("input", type=Path, help="Input mesh file (STL/OBJ/PLY/3MF).")
    p.add_argument("-n", "--pieces", type=int, required=True,
                   help="Number of wedge pieces (>= 2).")
    p.add_argument("--axis", choices=["X", "Y", "Z"], default="Z",
                   help="Named center axis, passing through the mesh centroid "
                        "(default: Z). Ignored if --axis-direction is given.")
    p.add_argument("--axis-origin", type=parse_vec3, default=None,
                   help="Override the axis origin point as 'x,y,z' "
                        "(default: mesh centroid).")
    p.add_argument("--axis-direction", type=parse_vec3, default=None,
                   help="Override the axis direction vector as 'x,y,z'.")
    p.add_argument("--pins", action="store_true",
                   help="Add cylindrical alignment-pin holes on every joint.")
    p.add_argument("--pin-diameter", type=float, default=4.0,
                   help="Pin hole diameter in mesh units (default: 4).")
    p.add_argument("--pin-depth", type=float, default=10.0,
                   help="Pin hole depth on each side of the joint, in mesh "
                        "units (default: 10).")
    p.add_argument("--pin-count", type=int, default=2,
                   help="Pins per joint (default: 2).")
    p.add_argument("-o", "--output-dir", type=Path, default=None,
                   help="Output directory (default: same dir as input).")
    p.add_argument("--prefix", default=None,
                   help="Output filename prefix (default: '<input-stem>_wedge').")
    p.add_argument("--format", default=None,
                   help="Output file extension, e.g. 'stl', '3mf' "
                        "(default: same as input).")
    args = p.parse_args(argv)

    try:
        run_split(
            input_path=args.input,
            pieces=args.pieces,
            axis=args.axis,
            axis_origin=args.axis_origin,
            axis_direction=args.axis_direction,
            pins=args.pins,
            pin_diameter=args.pin_diameter,
            pin_depth=args.pin_depth,
            pin_count=args.pin_count,
            output_dir=args.output_dir,
            prefix=args.prefix,
            fmt=args.format,
        )
    except (ValueError, FileNotFoundError, TypeError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
