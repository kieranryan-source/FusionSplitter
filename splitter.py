#!/usr/bin/env python3
"""Split a 3D mesh into N equal radial wedges around a central axis.

Standalone Python equivalent of the FusionSplitter Fusion 360 add-in. Reads
an STL/OBJ/PLY/3MF file, splits the geometry into N equal pie-slice wedges
around a chosen axis, optionally drills matching cylindrical alignment-pin
holes on every joint, and writes one file per piece.

Dependencies: trimesh, manifold3d, numpy. Install with:
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


def _check_dependencies() -> str | None:
    """Verify the optional trimesh deps needed for slicing+capping are
    importable from THIS Python interpreter. Returns None if all good, or a
    user-facing message (with the exact pip command) describing what's
    missing."""
    missing = []
    try:
        import scipy  # noqa: F401
    except ImportError:
        missing.append("scipy")
    try:
        import shapely  # noqa: F401
    except ImportError:
        missing.append("shapely")
    try:
        import networkx  # noqa: F401
    except ImportError:
        missing.append("networkx")
    has_tri = False
    try:
        import mapbox_earcut  # noqa: F401
        has_tri = True
    except ImportError:
        pass
    if not has_tri:
        try:
            import triangle  # noqa: F401
            has_tri = True
        except ImportError:
            pass
    if not has_tri:
        missing.append("mapbox-earcut")
    try:
        import manifold3d  # noqa: F401
    except ImportError:
        missing.append("manifold3d")

    if not missing:
        return None

    return (
        "Missing Python packages: " + ", ".join(missing) + "\n\n"
        "These need to be installed into the same Python that's running "
        "this script. Open PyCharm's Terminal (bottom of the window — make "
        "sure your venv is active; the prompt should start with '(venv)') "
        "and run:\n\n"
        f"    {sys.executable} -m pip install " + " ".join(missing) + "\n\n"
        "Then restart the script."
    )


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


def _inward_normal(theta: float, sense: int) -> np.ndarray:
    """Unit normal of the cutting plane at angle `theta`, pointing INTO the
    wedge. `sense` is +1 for the lower-angle boundary, -1 for the upper."""
    if sense > 0:
        return np.array([-math.sin(theta), math.cos(theta), 0.0])
    return np.array([math.sin(theta), -math.cos(theta), 0.0])


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
                 pin_count: int = 0,
                 log=print) -> list[trimesh.Trimesh]:
    """Split `mesh` into N radial wedge pieces around the given axis.

    Uses `Trimesh.slice_plane` (pure numpy, no manifold3d) so it works on
    meshes that aren't perfectly clean closed volumes. Pin holes are a
    best-effort boolean subtract — if that fails, the pieces are still
    returned without holes.
    """
    if n < 2:
        raise ValueError("n must be >= 2")

    T = align_axis_to_z(axis_origin, axis_direction)
    T_inv = np.linalg.inv(T)

    local = mesh.copy()
    local.apply_transform(T)

    bb_min, bb_max = local.bounds
    xy_corners = np.array([[bb_min[0], bb_min[1]],
                           [bb_max[0], bb_min[1]],
                           [bb_min[0], bb_max[1]],
                           [bb_max[0], bb_max[1]]])
    max_r = float(np.max(np.linalg.norm(xy_corners, axis=1)))
    z_lo, z_hi = float(bb_min[2]), float(bb_max[2])

    origin_zero = np.zeros(3)
    pieces: list[trimesh.Trimesh] = []
    for i in range(n):
        theta_a = 2.0 * math.pi * i / n
        theta_b = 2.0 * math.pi * (i + 1) / n

        piece = local.slice_plane(
            plane_origin=origin_zero,
            plane_normal=_inward_normal(theta_a, +1),
            cap=True)
        if n > 2 and piece is not None and len(piece.vertices) > 0:
            piece = piece.slice_plane(
                plane_origin=origin_zero,
                plane_normal=_inward_normal(theta_b, -1),
                cap=True)

        if piece is None or len(piece.vertices) == 0:
            log(f"  wedge {i + 1}: empty (skipped)")
            continue
        pieces.append(piece)

    # Pin holes -- best-effort. Skipped silently if the boolean engine
    # rejects the geometry (which is common for imperfect input meshes).
    if pin_diameter > 0 and pin_depth > 0 and pin_count > 0 and pieces:
        try:
            cylinders: list[trimesh.Trimesh] = []
            for j in range(n):
                theta = 2.0 * math.pi * (j + 1) / n
                radial = np.array([math.cos(theta), math.sin(theta), 0.0])
                cyl_axis = np.array([-math.sin(theta), math.cos(theta), 0.0])
                for k in range(pin_count):
                    t = (k + 1) / (pin_count + 1)
                    z = z_lo + (z_hi - z_lo) * t
                    center = radial * (max_r * 0.5) + np.array([0.0, 0.0, z])
                    cylinders.append(build_pin_cylinder(
                        center, cyl_axis,
                        radius=pin_diameter / 2.0,
                        length=2.0 * pin_depth))

            all_pins = trimesh.util.concatenate(cylinders)
            holed: list[trimesh.Trimesh] = []
            for piece in pieces:
                drilled = piece.difference(all_pins)
                if drilled is None or drilled.is_empty or len(drilled.vertices) == 0:
                    holed.append(piece)
                else:
                    holed.append(drilled)
            pieces = holed
        except Exception as exc:
            log(f"warning: pin holes skipped (boolean engine rejected the "
                f"geometry: {exc}). Split pieces saved without holes.")

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


def _ensure_volume(mesh: trimesh.Trimesh, log=print) -> trimesh.Trimesh:
    """Attempt to make `mesh` a closed, consistently-wound volume so the
    manifold3d boolean engine will accept it. Returns the (possibly repaired)
    mesh; raises a helpful error if repair can't produce a volume."""
    if mesh.is_volume:
        return mesh

    log(f"input mesh is not a closed volume "
        f"(watertight={mesh.is_watertight}, "
        f"winding_consistent={mesh.is_winding_consistent}); "
        f"attempting auto-repair…")

    m = mesh.copy()
    # Cheap structural cleanups
    m.merge_vertices()
    try:
        m.update_faces(m.nondegenerate_faces())
        m.update_faces(m.unique_faces())
    except Exception:
        pass
    m.remove_unreferenced_vertices()
    # Trimesh repair helpers (all in-place, all best-effort)
    for fn_name in ("fill_holes", "fix_winding", "fix_inversion", "fix_normals"):
        fn = getattr(trimesh.repair, fn_name, None)
        if fn is None:
            continue
        try:
            fn(m)
        except Exception:
            pass

    if m.is_volume:
        log("auto-repair succeeded.")
    else:
        log(f"warning: auto-repair could not fully fix the mesh "
            f"(watertight={m.is_watertight}, "
            f"winding_consistent={m.is_winding_consistent}). "
            f"Splitting will continue with plane-slice cuts; the output "
            f"pieces may inherit any open boundaries from the input.")
    return m


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
    mesh = _ensure_volume(mesh, log)

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
    """Run the splitter via a Tk simpledialog.Dialog form.

    Using simpledialog.Dialog (the same machinery as `askinteger` /
    `askopenfilename`) sidesteps the macOS dark-mode rendering issues that
    affect widgets sitting in a regular root Tk window. Widgets are left
    unstyled so they pick up the OS appearance.
    """
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog

    dep_msg = _check_dependencies()
    if dep_msg:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Missing dependencies", dep_msg, parent=root)
        root.destroy()
        return 1

    class SplitterDialog(simpledialog.Dialog):
        def body(self, master):
            self.input_var = tk.StringVar()
            self.pieces_var = tk.IntVar(value=4)
            self.axis_var = tk.StringVar(value="Z")
            self.pins_var = tk.BooleanVar(value=True)
            self.pin_dia_var = tk.DoubleVar(value=4.0)
            self.pin_depth_var = tk.DoubleVar(value=10.0)
            self.pin_count_var = tk.IntVar(value=2)

            pad = {"padx": 8, "pady": 5}
            row = 0

            tk.Label(master, text="Input mesh:").grid(
                row=row, column=0, sticky="w", **pad)
            tk.Entry(master, textvariable=self.input_var, width=42).grid(
                row=row, column=1, sticky="we", **pad)
            tk.Button(master, text="Browse…",
                      command=self._pick_input).grid(
                row=row, column=2, **pad)

            row += 1
            tk.Label(master, text="Number of pieces:").grid(
                row=row, column=0, sticky="w", **pad)
            pf = tk.Frame(master)
            pf.grid(row=row, column=1, columnspan=2, sticky="w", **pad)
            for v in (2, 3, 4):
                tk.Radiobutton(pf, text=str(v),
                               variable=self.pieces_var, value=v).pack(
                    side="left", padx=8)

            row += 1
            tk.Label(master, text="Center axis:").grid(
                row=row, column=0, sticky="w", **pad)
            af = tk.Frame(master)
            af.grid(row=row, column=1, columnspan=2, sticky="w", **pad)
            for v in ("X", "Y", "Z"):
                tk.Radiobutton(af, text=v,
                               variable=self.axis_var, value=v).pack(
                    side="left", padx=8)

            row += 1
            tk.Checkbutton(master, text="Add alignment pin holes",
                           variable=self.pins_var).grid(
                row=row, column=0, columnspan=3, sticky="w", **pad)

            row += 1
            tk.Label(master, text="Pin diameter (mm):").grid(
                row=row, column=0, sticky="w", **pad)
            tk.Entry(master, textvariable=self.pin_dia_var, width=10).grid(
                row=row, column=1, sticky="w", **pad)

            row += 1
            tk.Label(master, text="Pin depth each side (mm):").grid(
                row=row, column=0, sticky="w", **pad)
            tk.Entry(master, textvariable=self.pin_depth_var, width=10).grid(
                row=row, column=1, sticky="w", **pad)

            row += 1
            tk.Label(master, text="Pins per joint:").grid(
                row=row, column=0, sticky="w", **pad)
            tk.Entry(master, textvariable=self.pin_count_var, width=10).grid(
                row=row, column=1, sticky="w", **pad)

            master.columnconfigure(1, weight=1)
            return None  # focus default

        def _pick_input(self):
            path = filedialog.askopenfilename(
                parent=self,
                title="Select mesh to split",
                filetypes=[("Mesh files", "*.stl *.obj *.ply *.3mf"),
                           ("All files", "*.*")])
            if path:
                self.input_var.set(path)

        def buttonbox(self):
            # Replace default OK/Cancel with Split/Cancel.
            box = tk.Frame(self)
            tk.Button(box, text="Split", width=10, default=tk.ACTIVE,
                      command=self.ok).pack(side="left", padx=8, pady=8)
            tk.Button(box, text="Cancel", width=10,
                      command=self.cancel).pack(side="left", padx=8, pady=8)
            self.bind("<Return>", self.ok)
            self.bind("<Escape>", self.cancel)
            box.pack()

        def validate(self):
            if not self.input_var.get().strip():
                messagebox.showerror("Missing input",
                                     "Please pick an input mesh.",
                                     parent=self)
                return 0
            return 1

        def apply(self):
            self.result = {
                "input_path": Path(self.input_var.get()),
                "pieces": int(self.pieces_var.get()),
                "axis": self.axis_var.get(),
                "pins": bool(self.pins_var.get()),
                "pin_diameter": float(self.pin_dia_var.get()),
                "pin_depth": float(self.pin_depth_var.get()),
                "pin_count": int(self.pin_count_var.get()),
            }

    root = tk.Tk()
    root.withdraw()
    dialog = SplitterDialog(root, title="FusionSplitter — Radial Wedge Splitter")
    if not getattr(dialog, "result", None):
        root.destroy()
        return 0

    params = dialog.result
    try:
        written = run_split(
            input_path=params["input_path"],
            pieces=params["pieces"],
            axis=params["axis"],
            axis_origin=None,
            axis_direction=None,
            pins=params["pins"],
            pin_diameter=params["pin_diameter"],
            pin_depth=params["pin_depth"],
            pin_count=params["pin_count"],
            output_dir=None,
            prefix=None,
            fmt=None,
            log=print,
        )
        messagebox.showinfo(
            "Done",
            f"Wrote {len(written)} piece(s) to:\n{written[0].parent}",
            parent=root)
    except Exception as exc:
        msg = str(exc)
        if "triangulation engine" in msg.lower():
            msg = (
                "Polygon triangulation engine missing.\n\n"
                "Run this in PyCharm's Terminal (with your venv active):\n\n"
                f"    {sys.executable} -m pip install mapbox-earcut\n\n"
                "Then restart the script."
            )
        messagebox.showerror("Split failed", msg, parent=root)

    root.destroy()
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

    dep_msg = _check_dependencies()
    if dep_msg:
        print(dep_msg, file=sys.stderr)
        return 1

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
