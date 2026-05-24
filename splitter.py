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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Split a mesh into N equal radial wedges around a central axis.")
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

    if args.pieces < 2:
        p.error("--pieces must be >= 2")

    if not args.input.is_file():
        p.error(f"input file not found: {args.input}")

    mesh = trimesh.load(args.input, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        p.error(f"input did not load as a single mesh: {type(mesh).__name__}")
    if not mesh.is_watertight:
        print("warning: input mesh is not watertight; boolean ops may fail "
              "or produce odd results", file=sys.stderr)

    origin, direction = resolve_axis(
        mesh, args.axis, args.axis_origin, args.axis_direction)

    def _fmt(v):
        return "(" + ", ".join(f"{x:.3f}" for x in v) + ")"
    print(f"splitting {args.input.name} into {args.pieces} wedges "
          f"around axis origin={_fmt(origin)} dir={_fmt(direction)}")

    pieces = split_radial(
        mesh,
        n=args.pieces,
        axis_origin=origin,
        axis_direction=direction,
        pin_diameter=(args.pin_diameter if args.pins else 0.0),
        pin_depth=(args.pin_depth if args.pins else 0.0),
        pin_count=(args.pin_count if args.pins else 0),
    )

    if not pieces:
        print("error: no pieces produced", file=sys.stderr)
        return 1

    out_dir = args.output_dir or args.input.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or f"{args.input.stem}_wedge"
    ext = (args.format or args.input.suffix.lstrip(".") or "stl").lower()

    for idx, piece in enumerate(pieces, start=1):
        out_path = out_dir / f"{prefix}_{idx:02d}.{ext}"
        piece.export(out_path)
        print(f"  wrote {out_path}  ({len(piece.vertices)} verts, "
              f"{len(piece.faces)} faces)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
