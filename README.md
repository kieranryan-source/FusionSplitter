# FusionSplitter

A Fusion 360 add-in that splits a solid body into **N equal radial wedges**
around a chosen center axis, with optional **alignment pin holes** on each
mating face. Built for 3D-printing objects that are too big to print whole.

> Heads up: if you only need "too big for the print bed," the **Cut** tool
> in Bambu Studio / OrcaSlicer / PrusaSlicer already does this with
> auto-generated dovetail or peg connectors and may be a faster path than
> installing an add-in.

## What it does

- Select a solid body and a center axis (a construction axis or any linear
  edge — e.g. the Z-axis of the origin).
- Pick `N` (2–64). The body is split into `N` equal pie-slice wedges around
  that axis.
- Optionally adds matching cylindrical pin holes on every joint so you can
  dowel + glue the printed pieces back together.

## Install

1. Clone or download this repo.
2. Move the `FusionSplitter` folder to Fusion 360's add-ins directory:
   - **Windows:** `%APPDATA%\Autodesk\Autodesk Fusion 360\API\AddIns\`
   - **macOS:** `~/Library/Application Support/Autodesk/Autodesk Fusion 360/API/AddIns/`
3. Restart Fusion 360.
4. Open **Utilities → Add-Ins → Add-Ins tab**, find **FusionSplitter**, and
   click **Run**. Tick **Run on Startup** if you want it always loaded.

## Use

1. In the **Solid** workspace, open **Modify → Split Body into Wedges**.
2. Pick the **Body** to split.
3. Pick the **Center axis** (construction axis or a linear edge — e.g. an
   origin axis).
4. Set **Number of pieces** (must be ≥ 2).
5. Choose whether to add **alignment pin holes**, and set diameter, depth,
   and pins per joint.
6. Click **OK**. You'll get `N` new bodies named "Wedge 1" … "Wedge N", with
   the original body hidden (you can re-show it from the browser).

## Notes & limitations

- All measurements are in millimetres in the dialog; Fusion stores them as
  centimetres internally — that's normal.
- The original body is kept in the file so you can undo or compare. Toggle
  visibility with the eye icon in the browser tree.
- Pin tool bodies used for the subtract step are left in the file but
  hidden, so the Combine features that reference them stay valid. You can
  delete them after if desired.
- For complex geometry, the intersect of wedge × body can fail if the body
  has non-manifold or self-intersecting surfaces. Repair the body first if
  this happens.
- This add-in does *not* currently handle bodies whose footprint doesn't
  surround the axis (e.g. an off-center solid). The axis should pass through
  or near the body for the wedges to produce useful pieces.

## Files

- `FusionSplitter.py` — Fusion add-in entry point + command handlers + split logic.
- `FusionSplitter.manifest` — Fusion add-in metadata.
- `splitter.py` — standalone Python CLI (see below).
- `requirements.txt` — deps for the CLI.

---

# `splitter.py` — standalone CLI

Same functionality as the add-in, but runs outside Fusion: reads a mesh file
(STL/OBJ/PLY/3MF), splits it into N radial wedges, optionally drills pin
holes, and writes one file per piece. Useful if you want to split a model
that's already exported as STL, or to script the process.

## Install

```sh
pip install -r requirements.txt
```

The splits themselves use `trimesh.slice_plane` (pure-numpy plane slicing),
which is permissive about input quality — non-watertight meshes are
auto-repaired where possible. Pin holes use the `manifold3d` boolean
backend; if it rejects the geometry, the pieces are still written without
holes and a warning is printed.

## Use

**GUI mode** — run with no arguments (e.g. PyCharm's Run button, or just
`python splitter.py`) and a Tkinter dialog opens with a file picker, axis
selector, piece-count spinner, and pin-hole options.

**CLI mode** — pass args:

```sh
# 6 wedges around the mesh's Z axis through centroid, with 2 pin holes per joint
python splitter.py model.stl -n 6 --pins

# Custom axis (origin + direction) and pin parameters
python splitter.py model.stl -n 8 \
    --axis-origin 0,0,0 --axis-direction 0,0,1 \
    --pins --pin-diameter 5 --pin-depth 12 --pin-count 3 \
    -o ./pieces

# Output a different format
python splitter.py model.stl -n 4 --format 3mf -o ./pieces
```

Run `python splitter.py --help` for all options.

## Notes

- All lengths are in the mesh's native units. STLs from Fusion are
  millimetres by default, which matches the `--pin-diameter` / `--pin-depth`
  defaults (4 mm / 10 mm).
- The axis defaults to **Z through the mesh centroid**. Pass `--axis X` /
  `--axis Y` for the other principal axes, or `--axis-direction x,y,z` and
  `--axis-origin x,y,z` for any line in space.
- If a piece comes out empty (axis grazes the mesh, wedge angle misses
  geometry, etc.) you'll get a warning and the piece is skipped.
