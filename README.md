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

- `FusionSplitter.py` — entry point + command handlers + split logic.
- `FusionSplitter.manifest` — Fusion add-in metadata.
