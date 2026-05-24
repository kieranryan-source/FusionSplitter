"""FusionSplitter — split a solid body into N equal radial wedges.

Pick a body, pick a center axis, set the number of pieces. Optionally adds
matching cylindrical alignment-pin holes on each pair of mating faces so the
printed pieces can be aligned with dowels and glued together.
"""

import math
import traceback

import adsk.core
import adsk.fusion

CMD_ID = "FusionSplitterCmd"
CMD_NAME = "Split Body into Wedges"
CMD_DESC = (
    "Split a solid body into N equal radial wedges around an axis, "
    "optionally with cylindrical alignment-pin holes on each joint."
)
WORKSPACE_ID = "FusionSolidEnvironment"
PANEL_ID = "SolidModifyPanel"

_handlers = []
_app = None
_ui = None


# ---------------------------------------------------------------------------
# Add-in lifecycle
# ---------------------------------------------------------------------------

def run(context):
    global _app, _ui
    try:
        _app = adsk.core.Application.get()
        _ui = _app.userInterface

        cmd_def = _ui.commandDefinitions.itemById(CMD_ID)
        if cmd_def:
            cmd_def.deleteMe()
        cmd_def = _ui.commandDefinitions.addButtonDefinition(CMD_ID, CMD_NAME, CMD_DESC)

        on_created = _CommandCreatedHandler()
        cmd_def.commandCreated.add(on_created)
        _handlers.append(on_created)

        ws = _ui.workspaces.itemById(WORKSPACE_ID)
        panel = ws.toolbarPanels.itemById(PANEL_ID)
        existing = panel.controls.itemById(CMD_ID)
        if existing:
            existing.deleteMe()
        panel.controls.addCommand(cmd_def)
    except Exception:
        if _ui:
            _ui.messageBox("FusionSplitter run() failed:\n{}".format(traceback.format_exc()))


def stop(context):
    try:
        if not _ui:
            return
        ws = _ui.workspaces.itemById(WORKSPACE_ID)
        if ws:
            panel = ws.toolbarPanels.itemById(PANEL_ID)
            if panel:
                ctrl = panel.controls.itemById(CMD_ID)
                if ctrl:
                    ctrl.deleteMe()
        cmd_def = _ui.commandDefinitions.itemById(CMD_ID)
        if cmd_def:
            cmd_def.deleteMe()
    except Exception:
        if _ui:
            _ui.messageBox("FusionSplitter stop() failed:\n{}".format(traceback.format_exc()))


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

class _CommandCreatedHandler(adsk.core.CommandCreatedEventHandler):
    def __init__(self):
        super().__init__()

    def notify(self, args):
        try:
            cmd = adsk.core.Command.cast(args.command)
            ip = cmd.commandInputs

            sel_body = ip.addSelectionInput("body", "Body", "Solid body to split")
            sel_body.addSelectionFilter("SolidBodies")
            sel_body.setSelectionLimits(1, 1)

            sel_axis = ip.addSelectionInput("axis", "Center axis", "Construction axis or linear edge")
            sel_axis.addSelectionFilter("ConstructionLines")
            sel_axis.addSelectionFilter("LinearEdges")
            sel_axis.setSelectionLimits(1, 1)

            ip.addIntegerSpinnerCommandInput("n", "Number of pieces", 2, 64, 1, 4)

            ip.addBoolValueInput("add_pins", "Add alignment pin holes", True, "", True)
            ip.addValueInput("pin_dia", "Pin hole diameter", "mm",
                             adsk.core.ValueInput.createByReal(0.4))
            ip.addValueInput("pin_depth", "Pin hole depth (each side)", "mm",
                             adsk.core.ValueInput.createByReal(1.0))
            ip.addIntegerSpinnerCommandInput("pin_count", "Pins per joint", 1, 10, 1, 2)

            ip.addBoolValueInput("hide_original", "Hide original body when done", True, "", True)

            on_exec = _CommandExecuteHandler()
            cmd.execute.add(on_exec)
            _handlers.append(on_exec)
        except Exception:
            if _ui:
                _ui.messageBox("CommandCreated failed:\n{}".format(traceback.format_exc()))


class _CommandExecuteHandler(adsk.core.CommandEventHandler):
    def __init__(self):
        super().__init__()

    def notify(self, args):
        try:
            cmd = args.firingEvent.sender
            ip = cmd.commandInputs

            body = ip.itemById("body").selection(0).entity
            axis_ent = ip.itemById("axis").selection(0).entity
            n = ip.itemById("n").value
            add_pins = ip.itemById("add_pins").value
            pin_dia = ip.itemById("pin_dia").value
            pin_depth = ip.itemById("pin_depth").value
            pin_count = ip.itemById("pin_count").value
            hide_original = ip.itemById("hide_original").value

            split_radial(body, axis_ent, n,
                         add_pins, pin_dia, pin_depth, pin_count,
                         hide_original)
        except Exception:
            if _ui:
                _ui.messageBox("Execute failed:\n{}".format(traceback.format_exc()))


# ---------------------------------------------------------------------------
# Vector helpers (work in cm — Fusion's internal unit)
# ---------------------------------------------------------------------------

def _vec(x, y, z):
    return adsk.core.Vector3D.create(x, y, z)


def _pt(x, y, z):
    return adsk.core.Point3D.create(x, y, z)


def _add(p, v, t=1.0):
    return _pt(p.x + v.x * t, p.y + v.y * t, p.z + v.z * t)


def _dot(a, b):
    return a.x * b.x + a.y * b.y + a.z * b.z


def _cross(a, b):
    return _vec(a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x)


def _unit(v):
    n = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    if n == 0:
        raise RuntimeError("Zero-length vector")
    return _vec(v.x / n, v.y / n, v.z / n)


def _sub_pts(p, q):
    return _vec(p.x - q.x, p.y - q.y, p.z - q.z)


# ---------------------------------------------------------------------------
# Axis + bounding-box analysis
# ---------------------------------------------------------------------------

def axis_from_entity(ent):
    """Return (origin Point3D, unit-direction Vector3D) for a construction axis
    or a linear B-rep edge."""
    if isinstance(ent, adsk.fusion.ConstructionAxis):
        line = ent.geometry  # InfiniteLine3D
        return line.origin, _unit(line.direction)
    if isinstance(ent, adsk.fusion.BRepEdge):
        g = ent.geometry
        if isinstance(g, adsk.core.Line3D):
            return g.startPoint, _unit(_sub_pts(g.endPoint, g.startPoint))
    raise RuntimeError("Selected entity is not a usable axis")


def body_axis_extent(body, origin, axis_dir):
    """Project the body's bounding-box corners onto the axis to find
    axial range and the maximum radial distance from the axis."""
    bb = body.boundingBox
    mn, mx = bb.minPoint, bb.maxPoint
    corners = [_pt(x, y, z)
               for x in (mn.x, mx.x)
               for y in (mn.y, mx.y)
               for z in (mn.z, mx.z)]
    axials, radials = [], []
    for c in corners:
        rel = _sub_pts(c, origin)
        a = _dot(rel, axis_dir)
        perp = _vec(rel.x - axis_dir.x * a,
                    rel.y - axis_dir.y * a,
                    rel.z - axis_dir.z * a)
        axials.append(a)
        radials.append(math.sqrt(_dot(perp, perp)))
    return min(axials), max(axials), max(radials)


# ---------------------------------------------------------------------------
# Main split
# ---------------------------------------------------------------------------

def split_radial(body, axis_ent, n,
                 add_pins, pin_dia, pin_depth, pin_count,
                 hide_original):
    app = adsk.core.Application.get()
    design = adsk.fusion.Design.cast(app.activeProduct)
    if not design:
        raise RuntimeError("No active Fusion design")
    root = design.rootComponent

    origin, axis_dir = axis_from_entity(axis_ent)
    axial_min, axial_max, max_r = body_axis_extent(body, origin, axis_dir)
    margin = max(0.5, (axial_max - axial_min) * 0.05)
    a_lo = axial_min - margin
    a_hi = axial_max + margin
    big_r = max_r * 2.5 + 1.0  # cm — generous; wedges only need to enclose the body

    plane_origin = _add(origin, axis_dir, a_lo)
    base_plane_geom = adsk.core.Plane.create(plane_origin, axis_dir)

    cp_input = root.constructionPlanes.createInput()
    cp_input.setByPlane(base_plane_geom)
    construction_plane = root.constructionPlanes.add(cp_input)

    sketches = root.sketches
    extrudes = root.features.extrudeFeatures
    combines = root.features.combineFeatures

    # The sketch defines its own local (x,y) frame in world space. Read it from
    # an empty probe sketch so we can align wedge angles consistently.
    probe = sketches.add(construction_plane)
    sx = probe.xDirection
    sy = probe.yDirection
    probe.deleteMe()

    wedge_pieces = []
    for i in range(n):
        s = sketches.add(construction_plane)
        a1 = 2 * math.pi * i / n
        a2 = 2 * math.pi * (i + 1) / n
        p0 = _pt(0, 0, 0)
        p1 = _pt(big_r * math.cos(a1), big_r * math.sin(a1), 0)
        p2 = _pt(big_r * math.cos(a2), big_r * math.sin(a2), 0)
        lines = s.sketchCurves.sketchLines
        lines.addByTwoPoints(p0, p1)
        lines.addByTwoPoints(p1, p2)
        lines.addByTwoPoints(p2, p0)

        prof = s.profiles.item(0)
        ext_in = extrudes.createInput(
            prof, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
        ext_in.setDistanceExtent(
            False, adsk.core.ValueInput.createByReal(a_hi - a_lo))
        ext = extrudes.add(ext_in)
        wedge_body = ext.bodies.item(0)

        tools = adsk.core.ObjectCollection.create()
        tools.add(body)
        ci = combines.createInput(wedge_body, tools)
        ci.operation = adsk.fusion.FeatureOperations.IntersectFeatureOperation
        ci.isKeepToolBodies = True
        ci.isNewComponent = False
        combines.add(ci)
        wedge_pieces.append(wedge_body)

    # ---- Pin holes -------------------------------------------------------
    if add_pins and pin_count > 0 and pin_dia > 0 and pin_depth > 0:
        tbm = adsk.fusion.TemporaryBRepManager.get()
        base_feats = root.features.baseFeatures
        bf = base_feats.add()
        bf.startEdit()
        pin_bodies = []
        try:
            for j in range(n):
                # boundary between wedge j and wedge (j+1) % n
                theta = 2 * math.pi * (j + 1) / n
                # Radial direction at this angle, in world space
                rad = _unit(_vec(
                    sx.x * math.cos(theta) + sy.x * math.sin(theta),
                    sx.y * math.cos(theta) + sy.y * math.sin(theta),
                    sx.z * math.cos(theta) + sy.z * math.sin(theta),
                ))
                # Cylinder axis is perpendicular to the cutting plane, which is
                # spanned by (axis_dir, rad). So normal = axis_dir × rad.
                cyl_axis = _unit(_cross(axis_dir, rad))

                for k in range(pin_count):
                    t = (k + 1) / (pin_count + 1)
                    axial_pos = a_lo + (a_hi - a_lo) * t
                    radial_pos = max_r * 0.5  # mid-radius
                    center = _add(origin, axis_dir, axial_pos)
                    center = _add(center, rad, radial_pos)
                    p_one = _add(center, cyl_axis, -pin_depth)
                    p_two = _add(center, cyl_axis, pin_depth)
                    cyl = tbm.createCylinderOrCone(
                        p_one, pin_dia / 2.0, p_two, pin_dia / 2.0)
                    pin_body = root.bRepBodies.add(cyl, bf)
                    pin_bodies.append(pin_body)
        finally:
            bf.finishEdit()

        for piece in wedge_pieces:
            tools = adsk.core.ObjectCollection.create()
            for pb in pin_bodies:
                tools.add(pb)
            ci = combines.createInput(piece, tools)
            ci.operation = adsk.fusion.FeatureOperations.CutFeatureOperation
            ci.isKeepToolBodies = True
            combines.add(ci)

        for pb in pin_bodies:
            try:
                pb.isVisible = False
            except Exception:
                pass

    if hide_original:
        try:
            body.isVisible = False
        except Exception:
            pass

    # Friendly naming
    for idx, piece in enumerate(wedge_pieces, start=1):
        try:
            piece.name = "Wedge {}".format(idx)
        except Exception:
            pass

    if _ui:
        _ui.messageBox("Split into {} wedge pieces.".format(n))
