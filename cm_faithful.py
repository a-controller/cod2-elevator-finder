"""LINE-BY-LINE transposition of the CoD2 trace stack.

Do not "improve" anything here: every function follows its original,
including comparisons that look redundant -- approximating them changes the
result.

Sources:
  qcommon/cm_trace.cpp:130-215   traceWork_t setup
  qcommon/cm_trace.cpp:854-1087  CM_TraceThroughBrush (SWEPT path)
  qcommon/cm_trace.cpp:1173-1225 CM_TestBoxInBrush   (POINT path)
  bgame/bg_pmove.cpp:1326-1369   PM_CorrectAllSolid
  bgame/bg_pmove.cpp:3313-3400   PM_GroundTrace (up to the startsolid handling)
  qcommon/cm_local.h:20          SURFACE_CLIP_EPSILON = 0.125

Semantic points verified in the code (and contrary to what was assumed):
  - planes are NOT inflated by the epsilon; dist = plane.dist + radius
    + |normal.z| * offsetZ, identical in both paths (l. 994 and l. 1208);
  - the epsilon acts on the EARLY EXIT (d2 >= min(d1, EPS)) and on the entry
    fraction (f = d1 - EPS): the sweep stops 0.125 short of contact;
  - allsolid is set false as soon as d2 > 0: it depends on the sweep's END
    point, not its start (l. 1011-1014, 1044-1061).

Accepted approximation: the tree walk (CM_TraceThroughLeafBrushNode_r) is
replaced by a loop over a brush list supplied by the caller. This changes no
semantics, only performance.

The axial part of CM_TraceThroughBrush (l. 887-970) applies exactly the same
formula as the general loop, with radiusOffset = (radius, radius,
radius + offsetZ), to the 6 bbox planes. Since bsp_read already exposes
these 6 planes as the brush's first 6 sides, all sides are handled
uniformly -- equivalence verified algebraically, cf. brush_planes()'s
docstring.
"""

SURFACE_CLIP_EPSILON = 0.125          # cm_local.h:20

PLAYER_MINS = (-15.0, -15.0, 0.0)     # g_shared.h playerMins
PLAYER_MAXS_STANDING = (15.0, 15.0, 70.0)
PLAYER_MAXS_CROUCH = (15.0, 15.0, 50.0)   # bg_pmove.cpp:3105
PLAYER_MAXS_PRONE = (15.0, 15.0, 30.0)    # bg_pmove.cpp:3093

# bg_pmove.cpp:63-91 -- exact order, {0,0,1} first
CORRECT_SOLID_DELTAS = (
    (0, 0, 1), (-1, 0, 1), (0, -1, 1), (1, 0, 1), (0, 1, 1),
    (-1, 0, 0), (0, -1, 0), (1, 0, 0), (0, 1, 0),
    (0, 0, -1), (-1, 0, -1), (0, -1, -1), (1, 0, -1), (0, 1, -1),
    (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
    (-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0),
    (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def brush_planes(brush):
    """Raw (normal, dist) of every side, axial included.

    Equivalence with the axial part of CM_TraceThroughBrush: for j and
    sign=-1, the code computes d1 = (start[j] - mins[j])*(-1) - radiusOffset[j].
    With the plane (n = -e_j, dist = -mins[j]) the general loop computes
    d1 = -start[j] - (-mins[j] + radius + |n_z|*offsetZ), i.e.
    mins[j] - start[j] - radius - (offsetZ if j==2). Identical, since
    radiusOffset = (radius, radius, radius + offsetZ)."""
    return [(s['normal'], s['dist']) for s in brush['sides']]


def trace_setup(start, end, mins, maxs):
    """cm_trace.cpp:138-163. Returns (extents_start, extents_end, radius,
    offsetZ). The origin passed in is the FEET; extents.* is the capsule's
    center."""
    offset = tuple((mins[i] + maxs[i]) * 0.5 for i in range(3))
    size = tuple(maxs[i] - offset[i] for i in range(3))
    es = tuple(start[i] + offset[i] for i in range(3))
    ee = tuple(end[i] + offset[i] for i in range(3))
    radius = size[2] if size[0] > size[2] else size[0]
    offset_z = size[2] - radius
    return es, ee, radius, offset_z


def new_trace():
    return {'fraction': 1.0, 'startsolid': False, 'allsolid': False,
            'normal': None, 'contents': 0}


def trace_through_brush(es, ee, radius, offset_z, planes, trace):
    """cm_trace.cpp:854-1087, swept path. Modifies `trace` in place."""
    enter_frac = 0.0
    leave_frac = trace['fraction']
    allsolid = True
    leadside = None

    for n, d in planes:
        # l. 994: no epsilon inflation here
        dist = d + radius + abs(n[2]) * offset_z
        d1 = dot(es, n) - dist
        d2 = dot(ee, n) - dist

        if d1 > 0:
            # l. 1006: entirely in front of the face -> no intersection
            if d2 >= min(d1, SURFACE_CLIP_EPSILON):
                return
            if d2 > 0:
                allsolid = False
            delta = d1 - d2
            f = d1 - SURFACE_CLIP_EPSILON          # l. 1021
            if f > enter_frac * delta:
                enter_frac = f / delta
                if enter_frac >= leave_frac:
                    return
                leadside = (n, d)
            else:
                if leadside is None:
                    leadside = (n, d)
        else:
            if d2 > 0:
                delta = d1 - d2
                if d1 > leave_frac * delta:
                    leave_frac = d1 / delta
                    if enter_frac >= leave_frac:
                        return
                allsolid = False

    if leadside is not None:
        trace['fraction'] = enter_frac
        trace['normal'] = leadside[0]
        return

    trace['startsolid'] = True                      # l. 1080
    if allsolid:
        trace['allsolid'] = True
        trace['fraction'] = 0.0


def test_box_in_brush(es, radius, offset_z, planes, trace):
    """cm_trace.cpp:1173-1225, point path (start == end).
    No epsilon at all here."""
    for n, d in planes:
        dist = d + radius + abs(n[2]) * offset_z    # l. 1208
        if dot(es, n) - dist > 0:                   # l. 1215
            return
    trace['startsolid'] = True
    trace['allsolid'] = True
    trace['fraction'] = 0.0
    return


def player_trace(start, end, mins, maxs, brush_planes_list):
    """PM_playerTrace: start == end -> point trace, else swept.
    `brush_planes_list` = list of plane lists (one element per candidate
    solid brush)."""
    es, ee, radius, offset_z = trace_setup(start, end, mins, maxs)
    trace = new_trace()
    is_point = (es == ee)
    for planes in brush_planes_list:
        if is_point:
            test_box_in_brush(es, radius, offset_z, planes, trace)
            if trace['allsolid']:
                return trace
        else:
            trace_through_brush(es, ee, radius, offset_z, planes, trace)
            if trace['allsolid']:
                return trace
    return trace


def pm_ground_trace_allsolid(origin, maxs, brush_planes_list):
    """bg_pmove.cpp:3313-3335: sweep from origin+0.25 to origin-0.25, then
    `if (trace.allsolid)` -> PM_CorrectAllSolid. Returns (allsolid, trace)."""
    start = (origin[0], origin[1], origin[2] + 0.25)
    end = (origin[0], origin[1], origin[2] - 0.25)
    tr = player_trace(start, end, PLAYER_MINS, maxs, brush_planes_list)
    return tr['allsolid'], tr


def pm_correct_all_solid(origin, maxs, brush_planes_list):
    """bg_pmove.cpp:1326-1369. Returns (ok, new_origin).
    Strict translation, including the final Vec3Lerp."""
    for dx, dy, dz in CORRECT_SOLID_DELTAS:
        point = (origin[0] + dx, origin[1] + dy, origin[2] + dz)
        tr = player_trace(point, point, PLAYER_MINS, maxs, brush_planes_list)  # point trace
        if tr['startsolid']:
            continue
        new_origin = point
        down = (new_origin[0], new_origin[1], new_origin[2] - 1.0)
        tr2 = player_trace(new_origin, down, PLAYER_MINS, maxs, brush_planes_list)
        # Vec3Lerp(origin, down, fraction, origin)
        f = tr2['fraction']
        settled = tuple(new_origin[i] + (down[i] - new_origin[i]) * f for i in range(3))
        return True, settled
    return False, origin


MIN_WALK_NORMAL = 0.69999999          # bg_public.h:1298


def pm_ground_trace(origin, maxs, brush_planes_list):
    """bg_pmove.cpp:3313-3416, COMPLETE transposition.

    Returns (origin, state) where state contains:
      corrected  : PM_CorrectAllSolid moved the player
      stuck      : PM_CorrectAllSolid failed (returned false, l. 3336-3339)
      walking    : pml->walking
      groundPlane: pml->groundPlane
      onground   : ps->groundEntityNum != ENTITYNUM_NONE
    """
    st = {'corrected': False, 'stuck': False, 'walking': False,
          'groundPlane': False, 'onground': False}
    start = (origin[0], origin[1], origin[2] + 0.25)
    point = (origin[0], origin[1], origin[2] - 0.25)
    tr = player_trace(start, point, PLAYER_MINS, maxs, brush_planes_list)

    if tr['allsolid']:                                   # l. 3334
        ok, new_origin = pm_correct_all_solid(origin, maxs, brush_planes_list)
        if not ok:                                       # l. 3336-3339
            st['stuck'] = True
            return origin, st
        origin = new_origin
        st['corrected'] = True
        # PM_CorrectAllSolid overwrote pml->groundTrace with its own downward trace
        down = (origin[0], origin[1], origin[2] - 1.0)
        tr = player_trace(origin, down, PLAYER_MINS, maxs, brush_planes_list)

    if tr['startsolid']:                                 # l. 3342
        start2 = (origin[0], origin[1], origin[2] - 0.001)
        tr = player_trace(start2, point, PLAYER_MINS, maxs, brush_planes_list)
        if tr['startsolid']:                             # l. 3347-3354
            return origin, st                            # off the ground, not walking

    if tr['fraction'] == 1.0:                            # l. 3360 -> GroundTraceMissed
        return origin, st                                # free fall

    n = tr['normal']
    if n is not None and n[2] < MIN_WALK_NORMAL:         # l. 3389
        st['groundPlane'] = True
        return origin, st

    st['groundPlane'] = True                             # l. 3399-3409
    st['walking'] = True
    st['onground'] = True
    return origin, st


def _selftest():
    """Minimal sanity checks of the setup."""
    es, ee, r, oz = trace_setup((0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                                PLAYER_MINS, PLAYER_MAXS_STANDING)
    assert es == (0.0, 0.0, 35.0), es
    assert (r, oz) == (15.0, 20.0), (r, oz)
    es, ee, r, oz = trace_setup((0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                                PLAYER_MINS, PLAYER_MAXS_CROUCH)
    assert es == (0.0, 0.0, 25.0), es
    assert (r, oz) == (15.0, 10.0), (r, oz)

    # unit cube around the origin, raw planes
    cube = [((1, 0, 0), 50.0), ((-1, 0, 0), 50.0), ((0, 1, 0), 50.0),
            ((0, -1, 0), 50.0), ((0, 0, 1), 50.0), ((0, 0, -1), 50.0)]
    tr = player_trace((0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                      PLAYER_MINS, PLAYER_MAXS_STANDING, [cube])
    assert tr['allsolid'] and tr['startsolid'], tr
    tr = player_trace((500.0, 0.0, 0.0), (500.0, 0.0, 0.0),
                      PLAYER_MINS, PLAYER_MAXS_STANDING, [cube])
    assert not tr['startsolid'], tr
    print("selftest OK: radius/offsetZ 15/20 (standing) and 15/10 (crouch),"
          " point inside -> allsolid, point outside -> nothing")


if __name__ == '__main__':
    _selftest()
