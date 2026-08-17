"""BRUSH-level characterization of the elevator's geometric conjunction.

    python characterize.py map-cost <map> [--jobs N] [--no-budget]

Imports `cm_leafs.py`, `cm_faithful.py`, `detect.py` READ-ONLY, does not
modify them.

Hierarchy of the production pipeline:

  condition_b_geom + `b in brushes_for_sweep`
                        SEARCH. The asymmetry that makes an elevator possible:
                        the box is inside b, the point probe does not list b,
                        and the swept probe does. Requiring the sweep term is
                        what stops a big under-listed brush from being credited
                        with a neighbour's elevator.

  _column_has_potential SEARCH, widened to the 9 upward deltas. A prefilter
                        only has to avoid false negatives, so a superset is
                        legitimate here -- and necessary: restricting it to
                        index 0 under-explores and loses real spots (jm_zoop,
                        8 of them, confirmed in game).

  _best_climb_at_column THE VERDICT: the real `climb()`, i.e. the engine loop,
                        run from the start of every escape run. It replaces
                        `run_delta0_max`, a proxy that counted escapes at a
                        FIXED column and was wrong in BOTH directions -- it
                        missed climbs that mix deltas (the player leaves the
                        column) and, widened, invented climbs where the player
                        is merely shoved sideways once. Both failure modes were
                        checked in game.

`run_delta0_max` survives as the CSV column name only; it now carries a
`climb()` STEP COUNT, not a run length. The name is kept for one release so
existing CSVs stay readable.

`vertical_extent_asym` (condition b alone, on a single isolated brush) is not
sufficient by itself and does not always even point at the right column
(#6453: extent 217, higher than the 4 known positives, yet no point of ITS
winning column escapes via index 0).
"""
import csv
import math
import os
import random
import sys

from cm_faithful import (PLAYER_MINS, PLAYER_MAXS_STANDING, new_trace,
                          trace_setup, test_box_in_brush, player_trace,
                          CORRECT_SOLID_DELTAS)
import cm_leafs
from cm_leafs import ClipMap
import detect
from detect import Detector

HERE = os.path.dirname(os.path.abspath(__file__))
# Results directory, INSIDE the project directory (it used to be the PARENT
# directory, which wrote outside a cloned repository). Override with the
# COD2_OUT_DIR environment variable.
SCRATCH = os.environ.get('COD2_OUT_DIR') or os.path.join(HERE, 'scratch')

STEP_GRID = 8.0                      # seed grid spacing (the detector's own)
REFINE_RADIUS = 8.0                   # local walk radius per round -- FIXED, independent of STEP_GRID
                                       # (coupling the two narrows the walk's total reach
                                       # instead of refining it, cf. _optimize_column)
DZ = 1.0                             # Z sampling step (grain of a {0,0,1} delta)
MARGIN_Z = 200.0                      # Z scan extent beyond the brush's bounds
MAX_ITER_REFINE = 6                 # local refinement rounds (walk toward the best neighbor)

def mask_for(name):
    return 0x02810011 if name.startswith('mp_') else 0x0281C011


# --------------------------------------------------------- base geometry

def is_axial(n, tol=1e-6):
    nz = [abs(c) for c in n if abs(c) > tol]
    return len(nz) == 1 and abs(nz[0] - 1.0) < 1e-4


def leafs_overlapping(cm, mn, mx):
    """Leafs whose BSP volume overlaps the brush's exact bounds.

    `box_leafnums` (CM_BoxLeafnums_r applied to mn/mx) gives exactly the
    leafs whose partition region intersects the box -- the same function as
    the engine's point probe. The extra bbox filter discards border leafs
    where only the partition region (potentially larger than the leaf's
    stored bbox) touches the box."""
    out = []
    for li in cm.box_leafnums(mn, mx):
        lmn, lmx, _, _ = cm.leafs[li]
        if (lmn[0] < mx[0] and lmx[0] > mn[0] and lmn[1] < mx[1] and lmx[1] > mn[1]
                and lmn[2] < mx[2] and lmx[2] > mn[2]):
            out.append(li)
    return out


def build_listed_orphan_index(cm, mask):
    """Reconstructs, for each leaf, the set of brushes it lists (masked
    path). Requires `cm.build_cache` to already have been called (done by
    `Detector.__init__`)."""
    return cm._lb   # already: tuple of brushes per leaf, filtered by mask


# ------------------------------------------------------------- the core measurement

def condition_b_geom(cm, b, planes, x, y, z, maxs):
    """Cheap half of the asymmetry test EVALUATED AT POINT (x, y, z): the
    point is geometrically inside brush b (test_box_in_brush, exact point
    path), and b does not appear in the leafs the point probe selects at
    that point (brushes_for_point, cm_leafs.py:280-292). Tests ordered
    cheapest first: geometric containment, then the (smaller) point-probe
    box.

    Does NOT test the sweep-probe membership (b in brushes_for_sweep) --
    that is the costly half, deferred to the caller so it can reuse the
    sweep set `ground_allsolid` already computes instead of paying for
    `brushes_for_sweep` a second time (cf. `_delta0_escapes`)."""
    es, ee, radius, offset_z = trace_setup((x, y, z), (x, y, z), PLAYER_MINS, maxs)
    tr = new_trace()
    test_box_in_brush(es, radius, offset_z, planes, tr)
    if not tr['startsolid']:
        return False
    offset = tuple((PLAYER_MINS[j] + maxs[j]) * 0.5 for j in range(3))
    size = tuple(maxs[j] - offset[j] for j in range(3))
    start = tuple((x, y, z)[j] + offset[j] for j in range(3))
    bmin = tuple(start[j] - size[j] - 1.0 for j in range(3))
    bmax = tuple(start[j] + size[j] + 1.0 for j in range(3))
    lb = cm._lb
    for li in cm.box_leafnums(bmin, bmax):
        if b in lb[li]:
            return False        # listed somewhere in the point probe: no asymmetry here
    return True


def _z_window_inside(planes, x, y, maxs, zlo, zhi):
    """The z interval of column (x, y) where the player box is inside brush
    `planes`, computed ANALYTICALLY. Returns (lo, hi) clipped to
    [zlo, zhi], or None when the column never touches the brush.

    Why: `test_box_in_brush` is a conjunction of half-space tests, and the box
    only translates along z, so each plane's test is linear in z --
    `A + n_z*z <= D` -- and bounds z from one side (or, when `n_z == 0`,
    settles the whole column at once). Intersecting them gives the exact
    interval, so every z outside it can be skipped without a single leaf
    lookup.

    Measured motive: profiling 12 brushes of jm_zoop spent 67s of 133s inside
    `condition_b_geom`, called 1,845,760 times, 43s of it in `box_leafnums`.
    Most of those calls are at heights where the box is nowhere near the
    brush. Exact, no false negative: it reproduces `test_box_in_brush`'s own
    inequality (cm_faithful.py:136-146)."""
    offset = tuple((PLAYER_MINS[j] + maxs[j]) * 0.5 for j in range(3))
    size = tuple(maxs[j] - offset[j] for j in range(3))
    radius = size[2] if size[0] > size[2] else size[0]
    offset_z = size[2] - radius
    ex, ey = x + offset[0], y + offset[1]
    lo, hi = zlo, zhi
    for n, d in planes:
        dist = d + radius + abs(n[2]) * offset_z
        a = n[0] * ex + n[1] * ey + n[2] * offset[2]   # constant part
        if abs(n[2]) < 1e-12:
            if a - dist > 0:
                return None            # fails at every height
            continue
        bound = (dist - a) / n[2]
        if n[2] > 0:
            hi = min(hi, bound)
        else:
            lo = max(lo, bound)
        if lo > hi:
            return None
    return (lo, hi)


def _close_run(run_start, run_end, best):
    """`run_start`/`run_end` are the centers of a run's first and last True
    sample, spaced by DZ: the run covers DZ more than their raw gap (each
    sample represents a slice of height DZ, not a point). Reported bounds =
    centered cells. Without this half-step on each side, the measured run
    underestimates the true extent by DZ (bias observed on hill400_assault:
    #7588 measured 27 for a true 28)."""
    length = (run_end - run_start) + DZ
    if length > best[0]:
        return (length, run_start - DZ / 2.0, run_end + DZ / 2.0)
    return best


def _domain_columns(cm, mn, mx, orphan_leafs, maxs, max_cols=None):
    """The DOMAIN of columns (x, y) shared by vertical_extent_asym AND
    run_delta0_max (the two measures remain distinct, but search the SAME
    set of columns): grid step STEP_GRID + explicit corners/edges of the
    brush/orphan-leaf XY overlap zone, DILATED in X and Y by the player box's
    extent -- same reasoning as best_run_delta0's exact Z bound: (b)
    requires the player box AT THE PROBED POINT to intersect the brush, so a
    column outside the overlap rectangle can still see the box bite into the
    brush -- #4957's 8-unit gap was precisely that, structural, invisible at
    any grid resolution without this dilation. Exact, zero free parameter,
    same justification as in Z:
        x in [ rect.x0 - maxs[0], rect.x1 - PLAYER_MINS[0] ]   (and likewise in y)

    `max_cols`: give up and return None past that many columns. A brush can
    own hundreds of orphan leafs (libya #3683: 335), and building its full
    domain exhausts memory BEFORE the TOO_BIG gate ever gets to reject it --
    measured, a MemoryError with a 4 GB peak. Cost grows with the column
    count, so anything above the cap would have been rejected anyway: capping
    changes no verdict, it only stops paying for one.
    """
    dxlo, dxhi = maxs[0], -PLAYER_MINS[0]
    dylo, dyhi = maxs[1], -PLAYER_MINS[1]
    cols = set()
    for li in orphan_leafs:
        lmn, lmx, _, _ = cm.leafs[li]
        ox0, ox1 = max(mn[0], lmn[0]), min(mx[0], lmx[0])
        oy0, oy1 = max(mn[1], lmn[1]), min(mx[1], lmx[1])
        if ox1 < ox0 or oy1 < oy0:
            continue
        ox0, ox1 = ox0 - dxlo, ox1 + dxhi
        oy0, oy1 = oy0 - dylo, oy1 + dyhi
        # corners, guaranteed even if (ox1-ox0) isn't a multiple of STEP_GRID
        cols.add((ox0, oy0)); cols.add((ox0, oy1))
        cols.add((ox1, oy0)); cols.add((ox1, oy1))
        cols.add(((ox0 + ox1) * 0.5, (oy0 + oy1) * 0.5))
        if max_cols is not None and len(cols) > max_cols:
            return None
        # edges, step STEP_GRID along the 4 borders
        x = ox0
        while x <= ox1:
            cols.add((x, oy0)); cols.add((x, oy1))
            x += STEP_GRID
            if max_cols is not None and len(cols) > max_cols:
                return None
        y = oy0
        while y <= oy1:
            cols.add((ox0, y)); cols.add((ox1, y))
            y += STEP_GRID
            if max_cols is not None and len(cols) > max_cols:
                return None
        # inner grid. This one is quadratic in the rectangle, not linear like
        # the edges above, so it is where an oversized brush actually hurts:
        # libya #3683 spans the whole map and asks for 18434 x 14493 columns,
        # 267 million of them. The cap has to be tested here too.
        x = ox0
        while x <= ox1:
            y = oy0
            while y <= oy1:
                cols.add((x, y))
                y += STEP_GRID
            x += STEP_GRID
            if max_cols is not None and len(cols) > max_cols:
                return None
    return cols


def _optimize_column(cols, evaluate, refine=True):
    """Evaluates `evaluate(x, y) -> (length, zlo, zhi)` over `cols`
    (coarse+edges pass), then an ITERATIVE local walk (not a single pass:
    the best coarse candidate can be far away -- measured, 16.6 units on
    #3455 and 18.3 on #4957, beyond a single REFINE_RADIUS=8). Recenters on
    the best found each round, stops when it no longer moves (or after
    MAX_ITER_REFINE rounds). Reused by both vertical_extent_asym AND
    run_delta0_max, with two different `evaluate` functions: they search the
    same domain but legitimately peak at different columns.

    REFINE_RADIUS is FIXED, INDEPENDENT of STEP_GRID: the walk's total
    reach (REFINE_RADIUS * MAX_ITER_REFINE) must never depend on the seed
    grid's resolution, or tightening the grid shortens the walk instead of
    improving it -- a real trap, already measured on vertical_extent_asym
    (#3455 regresses from 154 to 131 if the two are coupled).

    Returns (best, best_col, n_tested)."""
    best = (0.0, None, None)
    best_col = None
    for (x, y) in cols:
        r = evaluate(x, y)
        if r[0] > best[0]:
            best, best_col = r, (x, y)

    n_tested = len(cols)
    tested = set(cols)
    for _ in range(MAX_ITER_REFINE if refine else 0):
        if best_col is None:
            break
        bx, by = best_col
        refined = set()
        xr = bx - REFINE_RADIUS
        while xr <= bx + REFINE_RADIUS:
            yr = by - REFINE_RADIUS
            while yr <= by + REFINE_RADIUS:
                refined.add((xr, yr))
                yr += 1.0
            xr += 1.0
        refined -= tested
        if not refined:
            break
        new_col = None
        for (x, y) in refined:
            r = evaluate(x, y)
            if r[0] > best[0]:
                best, new_col = r, (x, y)
        tested |= refined
        n_tested += len(refined)
        if new_col is None:
            break            # no improvement: at a local peak
        best_col = new_col

    return best, best_col, n_tested


def _delta0_escapes(cm, b, planes, det, mask, x, y, z, maxs, widen=False):
    """Tests ONLY index 0 ({0,0,1}) of PM_CorrectAllSolid, not all 26 --
    exact, not an approximation (per the source: {0,0,1} is tried first,
    escapes -> taken, regardless of the other 25).

    `widen=True` (SEARCH only, never measurement) also accepts the other 8
    upward deltas. It must stay confined to `_column_has_potential`:
    `run_delta0_max` counts escapes at a FIXED column, which only models a
    climb when the motion is purely vertical. A lateral delta such as
    (-1,0,1) or (0,1,1) still gains +1 in z, so the run keeps counting while
    the player actually leaves the column -- measured on mp_farmhouse #3454
    and mp_trainstation #3308: run 77 and 81 recorded, real climb 1, and
    in-game the player is simply shoved sideways once. Widening the SEARCH
    finds real spots (jm_zoop, 8 new `True`, two confirmed in game);
    widening the MEASUREMENT manufactures false DIVERGENCE.

    Prefilters via `condition_b_geom` AT THE ORIGIN (x, y, z) BEFORE the
    costly `ground_allsolid` gate: (b) is a NECESSARY precondition for any
    step of a climb attributable to THIS brush -- no attributable run can
    leave its (b) region, so testing (b), cheaper (`box_leafnums` alone),
    before paying for a full swept trace, loses no true positive.

    `ground_sweep_brushes` computes the swept brush set once; `b in sweep`
    (the other half of the asymmetry condition, cf. `condition_b_geom`'s
    docstring) is tested on it directly, BEFORE paying for the costly
    `planes_of` + trace that turns that set into an `allsolid` verdict
    (`ground_allsolid_from_sweep`) -- so a brush absent from the sweep
    skips the trace entirely, same as before this function existed, while
    a brush present in it never recomputes `brushes_for_sweep`."""
    if not condition_b_geom(cm, b, planes, x, y, z, maxs):
        return False
    pos = (x, y, z)
    sweep = det.ground_sweep_brushes(pos, maxs)
    if b not in sweep:
        return False            # never submitted to the sweep probe: cannot carry asymmetry
    if not det.ground_allsolid_from_sweep(pos, maxs, sweep):
        return False
    if widen:
        # Cheaper superset: a prefilter only has to avoid false negatives.
        # The true condition is "the FIRST escaping delta goes up"; "at least
        # one UPWARD delta escapes" implies it can never be missed, and only
        # walks the 9 deltas with dz > 0 instead of all 26.
        for dx, dy, dz in UPWARD_DELTAS:
            point = (x + dx, y + dy, z + dz)
            nums = cm.brushes_for_point(point, PLAYER_MINS, maxs, mask)
            if not player_trace(point, point, PLAYER_MINS, maxs,
                                cm.planes_of(nums))['startsolid']:
                return True
        return False
    if widen and BROAD_DELTA:   # experiment override, off by default
        # Experiment (COD2_BROAD_DELTA=1): the real engine condition is "the
        # FIRST delta that escapes goes up", not "delta 0 escapes". Measured
        # on cor27_intoxication #2311: at 2.0 of penetration delta 0 is
        # blocked, delta 1 (-1,0,1) escapes upward, and the climb still runs
        # 74 steps -- a case the narrow test cannot see.
        for dx, dy, dz in CORRECT_SOLID_DELTAS:
            point = (x + dx, y + dy, z + dz)
            nums = cm.brushes_for_point(point, PLAYER_MINS, maxs, mask)
            if not player_trace(point, point, PLAYER_MINS, maxs,
                                cm.planes_of(nums))['startsolid']:
                return dz > 0
        return False
    dx, dy, dz = CORRECT_SOLID_DELTAS[0]
    point = (x + dx, y + dy, z + dz)
    nums = cm.brushes_for_point(point, PLAYER_MINS, maxs, mask)
    tr = player_trace(point, point, PLAYER_MINS, maxs, cm.planes_of(nums))
    return not tr['startsolid']


def _best_climb_at_column(cm, b, planes, det, mask, x, y, zlo, zhi, maxs):
    """THE VERDICT: the real `climb()`, run at EVERY escaping height of this
    column. Returns (n_steps, z_start).

    NOTE: every escaping z, not just the start of each run. Starting lower does
    NOT imply a longer climb: on jm_kuwehr #17446 the run starts at a height
    where the climb is 1 step, while the best height of the SAME column climbs
    136 -- testing run starts only dropped a 137-step elevator outright.

    Replaces `_run_delta0_at_column` as the production criterion. That one
    counts index-0 escapes at a FIXED column, which only models a climb when
    the motion is purely vertical, so it is wrong in BOTH directions:

      - it MISSES real climbs that mix deltas -- the player leaves the column
        and the run stops short of MIN_STEP (jm_zoop 13/14/39/40/45/46/71/72,
        all confirmed in game);
      - widened to lateral upward deltas it INVENTS climbs -- the run keeps
        counting while the player escapes sideways (mp_farmhouse #3454: run
        77, real climb 1; confirmed in game as a single sideways shove).

    `climb()` is the engine loop itself, so there is nothing left to
    approximate. The widened gate stays where it belongs, as the SEARCH for
    candidate heights; the verdict is then exact. Measured: jm_zoop
    25 True/11 DIVERGENCE -> 36/8, mp_farmhouse 2/0 -> 2/0 with the three
    false positives gone, for about +20% runtime."""
    big, small = detect.boxes()
    probe = detect.MIN_STEP + 1        # enough to answer ">= MIN_STEP ?"
    best = (0, None)
    win = _z_window_inside(planes, x, y, maxs, zlo, zhi)
    if win is None:
        return best
    z, zhi = win[0], win[1]
    while z <= zhi:
        if _delta0_escapes(cm, b, planes, det, mask, x, y, z, maxs, widen=True):
            # Capped climb: the loop only has to decide whether the height
            # reaches MIN_STEP. Running it to 400 at every height was what
            # pushed the selftest past 900s -- a 136-step climb costs 136
            # iterations of 26 point traces, and the column holds ~130 such
            # heights. The winner alone is then measured in full.
            n, zf, i0 = det.climb((x, y, z), limit=probe)
            if n > best[0]:
                best = (n, z)
            if n >= detect.MIN_STEP:
                # `check()` inlined, minus the climb it would redo: standing
                # allsolid is implied by the climb, so only the crouched box
                # is left to test (detect.py:142-152, RELAX=False).
                if not det.ground_allsolid((x, y, z), small):
                    full, _, _ = det.climb((x, y, z), limit=400)
                    return (full, z)          # confirmed spot, stop here
        z += DZ
    if best[1] is not None and best[0] >= detect.MIN_STEP:
        full, _, _ = det.climb((x, y, best[1]), limit=400)
        return (full, best[1])
    return best


def _run_delta0_at_column(cm, b, planes, det, mask, x, y, zlo, zhi, maxs):
    """Longest continuous vertical run where delta 0 escapes precisely, at a
    fixed column (x, y)."""
    z = zlo
    run_start = None
    run_end = None
    best = (0.0, None, None)
    while z <= zhi:
        ok = _delta0_escapes(cm, b, planes, det, mask, x, y, z, maxs)
        if ok:
            if run_start is None:
                run_start = z
            run_end = z
        elif run_start is not None:
            best = _close_run(run_start, run_end, best)
            run_start = None
        z += DZ
    if run_start is not None:
        best = _close_run(run_start, run_end, best)
    return best


def _column_has_potential(cm, b, planes, det, mask, x, y, zlo, zhi, maxs):
    """EXACT filter, zero false negative: a run of >= MIN_STEP consecutive z
    (step DZ, same phase as the fine scan, which also starts at `zlo`)
    necessarily contains a point of the sub-sample
    `zlo, zlo + MIN_STEP*DZ, zlo + 2*MIN_STEP*DZ, ...` -- MIN_STEP consecutive
    terms of an arithmetic sequence with step DZ cover every residue class
    modulo MIN_STEP. If none of these points escapes via index 0, the column
    cannot carry ANY run >= MIN_STEP: a ~MIN_STEP factor on the coarse pass's
    cost, losing nothing that matters (runs < MIN_STEP are not spots anyway,
    cf. `detect.MIN_STEP`)."""
    win = _z_window_inside(planes, x, y, maxs, zlo, zhi)
    if win is None:
        return False          # the box never enters the brush on this column
    wlo, whi = win
    # Keep the sub-sampling PHASE anchored on zlo: the soundness argument
    # above is about residues modulo MIN_STEP of the fine scan, which also
    # starts at zlo. Restarting at wlo would shift the phase and could skip
    # the only escaping height.
    step_filter = detect.MIN_STEP * DZ
    z = zlo + math.ceil((wlo - zlo) / step_filter) * step_filter if wlo > zlo else zlo
    while z <= whi:
        if _delta0_escapes(cm, b, planes, det, mask, x, y, z, maxs, widen=True):
            return True
        z += step_filter
    return False


def best_run_delta0(cm, det, mask, b, cols, mn, mx, maxs, optimiser=True):
    """run_delta0_max: THE PREDICATE (a^b^c^d directly -- comparable term
    for term to ground truth's `step`). EXHAUSTIVE search (not a single-start
    greedy walk: on a landscape this sparse -- 0.4% of points escape on
    #6453 -- a greedy walk has no gradient to follow and can miss the true
    maximum through bad luck on its starting point, measured on #4957: the
    real column is 7.3 units from the coarse domain, never found). Instead:

      1. an EXHAUSTIVE coarse pass over the whole `cols` domain, filtered by
         `_column_has_potential` (sound, a ~MIN_STEP cost factor);
      2. a full scan (step DZ) + local XY refinement (like
         `vertical_extent_asym`) on EACH survivor of the filter -- rare by
         nature, so the full cost is affordable only on them.

    Returns (run_delta0_max, zlo, zhi, n_cols, winning_column,
    n_survivors). `run_delta0_max is None` = CENSORED (`<MIN_STEP`
    everywhere -- never write it as a 0). `n_survivors` = number of
    columns in the coarse domain that passed the mod-10 filter, BEFORE
    grouping -- the number that says whether production is viable, not the
    total time.

    EXACT Z bounds, not padding: `[zlo_asym, zhi_asym]` only measures
    (b) AT vertical_extent_asym's WINNING COLUMN -- (b)'s vertical extent
    VARIES with the column (measured: the seed point that starts the walk
    toward #4957's true maximum needs a z outside that range). The real
    bound, valid for ALL columns at once, follows only from the requirement
    that the player box intersect the brush:
        z in [ brush.mins.z - player.maxs.z , brush.maxs.z - player.mins.z ]
    Outside it, the box cannot touch the brush: (b) is false, whatever the
    column. Zero free parameter, O(1), by construction contains the union of
    every column's (b) region."""
    planes = cm._pl[b]
    zlo = mn[2] - maxs[2]
    zhi = mx[2] - PLAYER_MINS[2]
    survivors = [(x, y) for (x, y) in cols
                   if _column_has_potential(cm, b, planes, det, mask, x, y, zlo, zhi, maxs)]
    n_survivors = len(survivors)
    if not survivors:
        return None, None, None, len(cols), None, n_survivors

    # Proximity grouping BEFORE refinement: adjacent survivors converge to
    # the same maximum via the iterative walk -- refining them separately is
    # pure redundant work (measured: #7267, ~127 refinements). A survivor
    # already covered by the neighborhood explored around a prior
    # representative does not need its own.
    #
    # `optimiser=False` DISABLES both grouping AND memoization (refines EACH
    # survivor individually, no cache) -- reserved for verifying that these
    # two optimizations change NO value, not for assuming they do not. The
    # normal measurement always leaves optimiser=True.
    if optimiser:
        GROUP_RADIUS = 2.0 * REFINE_RADIUS
        representatives = []
        for (x, y) in survivors:
            if not any(abs(x - rx) <= GROUP_RADIUS and abs(y - ry) <= GROUP_RADIUS
                       for (rx, ry) in representatives):
                representatives.append((x, y))
    else:
        representatives = survivors

    # Memoization (x, y) -> result, shared across ALL of this brush's
    # refinements: `_run_delta0_at_column`'s result only depends on (x, y)
    # [b, planes, det, mask, zlo, zhi, maxs are fixed for the whole call] --
    # correct by construction, and nearby refinements' neighborhoods overlap
    # enormously.
    cache = {} if optimiser else None

    def evaluate(x, y):
        if cache is None:
            return _run_delta0_at_column(cm, b, planes, det, mask, x, y, zlo, zhi, maxs)
        key = (x, y)
        r = cache.get(key)
        if r is None:
            r = _run_delta0_at_column(cm, b, planes, det, mask, x, y, zlo, zhi, maxs)
            cache[key] = r
        return r

    best = (0.0, None, None)
    best_col = None
    n_local_total = 0
    for (x, y) in representatives:
        r_local, col_local, n_local = _optimize_column({(x, y)}, evaluate)
        n_local_total += n_local
        if r_local[0] > best[0]:
            best, best_col = r_local, col_local

    # cache = UNIQUE evaluations actually performed (optimiser=True); with no
    # cache, this falls back to the raw per-refinement sum (may count the
    # same column multiple times -- expected, that is exactly what
    # memoization avoids).
    n_tested = len(cols) + (len(cache) if cache is not None else n_local_total)
    return best[0], best[1], best[2], n_tested, best_col, n_survivors


# ---------------------------------------------------- production pipeline

def spot_exists(cm, det, mask, b, mn, mx, orphan_leafs, maxs=PLAYER_MAXS_STANDING):
    """PRODUCTION pipeline: `run_delta0_max` is the only valid discriminant
    between spots and negatives, no cheap column separates them:

      1. Prefilter: XY domain (identical to vertical_extent_asym's, but
         the extent itself is NO LONGER computed -- it serves no purpose
         once the mod-10 filter is validated as sufficient) + survivor count
         (mod-10 filter, sound, zero false negative). NECESSARY condition by
         construction: no survivor => no possible escape => NO parameter
         tuned. Measured on the 400: 4.0% (duhoc_assault) / 4.5% (elalamein)
         survival, ~96% reduction.
      2. On survivors ONLY: full scan (step DZ), WITHOUT XY refinement --
         dropped: refinement/scan ratio measured at ~288 to 1, and
         maximizing gains nothing since negatives cap at 1.0 (the known
         1-step artifact, never a true run).
      3. A question of EXISTENCE, not amplitude: does a survivor have a full
         run >= detect.MIN_STEP? EARLY exit at the first one found (no need
         for the maximum).

    Returns a dict: spot (True/False/None), run_delta0_max (the value of the
    survivor that decided it, or the max found if none reaches MIN_STEP, or
    None if no survivor), n_domain_columns, n_survivors, x_col/y_col.
    `spot is None` = censored (no survivor, no full scan performed -- never
    confused with `spot=False`, where at least one survivor was scanned
    without reaching MIN_STEP)."""
    planes = cm._pl[b]
    cols = _domain_columns(cm, mn, mx, orphan_leafs, maxs)
    zlo = mn[2] - maxs[2]
    zhi = mx[2] - PLAYER_MINS[2]
    survivors = [(x, y) for (x, y) in cols
                   if _column_has_potential(cm, b, planes, det, mask, x, y, zlo, zhi, maxs)]
    n_survivors = len(survivors)
    base = {'n_domain_columns': len(cols), 'n_survivors': n_survivors}
    if not survivors:
        return dict(base, spot=None, run_delta0_max=None, x_col=None, y_col=None, z_col=None)

    # Search criterion (cf. _spot_exists_isolated): SEARCHES for a column that
    # satisfies both run_delta0_max >= MIN_STEP AND Detector.check() (the
    # crouched-not-allsolid guard, RELAX=False) -- not just the first one
    # that passes MIN_STEP. A column where the mechanism runs but check()
    # refuses is a DIVERGENCE, not a spot.
    raw_best = (0.0, None, None, None)
    best_blocked = None
    for (x, y) in survivors:
        r = _best_climb_at_column(cm, b, planes, det, mask, x, y, zlo, zhi, maxs)
        if r[0] > raw_best[0]:
            raw_best = (r[0], x, y, r[1])
        if r[0] >= detect.MIN_STEP:
            chk = det.check((x, y, r[1]))
            if chk['spot']:
                return dict(base, spot=True, run_delta0_max=r[0], x_col=x, y_col=y, z_col=r[1])
            if best_blocked is None or r[0] > best_blocked[0]:
                best_blocked = (r[0], x, y, r[1])
    if best_blocked is not None:
        run0, x_col, y_col, z_col = best_blocked
        return dict(base, spot='DIVERGENCE', run_delta0_max=run0, x_col=x_col, y_col=y_col, z_col=z_col)
    run0, x_col, y_col, z_col = raw_best
    return dict(base, spot=False, run_delta0_max=run0, x_col=x_col, y_col=y_col, z_col=z_col)


# ------------------------------------------------------------ map loading

_CACHE = {}

# Disk fallback: if the memory dump `<name>.txt` does not exist (map never
# loaded in-game), the ClipMap is reconstructed from the compiled `.d3dbsp`
# via disk_clipmap.py. The `.txt` path stays PRIORITY and unchanged: this
# fallback only triggers when the dump is absent.
# /!\ The disk reader is NOT equivalent to the memory dump: measured on
# mp_farmhouse, it recovers only 1 of the 2 known spots. Brush #3455 differs
# between the two sources -- mins.z 200.0 (memory) vs 208.0 (disk), and 4
# non-axial planes instead of 3 (a bevel added at compile time) -- which
# changes its leaf coverage and gets it rejected by the prefilter. A positive
# obtained via this path should be confirmed by an in-game capture.
# Dumps and .d3dbsp files live in the SAME directory: the one given by
# --maps-dir / COD2_MAPS_DIR, or `maps/` next to these scripts. There is no
# separate fallback anymore -- the previous one pointed at the PARENT of the
# project directory, i.e. outside a cloned repository.


def _bsp_dirs():
    """--maps-dir (CLI) > COD2_MAPS_DIR (env) > maps/ next to these scripts."""
    return [cm_leafs.maps_dir()]


def _find_bsp(name):
    for d in _bsp_dirs():
        p = os.path.join(d, name + '.d3dbsp')
        if os.path.exists(p):
            return p
    return None


def _load_clipmap(name):
    """Memory dump if present, else fall back to the compiled .d3dbsp."""
    txt = os.path.join(cm_leafs.maps_dir(), name + '.txt')
    if os.path.exists(txt):
        try:
            return ClipMap(txt)
        except OSError as e:
            raise cm_leafs.MapLoadError("cannot read memory dump %s: %s" % (txt, e))
    bsp = _find_bsp(name)
    dirs = _bsp_dirs()
    if bsp is None:
        raise cm_leafs.MapLoadError(
            "neither memory dump (%s) nor .d3dbsp for '%s' (looked in %s)"
            % (txt, name, ' ; '.join(dirs)))
    from disk_clipmap import DiskClipMap
    try:
        return DiskClipMap(bsp)
    except OSError as e:
        raise cm_leafs.MapLoadError("cannot read %s: %s" % (bsp, e))


def get_map(name):
    """`Detector.__init__` reads the GLOBAL variable `detect.MASK` (not a
    parameter) for its `build_cache`, and `ground_allsolid`/
    `correct_all_solid` do the same on every call. With MP (0x02810011) and
    SP (0x0281C011) maps mixed in the same session, `detect.MASK` must be
    reset before EVERY use of a Detector, not just at its construction."""
    if name not in _CACHE:
        mask = mask_for(name)
        detect.MASK = mask
        cm = _load_clipmap(name)
        det = Detector(cm)
        _CACHE[name] = (cm, det, mask)
    return _CACHE[name]


def clear_cache():
    """Drops every loaded clipmap and Detector.

    The cache pays off while a scan hammers ONE map, but it never evicts. A
    caller that walks many maps in a row, touching each once, accumulates all
    of them in memory: `production.py estimate` over a full maps directory was
    killed by the OOM reaper this way. Such callers must clear between maps.
    """
    _CACHE.clear()


# --------------------------------------------------------------- orphans (neg)

BRUSH_KINDS = ('non-axial', 'axial', 'both')


def kind_of(cm, b):
    """'non-axial' if brush `b` has at least one non-axial plane, else
    'axial'. Never returns 'both', which is a selection, not a shape."""
    return 'non-axial' if cm.brushes[b][3] else 'axial'


def orphans_of(name, kind='non-axial'):
    """List of a map's orphan brush_ids: mask + `kind` + orphan (at least one
    leaf that overlaps it without listing it).

    `kind` selects on the brush's OWN planes:
      'non-axial'  at least one non-axial plane. The historical default, and
                   what every result so far was measured with.
      'axial'      pure AABB, no non-axial plane. Never scanned before.
      'both'       no selection on shape.

    Why 'axial' is worth scanning: the escape comes from the 2048 margin that
    `CM_TraceThroughTree` applies to a non-axial BSP *node* plane, which has
    nothing to do with the carrier brush's own shape. A pure AABB can be an
    orphan under such a node just as well. The "every spot has an oblique
    plane" invariant is a selection effect of this filter, not a finding.

    Cost: axial orphans are a minority (measured: 16 against 188 on jm_temple,
    12 against 156 on jm_zoop, 318 against 1686 on mp_farmhouse). And the CSV
    resume skips whatever is already recorded, so adding them to a map that was
    already scanned only processes the new ones.
    """
    if kind not in BRUSH_KINDS:
        raise ValueError('kind must be one of %s' % (BRUSH_KINDS,))
    cm, det, mask = get_map(name)
    out = []
    for b, (mn, mx, ct, pl) in enumerate(cm.brushes):
        if not (ct & mask):
            continue
        if kind == 'non-axial' and not pl:
            continue
        if kind == 'axial' and pl:
            continue
        rec = leafs_overlapping(cm, mn, mx)
        if any(b not in cm._lb[li] for li in rec):
            out.append(b)
    return out


DEFAULT_WORKERS = 3      # explicitly capped (not cpu_count()-1): the scan
                        # already uses 25-30% CPU, keep a margin

MEMORY_CEILING_MB = 2500                        # memory ceiling per child process
PERIODE_MEM = 5.0


# `brush_kind` was added late: CSVs written before it simply lack the column,
# and DictReader yields None for it. Readers must treat a missing value as
# unknown, never as 'axial'.
PRODUCTION_COLUMNS = ['map', 'brush_id', 'spot', 'run_delta0_max',
                        'n_domain_columns', 'n_survivors',
                        't_prefilter', 't_sweep', 'x_col', 'y_col', 'z_col',
                        'estimated_cost', 'brush_kind']

# Prefilter cost/test (mod-10 filter), calibrated on real production
# measurements (t_prefilter, n_domain_columns, height_z). n_tests =
# n_cols * (height_z / 10), EXACTLY the number of iterations of
# `_column_has_potential` (step_filter = MIN_STEP*DZ = 10). Median of the
# individual t/n_tests ratios (robust to the rare brush with an extreme
# domain): 0.0000242285 s/test -- very close to the total/total ratio
# (0.0000233070), the two converge.
COST_PER_TEST_S = 0.0000242285

# Hard cap on the domain size, independent of the time estimate.
#
# The time-derived cap alone is not enough: cost scales with n_cols * height_z,
# so a SHALLOW brush is allowed a huge column count while staying under the
# time budget. libya #3683 (335 orphan leafs) was granted 12 million columns
# that way, which is 1.5 GB of tuples per worker, and the raw build died on a
# MemoryError at 4 GB.
#
# A brush over this cap is reported TOO_BIG, which is honest -- it was never
# measured -- rather than taking the machine down.
#
# 100,000, recalibrated on 2026-08-17. This cap is a MEMORY guard, not a cost
# guard: cost is what BUDGET_BRUSH_S is for. Conflating the two is what made
# the previous value (20,000) drop a real spot.
#
# Calibration, redone properly: the domain of all 174 known elevators
# recomputed FROM THE CLIPMAP, not read from the CSV column. The earlier
# figure ("largest domain that ever produced a spot: 8,984") only covered the
# spots whose CSV happened to record `n_domain_columns`, which silently
# excluded every older CSV -- hill400_assault #8189 among them, a confirmed
# spot with a domain of 35,631 columns, rejected as TOO_BIG by the 20,000 cap.
#
#   median 177, p90 907, p99 8,984, MAX 35,631 (#8189, 4x the next one)
#   cap  20,000 -> 1 elevator lost      cap 50,000 -> 0
#   cap 100,000 -> 0, and 2.8x margin over the record
#
# 100,000 columns is about 12 MB of tuples, against the 1.5 GB per worker that
# motivated having a cap at all (libya #3683 asked for 267 million).
MAX_DOMAIN_COLUMNS = 100 * 1000

# Experiment switch, off by default: widen the delta gate from "index 0
# escapes" to "the first escaping delta goes up". See `_delta0_escapes`.
BROAD_DELTA = os.environ.get('COD2_BROAD_DELTA') == '1'

# Cheaper variant of the same widening (COD2_UP_DELTA=1): only the 9 deltas
# with dz > 0, as a superset prefilter. See `_delta0_escapes`.
UP_DELTA = os.environ.get('COD2_UP_DELTA') == '1'
UPWARD_DELTAS = tuple(d for d in CORRECT_SOLID_DELTAS if d[2] > 0)


def _spot_exists_isolated(name, b, q):
    """Target of a production worker: `spot_exists`, timed in 2 separate
    phases --
    1. prefilter (domain + survivor count, mod-10 filter);
    2. full scan WITHOUT refinement, on survivors only.
    With refinement removed, knowing which of the two dominates the total
    cost says what to optimize first if the number is bad.

    Search criterion: this is NOT an after-the-fact revalidation of the
    first column that reaches MIN_STEP. The pipeline SEARCHES for a column
    that satisfies BOTH conditions at once -- `run_delta0_max >= MIN_STEP`
    AND `Detector.check()` confirms (the `crouched not allsolid` guard,
    RELAX=False, same instrument as `detect.py check`). It keeps scanning
    survivors as long as none passes the guard -- a column where the
    mechanism runs but the guard refuses is NOT a spot, it is a DIVERGENCE:
    concluding on the first column that reaches MIN_STEP would wrongly
    reject real spots (measured on #3455/#4957, whose first column >=
    MIN_STEP fails the guard while a further column confirms it)."""
    import signal
    import time
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    cm, det, mask = get_map(name)
    mn, mx, ct, extra = cm.brushes[b]
    planes = cm._pl[b]
    rec = leafs_overlapping(cm, mn, mx)
    orphan_leafs = [li for li in rec if b not in cm._lb[li]]

    t0 = time.perf_counter()
    cols = _domain_columns(cm, mn, mx, orphan_leafs, PLAYER_MAXS_STANDING)
    zlo = mn[2] - PLAYER_MAXS_STANDING[2]
    zhi = mx[2] - PLAYER_MINS[2]
    survivors = [(x, y) for (x, y) in cols
                   if _column_has_potential(cm, b, planes, det, mask, x, y, zlo, zhi, PLAYER_MAXS_STANDING)]
    t1 = time.perf_counter()

    spot, run0, x_col, y_col, z_col = None, None, None, None, None
    if survivors:
        spot = False
        raw_best = (0.0, None, None, None)   # best run, whatever it is (info, if nothing reaches MIN_STEP)
        best_blocked = None                     # best run >= MIN_STEP but check() refuses (DIVERGENCE)
        found = False
        for (x, y) in survivors:
            r = _best_climb_at_column(cm, b, planes, det, mask, x, y, zlo, zhi, PLAYER_MAXS_STANDING)
            if r[0] > raw_best[0]:
                raw_best = (r[0], x, y, r[1])
            if r[0] >= detect.MIN_STEP:
                chk = det.check((x, y, r[1]))
                if chk['spot']:
                    spot, run0, x_col, y_col, z_col = True, r[0], x, y, r[1]
                    found = True
                    break
                if best_blocked is None or r[0] > best_blocked[0]:
                    best_blocked = (r[0], x, y, r[1])
        if not found:
            if best_blocked is not None:
                spot = 'DIVERGENCE'
                run0, x_col, y_col, z_col = best_blocked
            else:
                run0, x_col, y_col, z_col = raw_best
    t2 = time.perf_counter()

    q.put({'map': name, 'brush_id': b, 'spot': spot, 'run_delta0_max': run0,
           'n_domain_columns': len(cols), 'n_survivors': len(survivors),
           't_prefilter': t1 - t0, 't_sweep': t2 - t1,
           'x_col': x_col, 'y_col': y_col, 'z_col': z_col})


# Hard runtime budget per brush: some brushes with a huge domain can block a
# worker indefinitely.
#
# 60s, measured, not guessed. Across the 106 spots whose timing was recorded,
# the slowest ever took 16.37s (jm_lazlo #137) -- median 0.24s, p90 1.29s. So
# 60s leaves a 3.7x margin over anything that has ever produced a spot, while
# cutting the negative tail, which reaches 253s. At 10s two real spots would
# have been lost, so do not go lower without redoing this measurement.
# Caveat: 77 older spots come from CSVs written before the timing columns
# existed. They completed under the previous 600s budget, but nothing proves
# they ran under 60s.
# 5s, remeasured on 2026-08-17 after the z-window pruning (_z_window_inside).
#
# The old 60s (and the 600s before it) were calibrated on the run_delta0_max
# proxy, where the slowest spot ever took 16.4s. The climb() verdict stops at
# the FIRST confirmed column, so a brush that carries an elevator is now cheap:
# across every map measured with it, the slowest one costs 1.68s
# (mp_farmhouse), 0.23s on jm_zoop, 0.20s on mp_trainstation. The expensive
# brushes are the NEGATIVES, which must exhaust all their columns.
#
# Same verdicts at every budget tried -- jm_zoop 36 spots + 8 DIVERGENCE,
# jm_kuwehr 0 + 4, mp_trainstation 0 + 1 -- only the count of unmeasured
# brushes and the runtime move (jm_zoop, 3 workers):
#     3s -> 46s      5s -> 61s      10s -> 94s      120s -> 200s
# A higher budget buys no elevator, it only settles negatives. Since the
# z-window pruning the slowest carrier across every map measured costs 0.80s
# (median 0.12s, p90 0.28s), so 5s keeps a 6x margin; 3s would cut it to 3.75x,
# too thin on an unseen map given how often a "safe" calibration has turned out
# to rest on a biased sample.
BUDGET_BRUSH_S = 5

# Threshold for the TOO_BIG pre-rejection, on the ESTIMATED cost. Deliberately
# NOT lowered with the runtime budget: the estimate is a rough model, and a
# brush it prices at 90s can run in 2s. Keeping it where it was means this
# change can only convert a slow negative into BUDGET_EXCEEDED, never
# pre-reject a brush that would have been measured before. For reference, the
# most expensive spot ever was estimated at 13.4s.
TOO_BIG_ESTIMATE_S = 10 * 60


def _spot_exists_within_budget(name, b, budget_s=BUDGET_BRUSH_S):
    """DEDICATED sub-process with a hard budget: a thread cannot be canceled
    during a pure CPU computation, a sub-process can be `terminate()`d.
    Returns None if the budget is exceeded (or if the brush crashes with no
    result) -- the caller then writes BUDGET_EXCEEDED, NEVER <10, 0, or
    spot=False: that would fabricate data, and it is precisely on a brush
    with a huge domain that a real spot could be hiding -- never silently
    censor an unfinished brush."""
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=_spot_exists_isolated, args=(name, b, q))
    p.start()
    p.join(budget_s)
    if p.is_alive():
        p.terminate()
        p.join()
        return None
    try:
        return q.get_nowait()
    except Exception:
        return None


class _FileFakeQueue(object):
    """Accepts a single `.put()` -- same interface as multiprocessing.Queue,
    to reuse `_spot_exists_isolated` as-is in DIRECT execution (`--no-budget`
    mode, no sub-process, no budget)."""
    def __init__(self):
        self.item = None

    def put(self, x):
        self.item = x


def _estimated_cost(cm, mn, mx, orphan_leafs, maxs=PLAYER_MAXS_STANDING):
    """Estimates the prefilter's time BEFORE running it -- the domain is
    cheap to build (pure geometry, ~0.2-1s measured even for a huge domain);
    n_domain_columns and the Z interval are therefore known before any
    costly work. n_tests = n_cols * (height_z / 10), EXACTLY the number
    of iterations of the mod-10 filter (`_column_has_potential`). Returns
    (estimated_cost_s, n_cols, cols) -- `cols` is reusable by the caller to
    avoid rebuilding the domain twice.

    The domain is built under a cap derived from `TOO_BIG_ESTIMATE_S`: past it
    the brush is TOO_BIG whatever the exact count, so `cols` comes back None
    and the reported cost is a LOWER BOUND (the cap), enough to order a
    catch-up pass. Without this, a brush with hundreds of orphan leafs
    exhausts memory before it can be rejected."""
    zlo = mn[2] - maxs[2]
    zhi = mx[2] - PLAYER_MINS[2]
    height_z = zhi - zlo
    per_col = max(height_z / 10.0, 1e-9) * COST_PER_TEST_S
    max_cols = min(int(TOO_BIG_ESTIMATE_S / per_col) + 1, MAX_DOMAIN_COLUMNS)
    cols = _domain_columns(cm, mn, mx, orphan_leafs, maxs, max_cols=max_cols)
    if cols is None:
        return max_cols * per_col, None, None
    return len(cols) * per_col, len(cols), cols


def _spot_exists_with_estimate(name, b, without_budget=False,
                               budget_s=BUDGET_BRUSH_S):
    """Decides BEFORE starting the prefilter: if the estimated cost exceeds
    the budget, marks TOO_BIG immediately -- does NOT consume the 10
    minutes. The estimate is written in ALL cases (even brushes that pass),
    to sort a catch-up pass by increasing cost.

    `without_budget=True` (explicit catch-up pass, `--no-budget`): ignores
    both the estimate AND the 10-minute budget, DIRECT execution (no
    sub-process). Reserved for a deliberate, manual pass over brushes already
    marked TOO_BIG/BUDGET_EXCEEDED -- never the default behavior."""
    cm, det, mask = get_map(name)
    mn, mx, ct, extra = cm.brushes[b]
    rec = leafs_overlapping(cm, mn, mx)
    orphan_leafs = [li for li in rec if b not in cm._lb[li]]
    estimated_cost, n_cols, cols = _estimated_cost(cm, mn, mx, orphan_leafs)

    if cols is None or (not without_budget and estimated_cost > TOO_BIG_ESTIMATE_S):
        return {'map': name, 'brush_id': b, 'spot': 'TOO_BIG',
                'run_delta0_max': 'TOO_BIG',
                'n_domain_columns': '' if n_cols is None else n_cols,
                'n_survivors': '', 't_prefilter': '', 't_sweep': '',
                'x_col': '', 'y_col': '', 'z_col': '',
                'estimated_cost': estimated_cost, 'brush_kind': kind_of(cm, b)}

    if without_budget:
        q = _FileFakeQueue()
        _spot_exists_isolated(name, b, q)
        row = q.item
    else:
        row = _spot_exists_within_budget(name, b, budget_s)
        if row is None:
            return {'map': name, 'brush_id': b, 'spot': 'BUDGET_EXCEEDED',
                    'run_delta0_max': 'BUDGET_EXCEEDED', 'n_domain_columns': '',
                    'n_survivors': '', 't_prefilter': '', 't_sweep': '',
                    'x_col': '', 'y_col': '', 'z_col': '',
                    'estimated_cost': estimated_cost, 'brush_kind': kind_of(cm, b)}
    row['estimated_cost'] = estimated_cost
    row['brush_kind'] = kind_of(cm, b)
    return row


def _worker_production(q_taches, q_results, without_budget=False,
                       budget_s=BUDGET_BRUSH_S):
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    while True:
        item = q_taches.get()
        if item is None:
            break
        name, b = item
        row = _spot_exists_with_estimate(name, b, without_budget=without_budget,
                                         budget_s=budget_s)
        q_results.put(row)


PRODUCTION_JOURNAL_PATH = os.path.join(SCRATCH, 'production_journal.txt')

CATEGORIES_A_REJOUER = ('BUDGET_EXCEEDED', 'TOO_BIG')


def _brushes_already_done(name, exclude_categories=()):
    """(map, brush_id) already present in production_<map>.csv -- the basis
    for resuming. A brush already written (spot, BUDGET_EXCEEDED, whatever)
    must never be redone -- EXCEPT if it is in `exclude_categories` (the
    `--no-budget` catch-up pass: BUDGET_EXCEEDED/TOO_BIG should precisely
    be resumed, not skipped)."""
    path = os.path.join(SCRATCH, 'production_%s.csv' % name)
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path, newline='') as fh:
        for row in csv.DictReader(fh):
            if row.get('map') == name and row.get('spot') not in exclude_categories:
                out.add(int(row['brush_id']))
    return out


def _purge_categories(name, categories):
    """Removes `name`'s rows from the CSV whose `spot` is in `categories`
    (catch-up pass: these brushes are about to be recomputed and rewritten,
    the old stale row must not be kept as a duplicate)."""
    path = os.path.join(SCRATCH, 'production_%s.csv' % name)
    if not os.path.exists(path):
        return
    with open(path, newline='') as fh:
        rows = list(csv.DictReader(fh))
    kept = [r for r in rows if not (r.get('map') == name and r.get('spot') in categories)]
    if len(kept) == len(rows):
        return
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=PRODUCTION_COLUMNS)
        w.writeheader()
        for r in kept:
            w.writerow({k: r.get(k, '') for k in PRODUCTION_COLUMNS})



def migrate_csv_columns(path):
    """Bring an existing results CSV up to the current column set.

    The results file is opened in APPEND mode and its header is only written
    when the file does not exist. So a CSV created before a column was added
    keeps its old, shorter header while newly appended rows carry the extra
    values, and `csv.DictReader` silently drops them. That is exactly how
    `brush_kind` was lost on moscow: a 12-column header, 1050 rows of 12
    fields and 112 rows of 13.

    Rewrites the file in place with the current header, keeping every value:
    fields beyond the old header map, in order, to the columns that were added
    since. Rows shorter than the header are padded. No-op when the header
    already matches. Returns True when the file was rewritten.
    """
    if not os.path.exists(path):
        return False
    with open(path, newline='') as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return False
    old = rows[0]
    if old == PRODUCTION_COLUMNS:
        return False
    added = [c for c in PRODUCTION_COLUMNS if c not in old]
    out = []
    for r in rows[1:]:
        d = dict(zip(old, r))
        for i, v in enumerate(r[len(old):]):
            if i < len(added):
                d[added[i]] = v
        out.append({k: d.get(k, '') for k in PRODUCTION_COLUMNS})
    tmp = path + '.migrating'
    with open(tmp, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=PRODUCTION_COLUMNS)
        w.writeheader()
        w.writerows(out)
    os.replace(tmp, path)          # atomic: never leaves a half-written CSV
    return True


# --------------------------------------------------------------- progress bar
#
# Display only: no scan logic depends on any of this. Two lines redrawn in
# place -- the bar, and live counters underneath -- replacing the old "every
# 200 brushes" print and the per-brush event lines, without scrolling anything
# away. TOO_BIG / BUDGET_EXCEEDED no longer print one line each: they are
# counted live and detailed in the final recap.
#
# It is disabled when stdout is not a terminal (output redirected to a file, or
# piped): a stream of '\r' frames would bloat the file and hide the events. In
# that case the old periodic line comes back, unchanged.
BAR_ENABLED = sys.stdout.isatty()
BAR_WIDTH = 28


def _enable_vt():
    """Windows consoles need ENABLE_VIRTUAL_TERMINAL_PROCESSING before they
    honour ANSI escapes. Windows Terminal has it on already, the legacy
    conhost does not. Returns False if it cannot be turned on, and the bar
    then falls back to a single line -- never to garbage escape codes."""
    if os.name != 'nt':
        return True
    try:
        import ctypes
        k = ctypes.windll.kernel32
        h = k.GetStdHandle(-11)                 # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not k.GetConsoleMode(h, ctypes.byref(mode)):
            return False
        return bool(k.SetConsoleMode(h, mode.value | 0x0004))
    except Exception:
        return False


# Two-line display (bar + live counters) needs cursor-up, hence ANSI.
BAR_TWO_LINE = BAR_ENABLED and _enable_vt()
_BAR_ON = False          # a bar is currently drawn on screen


def _bar_draw(n_done, total, t0, peak_mem, counts=None):
    """Redraw in place:

        [####------] 37/204  18%  3.08 brush/s  eta 0:54  mem 154Mo
        Spots : 3   Divergence : 2   Skipped : 1

    No-op when not on a terminal. Falls back to one line when ANSI is
    unavailable (counters appended to the bar instead)."""
    global _BAR_ON
    if not BAR_ENABLED or not total:
        return
    import time
    dt = max(time.time() - t0, 1e-9)
    rate = n_done / dt
    frac = float(n_done) / total
    filled = int(BAR_WIDTH * frac)
    eta = (total - n_done) / rate if rate > 0 else 0.0
    bar = ("  [%s%s] %d/%d  %3.0f%%  %.2f brush/s  eta %d:%02d  mem %dMo"
           % ('#' * filled, '-' * (BAR_WIDTH - filled), n_done, total,
              frac * 100.0, rate, int(eta) // 60, int(eta) % 60, peak_mem))
    n_spot, n_div, n_skip = counts or (0, 0, 0)
    line2 = ("  Spots : %d   Divergence : %d   Skipped : %d"
             % (n_spot, n_div, n_skip))
    if BAR_TWO_LINE:
        sys.stdout.write('\r\x1b[K' + bar + '\n\x1b[K' + line2 + '\x1b[1A\r')
    else:
        sys.stdout.write('\r' + bar + line2.replace('  Spots', '  |  Spots'))
    sys.stdout.flush()
    _BAR_ON = True


def _bar_clear():
    """Wipe the live display, leaving the cursor where it can print freely."""
    global _BAR_ON
    if not _BAR_ON:
        return
    if BAR_TWO_LINE:
        sys.stdout.write('\r\x1b[K\n\x1b[K\x1b[1A\r')
    else:
        sys.stdout.write('\r' + ' ' * 200 + '\r')
    sys.stdout.flush()
    _BAR_ON = False


def _bar_log(msg):
    """Print an event line above the live display, without fragments."""
    _bar_clear()
    print(msg)


def _bar_close():
    """Leave the finished display on screen and move below it."""
    global _BAR_ON
    if _BAR_ON:
        sys.stdout.write('\n\n' if BAR_TWO_LINE else '\n')
        sys.stdout.flush()
    _BAR_ON = False


def cmd_map_cost(name, jobs=DEFAULT_WORKERS, without_budget=False,
                 kind='non-axial', budget_s=BUDGET_BRUSH_S):
    """PRODUCTION pipeline over ONE COMPLETE map (all its orphan brushes, not
    a sample). RESUMES, NEVER REDOES: a crash or abort only costs whatever
    is not yet written to the CSV, never the whole map. Memory ceiling
    monitored, journal written as it goes (`production_journal.txt`), a
    10-min budget per brush (`_worker_production` -> `_spot_exists_within_budget`,
    already implemented).

    `jobs`: number of workers (default DEFAULT_WORKERS=3, settable via
    `production.py --jobs` -- keep a CPU margin, do not go up to
    cpu_count()-1).

    `without_budget=True`: explicit catch-up pass -- resumes precisely the
    brushes marked TOO_BIG/BUDGET_EXCEEDED (excluded from a normal resume),
    with no time budget or pre-estimation. First purges their stale rows
    from the CSV to avoid duplicates.

    `kind`: which brushes to consider, see `orphans_of`. Resuming makes the
    passes composable: running a map with 'non-axial' then again with 'axial'
    measures each brush exactly once.

    `budget_s`: runtime budget per brush, see BUDGET_BRUSH_S. Raise it to
    revisit brushes the default cuts off; a brush already written as
    BUDGET_EXCEEDED is skipped by the resume, so use `--no-budget` to actually
    retry those."""
    import multiprocessing as mp
    import time
    import queue as _queue
    import memwatch

    brushes_all = orphans_of(name, kind)
    exclude = CATEGORIES_A_REJOUER if without_budget else ()
    already_done = _brushes_already_done(name, exclude_categories=exclude)
    remaining = [b for b in brushes_all if b not in already_done]
    print("%s : %d/%d orphan brushes to process (%d already done, resuming) -- %d workers%s"
          % (name, len(remaining), len(brushes_all), len(already_done), jobs,
             "  [NO BUDGET]" if without_budget else ""))
    if not remaining:
        print("nothing to do.")
        return 0

    os.makedirs(SCRATCH, exist_ok=True)
    if without_budget:
        _purge_categories(name, CATEGORIES_A_REJOUER)
    out_path = os.path.join(SCRATCH, 'production_%s.csv' % name)
    if migrate_csv_columns(out_path):
        print("  (CSV migrated to the current columns: %s)" % os.path.basename(out_path))
    write_header = not os.path.exists(out_path)

    ctx = mp.get_context('spawn')
    q_taches = ctx.Queue()
    q_results = ctx.Queue()
    for b in remaining:
        q_taches.put((name, b))
    for _ in range(jobs):
        q_taches.put(None)

    workers = [ctx.Process(target=_worker_production,
                           args=(q_taches, q_results, without_budget, budget_s))
               for _ in range(jobs)]
    for w in workers:
        w.start()

    # Watch OUR processes only. Summing every python.exe on the machine (what
    # memwatch did before) counts whatever else the user has open, so the
    # ceiling could fire on memory the scan never allocated -- and the
    # peak_mem written to the journal was inflated by the same amount.
    watched = [os.getpid()] + [w.pid for w in workers if w.pid]

    fh = open(out_path, 'a', newline='')
    w_csv = csv.DictWriter(fh, fieldnames=PRODUCTION_COLUMNS)
    if write_header:
        w_csv.writeheader()
        fh.flush()

    t0 = time.time()
    n_done = 0
    total_prefilter = 0.0
    total_sweep = 0.0
    domains = []
    n_survivors_total = 0
    n_spots = 0
    n_budget_exceeded = 0
    n_too_big = 0
    n_divergence = 0
    peak_mem = 0
    memory_abandoned = False
    try:
        while n_done < len(remaining):
            try:
                row = q_results.get(timeout=PERIODE_MEM)
            except _queue.Empty:
                m = memwatch.memory_mb(watched)
                peak_mem = max(peak_mem, m)
                # Refresh here too: with a per-brush budget the queue can stay
                # silent for seconds, and a frozen bar reads like a hang.
                _bar_draw(n_done, len(remaining), t0, peak_mem,
                          (n_spots, n_divergence, n_too_big + n_budget_exceeded))
                if m > MEMORY_CEILING_MB:
                    _bar_log("  !! MEMORY CEILING EXCEEDED (%d Mo > %d) -- stopping, rerun will resume"
                             % (m, MEMORY_CEILING_MB))
                    memory_abandoned = True
                    break
                continue
            w_csv.writerow(row)
            fh.flush()
            n_done += 1
            if row['spot'] == 'TOO_BIG':
                # No per-brush line for TOO_BIG / BUDGET_EXCEEDED: they are
                # counted live under the bar and detailed in the final recap
                # (`production.py budget` lists them individually).
                n_too_big += 1
            elif row['spot'] == 'BUDGET_EXCEEDED':
                n_budget_exceeded += 1
            else:
                total_prefilter += row['t_prefilter']
                total_sweep += row['t_sweep']
                domains.append(row['n_domain_columns'])
                if row['n_survivors'] >= 1:
                    n_survivors_total += 1
                if row['spot'] == 'DIVERGENCE':
                    # Both are shown live as counters under the bar, and listed
                    # in full (ids and coordinates) in the final recap.
                    n_divergence += 1
                elif row['spot'] is True:
                    n_spots += 1
            _bar_draw(n_done, len(remaining), t0, peak_mem,
                      (n_spots, n_divergence, n_too_big + n_budget_exceeded))
            if not BAR_ENABLED and (n_done % 200 == 0 or n_done == len(remaining)):
                dt = time.time() - t0
                print("  %d/%d  (%.0fs elapsed, %.2f brush/s, peak memory %dMo)"
                      % (n_done, len(remaining), dt, n_done / max(dt, 1e-9), peak_mem))
    finally:
        _bar_close()
        fh.close()
        for w in workers:
            if w.is_alive():
                w.terminate()
            w.join(timeout=5)
        while True:
            try:
                q_taches.get_nowait()
            except Exception:
                break

    dt_total = time.time() - t0
    if memory_abandoned:
        with open(PRODUCTION_JOURNAL_PATH, 'a') as jfh:
            jfh.write("%-16s MEMORY_ABANDONED after %d/%d brushes in %.0fs peak_mem=%dMo\n"
                      % (name, n_done, len(remaining), dt_total, peak_mem))
        print("\nabandoned for memory ceiling: %d/%d brushes done -> %s (rerunning `map-cost %s` will resume)"
              % (n_done, len(remaining), out_path, name))
        return 1

    domains.sort()
    n = len(domains)
    median = domains[n // 2] if n else 0
    average = sum(domains) / n if n else 0

    with open(PRODUCTION_JOURNAL_PATH, 'a') as jfh:
        jfh.write("%-16s OK %d/%d brushes (this round) in %.0fs "
                  "(%d too_big, %d budget_exceeded, %d divergence, %d spots, peak_mem=%dMo)\n"
                  % (name, n_done, len(remaining), dt_total,
                     n_too_big, n_budget_exceeded, n_divergence, n_spots, peak_mem))

    print()
    print("=== %s: production pipeline cost ===" % name)
    print("brushes processed this round : %d/%d remaining (%d already done before, %d total on the map)"
          % (n_done, len(remaining), len(already_done), len(brushes_all)))
    print("duration (wall, %d workers) : %.0fs (%.1f min)" % (jobs, dt_total, dt_total / 60))
    print("CUMULATIVE prefilter time (step 1, this round) : %.0fs (%.1f min)" % (total_prefilter, total_prefilter / 60))
    print("CUMULATIVE full-scan time (step 2, this round) : %.0fs (%.1f min)" % (total_sweep, total_sweep / 60))
    print("brushes kept by the prefilter (n_survivors>=1) : %d/%d (%.1f%%)"
          % (n_survivors_total, n_done, 100.0 * n_survivors_total / max(1, n_done)))
    print("domain size (columns) : median=%d  mean=%.0f  min=%d  max=%d"
          % (median, average, domains[0] if n else 0, domains[-1] if n else 0))
    print()
    print("VERDICT %s (this round) : %d spots, %d DIVERGENCE, %d BUDGET_EXCEEDED, %d TOO_BIG"
          % (name, n_spots, n_divergence, n_budget_exceeded, n_too_big))
    print("peak memory : %dMo" % peak_mem)
    print("\nwritten -> %s (rerunning `map-cost %s` is safe, it will skip what's already done)"
          % (out_path, name))
    return 0


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('map-cost',):
        print(__doc__)
        return 2
    cmd = sys.argv[1]
    if cmd == 'map-cost':
        if len(sys.argv) < 3:
            print("usage: characterize.py map-cost <map> [--jobs N] [--no-budget]"
                  " [--budget SECONDS] [--brushes non-axial|axial|both]"
                  " [--maps-dir d]")
            return 2
        jobs = DEFAULT_WORKERS
        if '--jobs' in sys.argv:
            jobs = int(sys.argv[sys.argv.index('--jobs') + 1])
        if '--maps-dir' in sys.argv:
            cm_leafs.set_maps_dir(sys.argv[sys.argv.index('--maps-dir') + 1])
        budget_s = BUDGET_BRUSH_S
        if '--budget' in sys.argv:
            budget_s = float(sys.argv[sys.argv.index('--budget') + 1])
            if budget_s <= 0:
                print("--budget must be a positive number of seconds")
                return 2
        kind = 'non-axial'
        if '--brushes' in sys.argv:
            kind = sys.argv[sys.argv.index('--brushes') + 1]
            if kind not in BRUSH_KINDS:
                print("--brushes must be one of %s" % (BRUSH_KINDS,))
                return 2
        without_budget = '--no-budget' in sys.argv
        return cmd_map_cost(sys.argv[2], jobs=jobs, without_budget=without_budget,
                            kind=kind, budget_s=budget_s)


if __name__ == '__main__':
    try:
        sys.exit(main() or 0)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        sys.exit(130)
    except cm_leafs.MapLoadError as e:
        print("error: %s" % e, file=sys.stderr)
        sys.exit(1)
