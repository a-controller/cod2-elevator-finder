"""BRUSH-level characterization of the elevator's geometric conjunction.

    python characterize.py map-cost <map> [--jobs N] [--no-budget]

Imports `cm_leafs.py`, `cm_faithful.py`, `detect.py` READ-ONLY, does not
modify them.

`vertical_extent_asym` (condition b alone, on a single isolated brush) is
not sufficient by itself and does not always even point at the right column
(#6453: extent 217, higher than the 4 known positives, yet no point of ITS
winning column escapes via index 0 -- verified with `detect.py`, an
independent instrument, on the column AND on the 6 relevant levels: a known
1-step artifact, not a spot).

New hierarchy:
  vertical_extent_asym  PREFILTER / search domain (b is necessary for
                           index 0 to escape -- a cheap superset).
  run_delta0_max           THE PREDICATE. Longest CONSECUTIVE run of z (step
                           DZ) where PM_CorrectAllSolid escapes precisely via
                           index 0 ({0,0,1}, tested directly, not all 26 --
                           exact per the source: tried first, taken if it
                           escapes). This is (a)^(b)^(c)^(d) directly,
                           comparable term for term to ground truth's `step`
                           -- not a proxy. Optimized by the SAME machinery
                           (grid+edges then iterative local walk), over the
                           SAME domain of columns as vertical_extent_asym,
                           but with its own objective: the two legitimately
                           peak at different columns.
"""
import csv
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

def condition_b(cm, b, planes, x, y, z, maxs):
    """Asymmetry EVALUATED AT POINT (x, y, z): the point is geometrically
    inside brush b (test_box_in_brush, exact point path), BUT b does not
    appear in the leafs the point probe selects at that point
    (brushes_for_point, cm_leafs.py:280-292)."""
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
            return False        # listed somewhere: no asymmetry here
    return True


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


def _domain_columns(cm, mn, mx, orphan_leafs, maxs):
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
        # edges, step STEP_GRID along the 4 borders
        x = ox0
        while x <= ox1:
            cols.add((x, oy0)); cols.add((x, oy1))
            x += STEP_GRID
        y = oy0
        while y <= oy1:
            cols.add((ox0, y)); cols.add((ox1, y))
            y += STEP_GRID
        # inner grid
        x = ox0
        while x <= ox1:
            y = oy0
            while y <= oy1:
                cols.add((x, y))
                y += STEP_GRID
            x += STEP_GRID
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
        fin = set()
        xr = bx - REFINE_RADIUS
        while xr <= bx + REFINE_RADIUS:
            yr = by - REFINE_RADIUS
            while yr <= by + REFINE_RADIUS:
                fin.add((xr, yr))
                yr += 1.0
            xr += 1.0
        fin -= tested
        if not fin:
            break
        nouveau_col = None
        for (x, y) in fin:
            r = evaluate(x, y)
            if r[0] > best[0]:
                best, nouveau_col = r, (x, y)
        tested |= fin
        n_tested += len(fin)
        if nouveau_col is None:
            break            # no improvement: at a local peak
        best_col = nouveau_col

    return best, best_col, n_tested


def _delta0_escapes(cm, b, planes, det, mask, x, y, z, maxs):
    """Tests ONLY index 0 ({0,0,1}) of PM_CorrectAllSolid, not all 26 --
    exact, not an approximation (per the source: {0,0,1} is tried first,
    escapes -> taken, regardless of the other 25).

    Prefilters via `condition_b` AT THE ORIGIN (x, y, z) BEFORE the costly
    `ground_allsolid` gate: (b) is a NECESSARY precondition for any step of
    a climb attributable to THIS brush -- no attributable run can leave its
    (b) region, so testing (b), cheaper (`box_leafnums` alone), before
    paying for a full swept trace, loses no true positive."""
    if not condition_b(cm, b, planes, x, y, z, maxs):
        return False
    pos = (x, y, z)
    if not det.ground_allsolid(pos, maxs):
        return False
    dx, dy, dz = CORRECT_SOLID_DELTAS[0]
    point = (x + dx, y + dy, z + dz)
    nums = cm.brushes_for_point(point, PLAYER_MINS, maxs, mask)
    tr = player_trace(point, point, PLAYER_MINS, maxs, cm.planes_of(nums))
    return not tr['startsolid']


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
    step_filter = detect.MIN_STEP * DZ
    z = zlo
    while z <= zhi:
        if _delta0_escapes(cm, b, planes, det, mask, x, y, z, maxs):
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
        r = _run_delta0_at_column(cm, b, planes, det, mask, x, y, zlo, zhi, maxs)
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


# --------------------------------------------------------------- orphans (neg)

def orphans_of(name):
    """List of a map's orphan brush_ids: mask + at least one non-axial plane
    + orphan (at least one leaf that overlaps it without listing it)."""
    cm, det, mask = get_map(name)
    out = []
    for b, (mn, mx, ct, pl) in enumerate(cm.brushes):
        if not (ct & mask) or not pl:
            continue
        rec = leafs_overlapping(cm, mn, mx)
        if any(b not in cm._lb[li] for li in rec):
            out.append(b)
    return out


DEFAULT_WORKERS = 3      # explicitly capped (not cpu_count()-1): the scan
                        # already uses 25-30% CPU, keep a margin

MEMORY_CEILING_MB = 2500                        # memory ceiling per child process
PERIODE_MEM = 5.0


PRODUCTION_COLUMNS = ['map', 'brush_id', 'spot', 'run_delta0_max',
                        'n_domain_columns', 'n_survivors',
                        't_prefilter', 't_sweep', 'x_col', 'y_col', 'z_col',
                        'estimated_cost']

# Prefilter cost/test (mod-10 filter), calibrated on real production
# measurements (t_prefilter, n_domain_columns, height_z). n_tests =
# n_cols * (height_z / 10), EXACTLY the number of iterations of
# `_column_has_potential` (step_filter = MIN_STEP*DZ = 10). Median of the
# individual t/n_tests ratios (robust to the rare brush with an extreme
# domain): 0.0000242285 s/test -- very close to the total/total ratio
# (0.0000233070), the two converge.
COST_PER_TEST_S = 0.0000242285


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
            r = _run_delta0_at_column(cm, b, planes, det, mask, x, y, zlo, zhi, PLAYER_MAXS_STANDING)
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


BUDGET_BRUSH_S = 10 * 60      # hard budget per brush: some brushes with a huge
                                # domain can block a worker indefinitely


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
    avoid rebuilding the domain twice."""
    cols = _domain_columns(cm, mn, mx, orphan_leafs, maxs)
    zlo = mn[2] - maxs[2]
    zhi = mx[2] - PLAYER_MINS[2]
    height_z = zhi - zlo
    n_tests = len(cols) * (height_z / 10.0)
    return n_tests * COST_PER_TEST_S, len(cols), cols


def _spot_exists_with_estimate(name, b, without_budget=False):
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

    if not without_budget and estimated_cost > BUDGET_BRUSH_S:
        return {'map': name, 'brush_id': b, 'spot': 'TOO_BIG',
                'run_delta0_max': 'TOO_BIG', 'n_domain_columns': n_cols,
                'n_survivors': '', 't_prefilter': '', 't_sweep': '',
                'x_col': '', 'y_col': '', 'z_col': '', 'estimated_cost': estimated_cost}

    if without_budget:
        q = _FileFakeQueue()
        _spot_exists_isolated(name, b, q)
        row = q.item
    else:
        row = _spot_exists_within_budget(name, b)
        if row is None:
            return {'map': name, 'brush_id': b, 'spot': 'BUDGET_EXCEEDED',
                    'run_delta0_max': 'BUDGET_EXCEEDED', 'n_domain_columns': '',
                    'n_survivors': '', 't_prefilter': '', 't_sweep': '',
                    'x_col': '', 'y_col': '', 'z_col': '', 'estimated_cost': estimated_cost}
    row['estimated_cost'] = estimated_cost
    return row


def _worker_production(q_taches, q_resultats, without_budget=False):
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    while True:
        item = q_taches.get()
        if item is None:
            break
        name, b = item
        row = _spot_exists_with_estimate(name, b, without_budget=without_budget)
        q_resultats.put(row)


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


def cmd_map_cost(name, jobs=DEFAULT_WORKERS, without_budget=False):
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
    from the CSV to avoid duplicates."""
    import multiprocessing as mp
    import time
    import queue as _queue
    import memwatch

    brushes_all = orphans_of(name)
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
    ecrire_entete = not os.path.exists(out_path)

    ctx = mp.get_context('spawn')
    q_taches = ctx.Queue()
    q_resultats = ctx.Queue()
    for b in remaining:
        q_taches.put((name, b))
    for _ in range(jobs):
        q_taches.put(None)

    workers = [ctx.Process(target=_worker_production, args=(q_taches, q_resultats, without_budget))
               for _ in range(jobs)]
    for w in workers:
        w.start()

    fh = open(out_path, 'a', newline='')
    w_csv = csv.DictWriter(fh, fieldnames=PRODUCTION_COLUMNS)
    if ecrire_entete:
        w_csv.writeheader()
        fh.flush()

    t0 = time.time()
    n_done = 0
    tot_prefilter = 0.0
    tot_sweep = 0.0
    domains = []
    n_survivors_tot = 0
    n_spots = 0
    n_budget_exceeded = 0
    n_too_big = 0
    n_divergence = 0
    peak_mem = 0
    memory_abandoned = False
    try:
        while n_done < len(remaining):
            try:
                row = q_resultats.get(timeout=PERIODE_MEM)
            except _queue.Empty:
                m = memwatch.memory_mo(None)
                peak_mem = max(peak_mem, m)
                if m > MEMORY_CEILING_MB:
                    print("  !! MEMORY CEILING EXCEEDED (%d Mo > %d) -- stopping, rerun will resume"
                          % (m, MEMORY_CEILING_MB))
                    memory_abandoned = True
                    break
                continue
            w_csv.writerow(row)
            fh.flush()
            n_done += 1
            if row['spot'] == 'TOO_BIG':
                n_too_big += 1
                print("  !! TOO BIG: %s #%s  estimated_cost=%.0fs (>%ds) -- skipped before even prefiltering"
                      % (row['map'], row['brush_id'], row['estimated_cost'], BUDGET_BRUSH_S))
            elif row['spot'] == 'BUDGET_EXCEEDED':
                n_budget_exceeded += 1
                print("  !! BUDGET EXCEEDED: %s #%s (>%d min) -- marked, not measured"
                      % (row['map'], row['brush_id'], BUDGET_BRUSH_S // 60))
            else:
                tot_prefilter += row['t_prefilter']
                tot_sweep += row['t_sweep']
                domains.append(row['n_domain_columns'])
                if row['n_survivors'] >= 1:
                    n_survivors_tot += 1
                if row['spot'] == 'DIVERGENCE':
                    n_divergence += 1
                    print("  !! DIVERGENCE: %s #%s  run_delta0_max=%s  (%.0f, %.0f) --"
                          " the pipeline says >=MIN_STEP but detect.py check() says no. NEITHER spot NOR negative."
                          % (row['map'], row['brush_id'], row['run_delta0_max'],
                             row['x_col'], row['y_col']))
                elif row['spot'] is True:
                    n_spots += 1
                    print("  ** SPOT (confirmed by detect.py): %s #%s  run_delta0_max=%s  (%.0f, %.0f)"
                          % (row['map'], row['brush_id'], row['run_delta0_max'],
                             row['x_col'], row['y_col']))
            if n_done % 200 == 0 or n_done == len(remaining):
                dt = time.time() - t0
                print("  %d/%d  (%.0fs elapsed, %.2f brush/s, peak memory %dMo)"
                      % (n_done, len(remaining), dt, n_done / max(dt, 1e-9), peak_mem))
    finally:
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
    print("CUMULATIVE prefilter time (step 1, this round) : %.0fs (%.1f min)" % (tot_prefilter, tot_prefilter / 60))
    print("CUMULATIVE full-scan time (step 2, this round) : %.0fs (%.1f min)" % (tot_sweep, tot_sweep / 60))
    print("brushes kept by the prefilter (n_survivors>=1) : %d/%d (%.1f%%)"
          % (n_survivors_tot, n_done, 100.0 * n_survivors_tot / max(1, n_done)))
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
            print("usage: characterize.py map-cost <map> [--jobs N] [--no-budget] [--maps-dir d]")
            return 2
        jobs = DEFAULT_WORKERS
        if '--jobs' in sys.argv:
            jobs = int(sys.argv[sys.argv.index('--jobs') + 1])
        if '--maps-dir' in sys.argv:
            cm_leafs.set_maps_dir(sys.argv[sys.argv.index('--maps-dir') + 1])
        without_budget = '--no-budget' in sys.argv
        return cmd_map_cost(sys.argv[2], jobs=jobs, without_budget=without_budget)


if __name__ == '__main__':
    try:
        sys.exit(main() or 0)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        sys.exit(130)
    except cm_leafs.MapLoadError as e:
        print("error: %s" % e, file=sys.stderr)
        sys.exit(1)
