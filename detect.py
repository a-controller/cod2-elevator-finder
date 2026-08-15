"""Detector for "elevator" spots in Call of Duty 2.

    python detect.py selftest [--maps-dir d]
    python detect.py check <map> <x> <y> <z> [--maps-dir d]
    python detect.py scan  <map> [--step 8] [--min-step 10] [--mask 0x...] [--out f] [--maps-dir d]

`<map>` = short name (mp_farmhouse) or path to a dump produced by
`dump_clipmap.lua`. `--maps-dir`: dumps/`.d3dbsp` directory, else the
`COD2_MAPS_DIR` environment variable, else `maps/` next to this file.

--------------------------------------------------------------------------
THE MECHANISM (measured with a debugger, not inferred)

The glitch comes from an ASYMMETRY in brush selection between the engine's
two trace types:

  SWEPT trace (start != end)       CM_TraceThroughTree
      offset = size[axis] + 0.125, and 2048 on non-axial planes
      -> very permissive, sees many brushes

  POINT trace (start == end)       CM_PositionTest -> CM_BoxLeafnums_r
      EXACT box test on start +- size +- 1
      -> sees far fewer brushes

When standing up under a bevel, the capsule expansion goes from 10 to 20 in
Z and the player enters the brush. PM_GroundTrace (swept) sees it -> allsolid
-> calls PM_CorrectAllSolid, which tests 26 deltas via POINT traces. The
brush is NOT submitted to those: the first delta {0,0,1} passes, the player
rises by +1.000, and it repeats every frame.

--------------------------------------------------------------------------
THE CRITERION

A position is a spot if:
  1. standing : PM_GroundTrace gives allsolid
  2. crouched : PM_GroundTrace does NOT give allsolid (the state is born from standing up)
  3. PM_CorrectAllSolid picks a delta with vertical gain > 0
  4. and above all: the climb LOOPS (>= MIN_STEP iterations)

Point 4 is essential. Without it, `mp_farmhouse` yields 69 positions, 61 of
which rise by only ONE unit before falling back (verified in-game: the
engine really does make 1 call / 1 success, exactly as the model predicts).
With it, exactly the 2 known spots come out.
"""
import sys
import os
import time

from cm_faithful import (PLAYER_MINS, PLAYER_MAXS_STANDING, PLAYER_MAXS_CROUCH,
                         PLAYER_MAXS_PRONE, CORRECT_SOLID_DELTAS, player_trace)
import cm_leafs
from cm_leafs import ClipMap

# pm->tracemask, read from memory (pm+0x3C) on the MULTIPLAYER client.
# On another binary (single-player) it can differ: dump_clipmap.lua prints it
# on every dump. If it isn't 0x02810011, pass --mask <value> to scan.
MASK = 0x02810011
MIN_STEP = 10             # a real climb makes ~50 steps; an artifact makes 1
REST = 0.125             # the player rests 0.125 above surfaces
STEP_GRID = 8.0
MAX_SPAN = 3000.0        # ignore skybox and giant brushes
# `RELAX` removes the "crouched not allsolid" condition. This condition is
# NOT a requirement of the engine: it is an accessibility heuristic, there to
# guarantee that the state is born from standing up. One can also enter via
# teleport or noclip (hill400_assault). A spot where BOTH standing and
# crouched are allsolid but an upward push still exists would be a valid
# spot, rejected today.
# Measured cost: ~6x slower (climb() runs on 11-48% of positions instead of
# 0.2-1.3%). Only use on a bounded perimeter.
RELAX = False

# POSTURE. The glitch is born from an ENLARGEMENT of the box: the big one
# gives `allsolid`, the small one does not, and standing up crosses from one
# to the other. The detector long knew only one transition, crouch -> stand.
# There is a second one, prone -> crouch, invisible by construction: under an
# overhang 30-50 units off the ground, the player CANNOT stand (a box of 70
# is deeply stuck inside, all 26 deltas fail, it is jammed), so no sampling
# with the standing box can ever find it. This is not a question of position
# but of box size.
POSTURES = {'standing':   (PLAYER_MAXS_STANDING, PLAYER_MAXS_CROUCH),
            'crouching': (PLAYER_MAXS_CROUCH, PLAYER_MAXS_PRONE)}
POSTURE = 'standing'


def boxes():
    """(big, small) for the current posture."""
    return POSTURES[POSTURE]


class Detector(object):
    def __init__(self, cm):
        self.cm = cm.build_cache(MASK)

    # ---------------------------------------------------------- primitives

    def ground_allsolid(self, origin, maxs):
        """PM_GroundTrace: sweep origin+0.25 -> origin-0.25."""
        cm = self.cm
        start = (origin[0], origin[1], origin[2] + 0.25)
        end = (origin[0], origin[1], origin[2] - 0.25)
        nums = cm.brushes_for_sweep(start, end, PLAYER_MINS, maxs, MASK)
        return player_trace(start, end, PLAYER_MINS, maxs,
                            cm.planes_of(nums))['allsolid']

    def correct_all_solid(self, origin, maxs):
        """PM_CorrectAllSolid: 26 deltas tested via POINT traces."""
        cm = self.cm
        for idx, (dx, dy, dz) in enumerate(CORRECT_SOLID_DELTAS):
            point = (origin[0] + dx, origin[1] + dy, origin[2] + dz)
            nums = cm.brushes_for_point(point, PLAYER_MINS, maxs, MASK)
            tr = player_trace(point, point, PLAYER_MINS, maxs, cm.planes_of(nums))
            if not tr['startsolid']:
                return idx, point
        return None, origin

    # ------------------------------------------------------------ critere

    def climb(self, origin, limit=400, already_solid=False):
        """Simulates the engine's loop. Returns (n_steps, final_z, delta0).

        `already_solid`: the caller has just checked `ground_allsolid` on
        `origin` and avoids redoing it on the first round. The scan used to
        test the same swept trace twice in a row on every position reaching
        `climb()` — 11% of positions in 'full' mode. No change in
        semantics: it is exactly the same test, simply not redone."""
        big = boxes()[0]
        pos, n, first = origin, 0, None
        skip = already_solid
        while n < limit:
            if skip:
                skip = False
            elif not self.ground_allsolid(pos, big):
                break
            idx, point = self.correct_all_solid(pos, big)
            if idx is None or point[2] <= pos[2]:
                break
            if first is None:
                first = idx
            pos, n = point, n + 1
        return n, pos[2], first

    def check(self, origin):
        """Complete diagnostic of a position."""
        big, small = boxes()
        standing = self.ground_allsolid(origin, big)
        crouching = self.ground_allsolid(origin, small)
        r = {'standing': standing, 'crouching': crouching, 'step': 0, 'z': origin[2],
             'delta': None, 'spot': False}
        if standing and (RELAX or not crouching):
            r['step'], r['z'], r['delta'] = self.climb(origin)
            r['spot'] = r['step'] >= MIN_STEP
        return r

    # ------------------------------------------------------------- balayage

    def candidates(self, step=STEP_GRID, mode='ground'):
        """Positions to test, XY grid over the bbox of each masked brush.

        mode 'ground' (default, historical, validated by the selftest)
            a single level: the top of the brush, `maxs.z + 0.125`. The
            player arrives walking.

        mode 'air'
            under each brush, in the crouch/stand window — i.e. `z` in
            `]mins.z - 70, mins.z - 50[` — regardless of what is underneath.
            5 levels spaced by 4.

            Why this mode exists: in 'ground' mode, a position is only tested if
            a brush's TOP happens to fall exactly in the 20-unit window under
            the overhang. That is a support requirement disguised as
            sampling, whereas the elevator demands none — `PM_GroundTrace`
            returns `allsolid` with no ground contact. A spot reachable by
            jumping, teleporting, or noclip was therefore invisible.
            `hill400_assault`'s 5 zones were only ever found by accident: the
            top of a rotated brush's BBOX happened to land in the window.

        Mode 'air' produces ~5x more positions than 'ground'. At that scale the
        LIST no longer fits in memory: use `iter_candidates()`, which
        materializes nothing. `candidates()` remains for 'ground' mode and
        diagnostic uses."""
        return list(self.iter_candidates(step, mode))

    def iter_candidates(self, step=STEP_GRID, mode='ground'):
        """Streamed version of `candidates()`. Bounded memory.

        Deduplication is only active in 'ground' mode: its `set` weighs as much
        as the list itself (~11M entries on a big map in 'air'), and in
        'air' the Z levels differ from one brush to another, so collisions
        are rare. The rare duplicates cost time, never a wrong result —
        `cmd_scan` removes them from the hit list, which itself is tiny."""
        big, small = boxes()
        h_big, h_small = big[2], small[2]
        seen = set() if mode == 'ground' else None
        for b, (mn, mx, ct, _pl) in enumerate(self.cm.brushes):
            if not (ct & MASK):
                continue
            if (mx[0] - mn[0]) > MAX_SPAN or (mx[1] - mn[1]) > MAX_SPAN:
                continue

            if mode == 'full':
                # COLUMN BY COLUMN. For each (x, y), `column_z` is asked for
                # the exact interval where the box can touch the brush, and
                # the one where it is entirely buried (hence stuck, never
                # triggering). Two EXACT prunings, proven, not heuristics:
                #   - columns outside the brush disappear entirely. This is
                #     what made `complet` explode: on sloped terrain, the
                #     bbox is nearly empty (hill400_assault: 1.03e9 positions
                #     sampled over the bbox).
                #   - the Z interval follows the slope instead of covering
                #     the bbox's full height.
                # The XY footprint is widened by 16 to cover LATERAL contact
                # (mp_farmhouse's spot1: the player is beside the brush, its
                # box juts into it).
                x = mn[0] - 16.0
                while x <= mx[0] + 16.0:
                    y = mn[1] - 16.0
                    while y <= mx[1] + 16.0:
                        iv = self.cm.column_z(b, x, y, h_big)
                        if iv is not None:
                            zlo, zhi, zin0, zin1 = iv
                            z = zlo
                            while z <= zhi:
                                if not (zin0 <= z <= zin1):
                                    yield (x, y, z)
                                z += 4.0
                        y += step
                    x += step
                continue

            if mode == 'air':
                zs = [mn[2] - h_big + 1.0 + 4.0 * k for k in range(5)]
            else:
                zs = [mx[2] + REST]
            for z in zs:
                x = mn[0]
                while x <= mx[0]:
                    y = mn[1]
                    while y <= mx[1]:
                        if seen is not None:
                            k = (int(x // step), int(y // step), int(z * 8))
                            if k in seen:
                                y += step
                                continue
                            seen.add(k)
                        yield (x, y, z)
                        y += step
                    x += step

    def count_candidates(self, step=STEP_GRID, mode='ground'):
        """Number of positions for 'air' mode, computed arithmetically —
        without storing anything. Used only to display progress."""
        if mode == 'full':      # column-by-column generation: just count
            return sum(1 for _ in self.iter_candidates(step, mode))
        n = 0
        for mn, mx, ct, _pl in self.cm.brushes:
            if not (ct & MASK):
                continue
            if (mx[0] - mn[0]) > MAX_SPAN or (mx[1] - mn[1]) > MAX_SPAN:
                continue
            nx = int((mx[0] - mn[0]) // step) + 1
            ny = int((mx[1] - mn[1]) // step) + 1
            niv = 5 if mode == 'air' else 1
            n += nx * ny * niv
        return n

    def scan(self, step=STEP_GRID, verbose=True, mode='ground'):
        pts = self.candidates(step, mode)
        if verbose:
            print("candidate positions: %d" % len(pts))
        t0 = time.time()
        hits = []
        big, small = boxes()
        for i, p in enumerate(pts):
            if not self.ground_allsolid(p, big):
                continue
            if not RELAX and self.ground_allsolid(p, small):
                continue
            n, z, d = self.climb(p, already_solid=True)
            if n >= MIN_STEP:
                hits.append((p, n, z, d))
            if verbose and (i + 1) % 50000 == 0:
                print("  %d/%d  (%.0fs, %d spots)"
                      % (i + 1, len(pts), time.time() - t0, len(hits)))
        if verbose:
            dt = time.time() - t0
            print("done in %.1fs  (%.3f ms/position)"
                  % (dt, 1000 * dt / max(1, len(pts))))
        return hits


# ------------------------------------------------------- parallel scan
#
# `scan()`'s loop body is PURE: read-only over immutable geometry, no state
# shared between positions. Splitting it across processes therefore changes
# no computation.
#
# TWO PITFALLS, both handled here:
#
#  1. HIT ORDER. `group()` takes the FIRST point of a zone as its center, and
#     that is the one printed as `setviewpos` in the reports. If hits came
#     back in block-completion order, the published coordinates would change
#     (same zones, different representative). So each position's INDEX is
#     carried along and sorted on afterward: strictly the same order as a
#     sequential scan.
#
#  2. CANDIDATE LIST. Each worker used to regenerate `candidates()` on its
#     own. Positions are now STREAMED in blocks from the parent (see
#     `scan_parallel`): no more per-worker regeneration, no full list held in
#     memory. In 'air' mode a big map produces ~11M positions — materializing
#     them in the parent AND in each worker easily exhausts available memory.

_W = {}
CHUNK = 20000                 # positions per block sent to a worker


def _worker_init(path, mask, min_step, relax, posture):
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global MASK, MIN_STEP, RELAX, POSTURE
    MASK, MIN_STEP, RELAX, POSTURE = mask, min_step, relax, posture
    from cm_leafs import ClipMap
    _W['d'] = Detector(ClipMap(path))


def _worker_chunk(chunk):
    d = _W['d']
    big, small = boxes()
    out = []
    for i, p in chunk:
        if not d.ground_allsolid(p, big):
            continue
        if not RELAX and d.ground_allsolid(p, small):
            continue
        n, z, dd = d.climb(p, already_solid=True)
        if n >= MIN_STEP:
            out.append((i, p, n, z, dd))
    return out


def _blocks(it, size):
    """Splits an iterator into lists of `size` elements."""
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def scan_parallel(path, source, jobs, verbose=True, total=None):
    """Same result as `Detector.scan`, spread over `jobs` processes.

    `source` is an ITERATOR of positions (not a list): the parent's memory
    stays bounded to the number of in-flight blocks, regardless of map size.
    Submission is capped at `2 * jobs` simultaneous blocks — `Pool.imap`
    does not throttle and would swallow the whole iterator up front.

    Each position's index is carried along and used to re-sort hits into
    sequential-scan order: `group()` elects the FIRST point of a zone as its
    representative, and that is the one printed as `setviewpos`."""
    import multiprocessing as mp
    from collections import deque
    t0 = time.time()
    raw = []
    vus = 0
    ctx = mp.get_context('spawn')
    with ctx.Pool(jobs, initializer=_worker_init,
                  initargs=(path, MASK, MIN_STEP, RELAX, POSTURE)) as pool:
        inflight = deque()
        ceiling = max(2, 2 * jobs)
        for chunk in _blocks(enumerate(source), CHUNK):
            inflight.append(pool.apply_async(_worker_chunk, (chunk,)))
            while len(inflight) >= ceiling:
                raw.extend(inflight.popleft().get())
                vus += CHUNK
                if verbose and vus % (CHUNK * 50) == 0:
                    print("  %d positions%s  (%.0fs, %d spots)"
                          % (vus, "/%d" % total if total else "",
                             time.time() - t0, len(raw)))
        while inflight:
            raw.extend(inflight.popleft().get())
    raw.sort(key=lambda r: r[0])
    if verbose:
        dt = time.time() - t0
        print("done in %.1fs  (%d processes)" % (dt, jobs))
    return [(p, k, z, d) for _i, p, k, z, d in raw]


def group(hits, radius=64.0):
    """Groups positions into zones (one spot = several positions)."""
    zones = []
    for p, n, z, d in hits:
        for zz in zones:
            c = zz['pts'][0]
            if (abs(p[0] - c[0]) <= radius and abs(p[1] - c[1]) <= radius
                    and abs(p[2] - c[2]) <= radius):
                zz['pts'].append(p)
                zz['step'] = max(zz['step'], n)
                break
        else:
            # 'step'/'z' of the representative (first point, the one printed
            # as setviewpos) kept separately: 'step' becomes the zone's max
            # and used to get mixed up with the representative's z.
            zones.append({'pts': [p], 'step': n, 'z': z, 'step0': n})
    return zones


def dump_path(name):
    if os.path.sep in name or name.endswith('.txt'):
        path = name
    else:
        path = os.path.join(cm_leafs.maps_dir(), name + '.txt')
    if not os.path.exists(path):
        print("Dump not found: %s" % path)
        print("Produce the dump with dump_clipmap.lua (see README).")
        sys.exit(2)
    return path


def load(name):
    return Detector(ClipMap(dump_path(name)))


# ------------------------------------------------------------------ CLI

def _extrait_maps_dir(argv):
    """Removes --maps-dir <path> from `argv` if present, applies it via
    `cm_leafs.set_maps_dir`. Returns the remaining list."""
    argv = list(argv)
    if '--maps-dir' in argv:
        i = argv.index('--maps-dir')
        cm_leafs.set_maps_dir(argv[i + 1])
        del argv[i:i + 2]
    return argv


def cmd_selftest(argv=()):
    """Non-regression on mp_farmhouse: values measured with a debugger."""
    _extrait_maps_dir(argv)
    d = load('mp_farmhouse')
    print("map : %s\n" % d.cm.name)
    ok = True
    cas = [("spot1", (285.871063, 107.856026, 200.125), 53, 253.125),
           ("spot2", (-457.03, 162.97, 106.125), 51, 157.125)]
    for lbl, p, expected_step, expected_z in cas:
        r = d.check(p)
        passed = r['spot'] and r['step'] == expected_step and abs(r['z'] - expected_z) < 0.01
        ok &= passed
        print("%-7s spot=%-5s %3d steps -> %.3f   expected %d -> %.3f   %s"
              % (lbl, r['spot'], r['step'], r['z'], expected_step, expected_z,
                 "OK" if passed else "FAIL"))
    # a known artifact: rises by a single unit
    r = d.check((-2684.2822, -700.1519, 199.125))
    passed = (not r['spot']) and r['step'] == 1
    ok &= passed
    print("%-7s spot=%-5s %3d steps              expected non-spot, 1 step   %s"
          % ("false+", r['spot'], r['step'], "OK" if passed else "FAIL"))
    print("\n%s" % ("SELFTEST OK" if ok else "SELFTEST FAILED"))
    return 0 if ok else 1


def cmd_check(argv):
    argv = _extrait_maps_dir(argv)
    if len(argv) < 4:
        print("usage: detect.py check <map> <x> <y> <z>")
        return 2
    d = load(argv[0])
    p = (float(argv[1]), float(argv[2]), float(argv[3]))
    r = d.check(p)
    print("map      : %s" % d.cm.name)
    print("position : (%.4f, %.4f, %.4f)" % p)
    print("  standing allsolid : %s" % r['standing'])
    print("  crouch allsolid   : %s   (must be False)" % r['crouching'])
    print("  climb             : %d steps -> z=%.3f" % (r['step'], r['z']))
    if r['delta'] is not None:
        print("  first delta       : %d %s" % (r['delta'], CORRECT_SOLID_DELTAS[r['delta']]))
    print("  SPOT              : %s" % ("YES" if r['spot'] else "no"))
    if r['standing'] and not r['crouching'] and not r['spot']:
        print("  (the triggering state exists but does not loop: artifact)")
    print("\n  setviewpos %.0f %.0f %.0f   (setviewpos's Z = eyes = z+60)"
          % (p[0], p[1], p[2] + 60))
    return 0


def cmd_scan(argv):
    argv = _extrait_maps_dir(argv)
    if not argv:
        print("usage: detect.py scan <map> [--step 8] [--min-step 10] [--mask 0x...]"
              " [--mode ground|air|full] [--jobs N] [--out f] [--maps-dir d]")
        return 2
    global MIN_STEP, MASK, RELAX, POSTURE
    name, step, out = argv[0], STEP_GRID, None
    mode = 'ground'
    # One core left free: the machine stays usable during the scan, and since
    # each worker holds its own copy of the clipmap, that is also one less
    # copy in memory. Measured cost: ~15% of speed.
    jobs = max(1, (os.cpu_count() or 1) - 1)
    i = 1
    while i < len(argv):
        if argv[i] == '--step':
            step = float(argv[i + 1]); i += 2
        elif argv[i] == '--min-step':
            MIN_STEP = int(argv[i + 1]); i += 2
        elif argv[i] == '--mask':
            MASK = int(argv[i + 1], 0); i += 2
        elif argv[i] == '--posture':
            POSTURE = argv[i + 1]; i += 2
        elif argv[i] == '--relax':
            RELAX = True; i += 1
        elif argv[i] == '--mode':
            mode = argv[i + 1]; i += 2
        elif argv[i] == '--jobs':
            jobs = max(1, int(argv[i + 1])); i += 2
        elif argv[i] == '--out':
            out = argv[i + 1]; i += 2
        else:
            i += 1
    path = dump_path(name)
    d = load(name)
    if POSTURE not in POSTURES:
        print("--posture must be 'standing' (default) or 'crouching'")
        return 2
    if mode not in ('ground', 'air', 'full'):
        print("--mode must be 'ground' (default), 'air', or 'full'")
        return 2
    print("map : %s   (step=%g, min-step=%d, mask=0x%08X, mode=%s, posture=%s, relax=%s, jobs=%d)"
          % (d.cm.name, step, MIN_STEP, MASK, mode, POSTURE, RELAX, jobs))
    if jobs > 1:
        if mode == 'ground':
            pts = d.candidates(step, mode)      # deduplicated, fits in memory
            print("candidate positions : %d" % len(pts))
            hits = scan_parallel(path, iter(pts), jobs, total=len(pts))
            del pts
        else:
            total = d.count_candidates(step, mode)
            print("candidate positions : %d (estimated, no dedup)" % total)
            hits = scan_parallel(path, d.iter_candidates(step, mode), jobs,
                                 total=total)
            vus, uniq = set(), []              # duplicates possible in 'air'
            for h in hits:
                if h[0] not in vus:
                    vus.add(h[0])
                    uniq.append(h)
            if len(uniq) != len(hits):
                print("  %d duplicate position(s) removed" % (len(hits) - len(uniq)))
            hits = uniq
    else:
        hits = d.scan(step, mode=mode)
    zones = group(hits)
    print("\n%d triggering positions, %d ZONE(S):" % (len(hits), len(zones)))
    for zz in sorted(zones, key=lambda z: -len(z['pts'])):
        c = zz['pts'][0]
        xs = [q[0] for q in zz['pts']]
        ys = [q[1] for q in zz['pts']]
        print("  %3d positions  %3d steps (max zone %3d)  z=%.3f -> %.3f   "
              "x[%.0f..%.0f] y[%.0f..%.0f]"
              % (len(zz['pts']), zz['step0'], zz['step'], c[2], zz['z'],
                 min(xs), max(xs), min(ys), max(ys)))
        # %.3f, not %.0f: on a thin brush, rounding to the unit moves the
        # point out of the triggering zone (measured on #4942).
        print("      setviewpos %.3f %.3f %.3f" % (c[0], c[1], c[2] + 60))
    if out:
        with open(out, 'w') as fh:
            for p, n, z, dd in hits:
                fh.write("%.4f %.4f %.4f step=%d z=%.4f delta=%s\n"
                         % (p[0], p[1], p[2], n, z, dd))
        print("\ndetailed positions -> %s" % out)
    return 0


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    c = sys.argv[1]
    if c == 'selftest':
        return cmd_selftest(sys.argv[2:])
    if c == 'check':
        return cmd_check(sys.argv[2:])
    if c == 'scan':
        return cmd_scan(sys.argv[2:])
    print(__doc__)
    return 2


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        sys.exit(130)
