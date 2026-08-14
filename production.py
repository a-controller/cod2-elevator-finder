"""Production launcher -- the `spot_exists` pipeline over a set of maps.

    python production.py run <map> [--jobs N] [--no-budget] [--maps-dir d]
    python production.py run --all [--jobs N] [--no-budget] [--maps-dir d]
    python production.py estimate                 # cost, without launching anything
    python production.py budget                   # TOO_BIG/BUDGET_EXCEEDED brushes, sorted by cost
    python production.py selftest [--maps-dir d]  # guard: reruns mp_farmhouse,
                                                    # verifies #3455/#4957 are found

`--maps-dir`: dumps/`.d3dbsp` directory, else `COD2_MAPS_DIR`, else `maps/`
next to this file.

`--no-budget`: catch-up pass. Resumes ONLY the TOO_BIG/BUDGET_EXCEEDED
brushes (a normal resume skips them), with no time budget or pre-estimation
-- reserved for deliberate manual use, never the default behavior. Use
`budget` first to see what is waiting and in what cost order.

WHY THIS SCRIPT EXISTS
A full scan on a big map can take hours; without resuming, a crash or an
abort forces starting over from scratch.

  1. ONE MAP = ONE PROCESS (`characterize.py map-cost <map>`, already
     equipped internally with a worker pool, memory ceiling, per-brush
     resume, and a 10-min/brush budget -- this launcher wraps it, does not
     reimplement it).
  2. RESULTS WRITTEN AS THEY GO into `scratch/production_<map>.csv`, one
     flush per brush. A crash costs at most the brush in progress (or
     10 min if a budget is exceeded), never the whole map.
  3. AUTOMATIC RESUME: rerunning `run <map>` skips everything already in
     the CSV. No configuration needed.

No buffered output: progress is visible live.
"""
import csv
import glob
import os
import subprocess
import sys
import time

ICI = os.path.dirname(os.path.abspath(__file__))
# Results directory, INSIDE the project directory. Override with the
# COD2_OUT_DIR environment variable. Must stay identical to
# characterize.SCRATCH: production.py spawns characterize.py as a subprocess
# and both read the same CSV.
SCRATCH = os.environ.get('COD2_OUT_DIR') or os.path.join(ICI, 'scratch')
DEFAULT_JOBS = 3

# Rate measured on mp_farmhouse (a clean reference: 1686 brushes, 273s, 3
# workers). OPTIMISTIC: a brush with a huge domain (>500k columns, over
# 7 min for the prefilter alone) can multiply this figure locally, unrelated
# to the brush count. Treat as a floor, not a reliable forecast.
SECONDS_PER_BRUSH = 273.0 / 1686


def list_maps():
    """Memory dumps (`<maps-dir>/*.txt`) + compiled maps with no dump
    (`*.d3dbsp`), the latter via characterize._load_clipmap's disk fallback."""
    c = _import_characterize()
    noms = {f[:-4] for f in os.listdir(c.cm_leafs.maps_dir()) if f.endswith('.txt')}
    for d in c._bsp_dirs():
        if os.path.isdir(d):
            noms.update(f[:-7] for f in os.listdir(d) if f.endswith('.d3dbsp'))
    return sorted(noms)


def _import_characterize():
    import characterize
    return characterize


def orphans_count(name):
    c = _import_characterize()
    return len(c.orphans_of(name))


def cmd_estimate():
    print("%-22s %10s %12s" % ("map", "orphans", "estimate"))
    tot = 0
    for name in list_maps():
        try:
            n = orphans_count(name)
        except Exception as e:
            print("%-22s ERROR %s" % (name, e))
            continue
        tot += n
        print("%-22s %10d %9.1f min" % (name, n, n * SECONDS_PER_BRUSH / 60))
    print("\nTOTAL %d orphan brushes, ~%.1f h (OPTIMISTIC estimate -- see the warning"
          " at the top of this script: do not rely on it for a single map, only to"
          " order the batch)" % (tot, tot * SECONDS_PER_BRUSH / 3600))
    return 0


def _verdict(name):
    """Human-readable verdict line. Priority given to the EXACT origin (full
    floating-point precision, to be written directly into memory via an
    external tool): in-game `setviewpos` is INSUFFICIENT, the console does
    not accept decimals and rounding alone can be enough to miss the column
    (measured: a 0.000414 gap was enough to trigger nothing on a brush).
    Retained columns often fall exactly on the dilated domain's boundary
    (#4957 too, confirmed spot) -- sub-unit precision is not incidental, it
    is necessary. `setviewpos` is still shown, but explicitly marked
    approximate."""
    path = os.path.join(SCRATCH, 'production_%s.csv' % name)
    if not os.path.exists(path):
        print("(no result written for %s)" % name)
        return
    with open(path, newline='') as fh:
        rows = list(csv.DictReader(fh))
    n = len(rows)
    retenus = sum(1 for r in rows
                  if r.get('n_survivors') not in ('', None)
                  and float(r['n_survivors']) >= 1)
    spots = [r for r in rows if r.get('spot') == 'True']
    divergences = [r for r in rows if r.get('spot') == 'DIVERGENCE']
    budget = sum(1 for r in rows if r.get('spot') == 'BUDGET_EXCEEDED')
    too_big = sum(1 for r in rows if r.get('spot') == 'TOO_BIG')
    print("\nVERDICT %s : %d spots, %d DIVERGENCE, %d BUDGET_EXCEEDED, %d TOO_BIG"
          % (name, len(spots), len(divergences), budget, too_big))
    print("  (%d orphan brushes processed, %d kept by the prefilter)" % (n, retenus))
    for r in spots:
        x, y, z = float(r['x_col']), float(r['y_col']), float(r['z_col'])
        print("  brush #%-8s run_delta0_max=%s" % (r['brush_id'], r['run_delta0_max']))
        print("     origin (CE, Float) : x=%r  y=%r  z=%r" % (x, y, z))
        print("     setviewpos %.0f %.0f %.0f   (approximate -- may not trigger, rounded)"
              % (x, y, z + 60))
    if divergences:
        print("  WARNING -- brushes in DIVERGENCE (pipeline >= MIN_STEP, but `detect.py check()`"
              " does not confirm -- NEITHER spot NOR negative, needs manual review):")
        for r in divergences:
            print("     brush #%-8s run_delta0_max=%s  (x=%s, y=%s, z=%s)"
                  % (r['brush_id'], r['run_delta0_max'], r['x_col'], r['y_col'], r['z_col']))
    if budget or too_big:
        print("  (%d BUDGET_EXCEEDED + %d TOO_BIG : NOT measured, do not read them as"
              " negatives -- `python production.py budget` to list them, `run <map>"
              " --no-budget` for the catch-up pass.)"
              % (budget, too_big))


def run_one(name, jobs, without_budget=False, maps_dir=None):
    """Runs `characterize.py map-cost <name>` as a child process.

    `Popen`+`wait()`, not `subprocess.run()`: on Ctrl+C, both this process
    and the child receive the interrupt directly (same console process
    group on Windows), and the child is expected to catch its own and exit
    cleanly with code 130, printing its own one-line message. Waiting for
    that (instead of killing immediately, as `subprocess.run()` would) lets
    the child's message be the only one printed -- returns the string
    'interrupted' instead of a bool so the caller knows not to print its own."""
    cmd = [sys.executable, '-u', os.path.join(ICI, 'characterize.py'), 'map-cost', name,
           '--jobs', str(jobs)]
    if without_budget:
        cmd.append('--no-budget')
    if maps_dir:
        cmd += ['--maps-dir', maps_dir]
    print("\n=== %s%s ===" % (name, "  [NO BUDGET]" if without_budget else ""))
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=ICI)
    try:
        rc = p.wait()
    except KeyboardInterrupt:
        try:
            rc = p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()
            rc = p.wait()
    if rc == 130:
        return 'interrupted'
    dt = time.time() - t0
    ok = rc == 0
    print("  -> %s in %.0fs" % ("OK" if ok else "ERROR/ABORT (code %d)" % rc, dt))
    _verdict(name)
    return ok


def cmd_run(argv):
    argv = list(argv)
    jobs = DEFAULT_JOBS
    if '--jobs' in argv:
        jobs = int(argv[argv.index('--jobs') + 1])
    maps_dir = None
    if '--maps-dir' in argv:
        i = argv.index('--maps-dir')
        maps_dir = argv[i + 1]
        del argv[i:i + 2]
    without_budget = '--no-budget' in argv
    if '--all' in argv:
        if maps_dir:
            _import_characterize().cm_leafs.set_maps_dir(maps_dir)
        cibles = list_maps()
        print("%d maps to process (already-done ones skipped automatically), %d workers%s"
              % (len(cibles), jobs, "  [NO BUDGET]" if without_budget else ""))
        tout_ok = True
        for i, name in enumerate(cibles, 1):
            print("\n[%d/%d]" % (i, len(cibles)), end=' ')
            r = run_one(name, jobs, without_budget, maps_dir)
            if r == 'interrupted':
                return 130
            tout_ok &= r
        return 0 if tout_ok else 1
    if not argv or argv[0].startswith('--'):
        print("usage: production.py run <map> [--jobs N] [--no-budget] [--maps-dir d]  |"
              "  run --all [--jobs N] [--no-budget] [--maps-dir d]")
        return 2
    ok = run_one(argv[0], jobs, without_budget, maps_dir)
    if ok == 'interrupted':
        return 130
    return 0 if ok else 1


def cmd_budget():
    """Lists all TOO_BIG/BUDGET_EXCEEDED brushes from ALL maps already run
    through production, sorted by increasing cost -- to organize a catch-up
    pass (`run <map> --no-budget`) starting with the cheapest."""
    lines = []
    for path in sorted(glob.glob(os.path.join(SCRATCH, 'production_*.csv'))):
        name = os.path.basename(path)[len('production_'):-len('.csv')]
        with open(path, newline='') as fh:
            for r in csv.DictReader(fh):
                if r.get('spot') in ('TOO_BIG', 'BUDGET_EXCEEDED'):
                    try:
                        cost = float(r.get('estimated_cost') or 'nan')
                    except ValueError:
                        cost = float('nan')
                    lines.append((cost, name, r['brush_id'], r['spot']))
    if not lines:
        print("no TOO_BIG/BUDGET_EXCEEDED brush in the production CSVs.")
        return 0
    # nan (unknown cost, CSVs written before estimated_cost was added) goes last
    import math
    lines.sort(key=lambda t: (math.isnan(t[0]), t[0]))
    print("%-16s %-8s %-16s %14s" % ("map", "brush", "category", "estimated_cost"))
    for cost, name, b, cat in lines:
        print("%-16s %-8s %-16s %14s" % (name, b, cat, ("%.0fs" % cost) if cost == cost else "unknown"))
    print("\n%d brushes to catch up. `python production.py run <map> --no-budget`"
          " to resume them, one at a time or map by map." % len(lines))
    return 0


def cmd_selftest(argv=()):
    """Guard: reruns mp_farmhouse ENTIRELY (erases the previous result so it
    cannot benefit from resuming) and checks that #3455 AND #4957 both come
    out with spot=True. This is what makes it possible to trust a result
    without re-running the whole project's history."""
    argv = list(argv)
    maps_dir = None
    if '--maps-dir' in argv:
        maps_dir = argv[argv.index('--maps-dir') + 1]
    name = 'mp_farmhouse'
    path = os.path.join(SCRATCH, 'production_%s.csv' % name)
    if os.path.exists(path):
        os.remove(path)
        print("(previous result for %s erased for a full selftest)" % name)
    ok = run_one(name, DEFAULT_JOBS, maps_dir=maps_dir)
    if ok == 'interrupted':
        return 130
    if not ok:
        print("\nSELFTEST FAILED: the command returned an error/abort.")
        return 1
    with open(path, newline='') as fh:
        rows = {int(r['brush_id']): r for r in csv.DictReader(fh)}
    attendus = (3455, 4957)
    passed = True
    for b in attendus:
        r = rows.get(b)
        found = r is not None and r.get('spot') == 'True'
        passed &= found
        print("  brush #%d : %s" % (b, "spot=True found" if found else "MISSING or spot != True"))
    print("\n%s" % ("SELFTEST OK" if passed else "SELFTEST FAILED"))
    return 0 if passed else 1


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    c = sys.argv[1]
    if c == 'run':
        return cmd_run(sys.argv[2:])
    if c == 'estimate':
        return cmd_estimate()
    if c == 'budget':
        return cmd_budget()
    if c == 'selftest':
        return cmd_selftest(sys.argv[2:])
    print(__doc__)
    return 2


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        # Only reached for an interrupt outside run_one's own child-wait
        # (e.g. during `estimate`/`budget`, or before the child process
        # started) -- run_one's own KeyboardInterrupt handling already
        # covers the child printing its message, so this is the only path
        # left needing its own line.
        print("interrupted", file=sys.stderr)
        sys.exit(130)
