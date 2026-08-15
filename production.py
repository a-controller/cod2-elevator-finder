"""Production launcher -- the `spot_exists` pipeline over a set of maps.

    python production.py run <map> [--jobs N] [--no-budget] [--maps-dir d]
    python production.py run --all [--jobs N] [--no-budget] [--maps-dir d]
    python production.py estimate [--maps-dir d]  # cost, without launching anything
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

HERE = os.path.dirname(os.path.abspath(__file__))
# Results directory, INSIDE the project directory. Override with the
# COD2_OUT_DIR environment variable. Must stay identical to
# characterize.SCRATCH: production.py spawns characterize.py as a subprocess
# and both read the same CSV.
SCRATCH = os.environ.get('COD2_OUT_DIR') or os.path.join(HERE, 'scratch')
DEFAULT_JOBS = 3

# Rate measured on mp_farmhouse (a clean reference: 1686 brushes, 273s, 3
# workers). OPTIMISTIC: a brush with a huge domain (>500k columns, over
# 7 min for the prefilter alone) can multiply this figure locally, unrelated
# to the brush count. Treat as a floor, not a reliable forecast.
SECONDS_PER_BRUSH = 273.0 / 1686


def _take_maps_dir(argv):
    """Pulls `--maps-dir <dir>` out of `argv`, applies it, returns (dir, rest).

    Every command that touches map files goes through this. Parsing the option
    inside `run` alone was a real bug: `estimate --maps-dir <dir>` silently
    looked in the default `maps/` next to the scripts and died on a bare
    FileNotFoundError.
    """
    argv = list(argv)
    maps_dir = None
    if '--maps-dir' in argv:
        i = argv.index('--maps-dir')
        if i + 1 >= len(argv):
            print("--maps-dir needs a directory")
            return None, argv
        maps_dir = argv[i + 1]
        del argv[i:i + 2]
        _import_characterize().cm_leafs.set_maps_dir(maps_dir)
    return maps_dir, argv


def list_maps():
    """Memory dumps (`<maps-dir>/*.txt`) + compiled maps with no dump
    (`*.d3dbsp`), the latter via characterize._load_clipmap's disk fallback."""
    c = _import_characterize()
    d = c.cm_leafs.maps_dir()
    if not os.path.isdir(d):
        raise SystemExit(
            "maps directory not found: %s\n"
            "Pass --maps-dir <dir>, set COD2_MAPS_DIR, or create a maps/ "
            "directory next to these scripts." % d)
    names = {f[:-4] for f in os.listdir(d) if f.endswith('.txt')}
    for d in c._bsp_dirs():
        if os.path.isdir(d):
            names.update(f[:-7] for f in os.listdir(d) if f.endswith('.d3dbsp'))
    return sorted(names)


def _import_characterize():
    import characterize
    return characterize


def orphans_count(name):
    c = _import_characterize()
    return len(c.orphans_of(name))


def cmd_estimate(argv=()):
    _take_maps_dir(argv)
    c = _import_characterize()
    print("%-22s %10s %12s" % ("map", "orphans", "estimate"))
    total = 0
    for name in list_maps():
        try:
            n = orphans_count(name)
        except Exception as e:
            print("%-22s ERROR %s" % (name, e))
            continue
        finally:
            # Each map is visited once here, so characterize's clipmap cache is
            # pure cost: it never evicts, and over a full maps directory it grew
            # until the process was killed. Drop it after every map.
            c.clear_cache()
        total += n
        print("%-22s %10d %9.1f min" % (name, n, n * SECONDS_PER_BRUSH / 60))
    print("\nTOTAL %d orphan brushes, ~%.1f h (OPTIMISTIC estimate -- see the warning"
          " at the top of this script: do not rely on it for a single map, only to"
          " order the batch)" % (total, total * SECONDS_PER_BRUSH / 3600))
    return 0


# 8 decimals, not 6: at 6, rounding moved the float32 for 15 of the 3643
# coordinates measured so far (small values near zero, where float32 resolves
# finer than 5e-7). 8 is the first width that leaves every one of them
# untouched, and map coordinates stay far below the magnitude where it would
# stop being enough.
COORD = "%.8f"


def _ansi():
    """ANSI color codes, or empty strings when coloring would be wrong.

    Colors are opt-out in three ways, because a public tool gets redirected:
    `NO_COLOR` (the de facto standard), a non-tty stdout (`> results.txt` must
    not collect escape sequences), and a Windows console that refuses to switch
    its virtual-terminal mode on.
    """
    off = {k: "" for k in ("dim", "bold", "good", "warn", "head", "off")}
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return off
    if os.name == "nt":
        try:
            import ctypes
            h = ctypes.windll.kernel32.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            if not ctypes.windll.kernel32.GetConsoleMode(h, ctypes.byref(mode)):
                return off
            # ENABLE_VIRTUAL_TERMINAL_PROCESSING
            if not ctypes.windll.kernel32.SetConsoleMode(h, mode.value | 0x0004):
                return off
        except Exception:
            return off
    return {"dim": "\033[2m", "bold": "\033[1m", "good": "\033[32m",
            "warn": "\033[33m", "head": "\033[36m", "off": "\033[0m"}


def _table(headers, rows, aligns):
    """Column-aligned lines. Widths come from the content, so a long map name
    or a 5-digit coordinate never breaks the alignment."""
    cols = list(zip(*([headers] + rows))) if rows else [(h,) for h in headers]
    w = [max(len(str(c)) for c in col) for col in cols]
    def line(cells):
        return "  ".join(
            (str(c).rjust(w[i]) if aligns[i] == "r" else str(c).ljust(w[i]))
            for i, c in enumerate(cells)).rstrip()
    return line(headers), [line(r) for r in rows]


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
    kept = sum(1 for r in rows
                  if r.get('n_survivors') not in ('', None)
                  and float(r['n_survivors']) >= 1)
    spots = [r for r in rows if r.get('spot') == 'True']
    divergences = [r for r in rows if r.get('spot') == 'DIVERGENCE']
    budget = sum(1 for r in rows if r.get('spot') == 'BUDGET_EXCEEDED')
    too_big = sum(1 for r in rows if r.get('spot') == 'TOO_BIG')
    c = _ansi()

    def amplitude(r):
        try:
            return float(r['run_delta0_max'])
        except (TypeError, ValueError):
            return 0.0

    # Highest rise first: that is the one worth trying in-game first.
    spots = sorted(spots, key=amplitude, reverse=True)
    divergences = sorted(divergences, key=amplitude, reverse=True)

    print("\n%sVERDICT %s%s : %s%d spots%s, %d DIVERGENCE, %d BUDGET_EXCEEDED, %d TOO_BIG"
          % (c["bold"], name, c["off"],
             c["good"] if spots else "", len(spots), c["off"],
             len(divergences), budget, too_big))
    print("  (%d orphan brushes processed, %d kept by the prefilter)" % (n, kept))

    def coord_rows(rs):
        out = []
        for r in rs:
            x, y, z = float(r['x_col']), float(r['y_col']), float(r['z_col'])
            out.append(["#" + str(r['brush_id']), r['run_delta0_max'],
                        COORD % x, COORD % y, COORD % z,
                        "setviewpos %.0f %.0f %.0f 0" % (x, y, z + 60)])
        return out

    ALIGN = ["l", "r", "r", "r", "r", "l"]
    HEAD = ["brush", "rise", "x", "y", "z", "console (approximate)"]

    if spots:
        head, lines = _table(HEAD, coord_rows(spots), ALIGN)
        print("\n  %s%s%s" % (c["head"], head, c["off"]))
        for l in lines:
            print("  %s%s%s" % (c["good"], l, c["off"]))
        print("  %sx/y/z are exact, write them straight into memory. `setviewpos` is"
              " rounded and may miss the column; the trailing 0 is the yaw the"
              " command requires.%s" % (c["dim"], c["off"]))

    if divergences:
        print("\n  %sDIVERGENCE -- pipeline >= MIN_STEP, but `detect.py check()` does not"
              " confirm. NEITHER spot NOR negative: needs manual review.%s"
              % (c["warn"], c["off"]))
        head, lines = _table(HEAD, coord_rows(divergences), ALIGN)
        print("  %s%s%s" % (c["head"], head, c["off"]))
        for l in lines:
            print("  %s%s%s" % (c["warn"], l, c["off"]))
    if budget or too_big:
        print("\n  %s%d BUDGET_EXCEEDED + %d TOO_BIG : NOT measured, do not read them as"
              " negatives -- `python production.py budget` to list them, `run <map>"
              " --no-budget` for the catch-up pass.%s"
              % (c["dim"], budget, too_big, c["off"]))


def run_one(name, jobs, without_budget=False, maps_dir=None):
    """Runs `characterize.py map-cost <name>` as a child process.

    `Popen`+`wait()`, not `subprocess.run()`: on Ctrl+C, both this process
    and the child receive the interrupt directly (same console process
    group on Windows), and the child is expected to catch its own and exit
    cleanly with code 130, printing its own one-line message. Waiting for
    that (instead of killing immediately, as `subprocess.run()` would) lets
    the child's message be the only one printed -- returns the string
    'interrupted' instead of a bool so the caller knows not to print its own."""
    cmd = [sys.executable, '-u', os.path.join(HERE, 'characterize.py'), 'map-cost', name,
           '--jobs', str(jobs)]
    if without_budget:
        cmd.append('--no-budget')
    if maps_dir:
        cmd += ['--maps-dir', maps_dir]
    print("\n=== %s%s ===" % (name, "  [NO BUDGET]" if without_budget else ""))
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=HERE)
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
    maps_dir, argv = _take_maps_dir(argv)
    without_budget = '--no-budget' in argv
    if '--all' in argv:
        targets = list_maps()
        print("%d maps to process (already-done ones skipped automatically), %d workers%s"
              % (len(targets), jobs, "  [NO BUDGET]" if without_budget else ""))
        all_ok = True
        for i, name in enumerate(targets, 1):
            print("\n[%d/%d]" % (i, len(targets)), end=' ')
            r = run_one(name, jobs, without_budget, maps_dir)
            if r == 'interrupted':
                return 130
            all_ok &= r
        return 0 if all_ok else 1
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
    maps_dir, argv = _take_maps_dir(argv)
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
    expected = (3455, 4957)
    passed = True
    for b in expected:
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
        return cmd_estimate(sys.argv[2:])
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
