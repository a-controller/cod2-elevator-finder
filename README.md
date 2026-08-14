# cod2-elevator-finder

Finds elevator spots in Call of Duty 2 maps by reading their collision data.
Never touches the game.

The "elevator" is a movement glitch in Call of Duty 2's id Tech 3-derived
engine: at certain spots, the player gets pushed upward one unit per frame
and rides up as if on a lift. This tool reads a map's collision data offline
and reports every brush where the conditions for it hold, so you can go and
try them in-game.

No third-party dependencies. Pure Python 3 standard library.

## The mechanism

The engine uses two different collision-selection paths, and they can
disagree on a **non-axial** BSP node plane:

* the **swept** trace (`CM_TraceThroughTree`, used whenever start != end)
  applies a fixed **2048** margin as soon as the trace is not a single
  point, and so it sees a given brush;
* the **point** trace (`CM_BoxLeafnums_r` then `BoxOnPlaneSide`, used when
  start == end) resolves the plane side exactly, and does not see that
  brush.

`PM_CorrectAllSolid` runs its very first correction delta ({0,0,1}) through
the point path. When a brush is visible to the swept path but not to the
point path, the correction "fixes" a solid state the point test cannot see,
and pushes the player up by +1 unit. This repeats on every frame the
condition holds, which turns a one-off nudge into a real vertical climb.

The scanner searches every orphan brush (one the swept path can reach but
the point path's leaf listing does not carry) for a column and a Z range
where that escape happens over a long enough consecutive run to be a genuine
climb rather than a single-step artifact.

## Quick start

Scan one map, using a memory dump you produced yourself:

```
python production.py run mp_farmhouse --maps-dir "C:\path\to\your\maps"
```

That prints a verdict per brush and writes `scratch/production_mp_farmhouse.csv`.
A confirmed spot looks like this:

```
brush #4957      run_delta0_max=64.0
   origin (CE, Float) : x=283.875  y=125.073719  z=252.5
   setviewpos 284 125 313   (approximate -- may not trigger, rounded)
```

The `origin` line is the exact position, at full floating-point precision.
Use it if you can write coordinates directly into memory. The `setviewpos`
line is the in-game console command, but the console does not accept
decimals, and rounding alone is sometimes enough to miss the column.

The scan runs on three worker processes by default. On a machine you want to
keep using while it works, or one that is short on cores or memory, lower it:

```
python production.py run mp_farmhouse --jobs 1 --maps-dir "C:\path\to\your\maps"
```

That is roughly three times slower and completely safe: results are written
brush by brush, and re-running the same command resumes where it stopped. For
scale, `mp_farmhouse` (1686 brushes) takes about 4.5 minutes and 154 MB on
three workers.

To check the tool is working before trusting a result:

```
python production.py selftest --maps-dir "C:\path\to\your\maps"
```

It rescans `mp_farmhouse` and verifies that brushes `#3455` and `#4957` are
found. Both are known, verified spots.

## Bring your own map files

**This repository ships no game content.** No `.d3dbsp`, no `.map`, no
memory dump. You provide your own, from your own copy of the game.

Two input paths are supported. Both are resolved from `--maps-dir`, or the
`COD2_MAPS_DIR` environment variable, or a `maps/` directory next to the
scripts:

* **Memory dump** (`<map>.txt`), the priority path. Produced in-game by
  `dump_clipmap.lua`, included here. It is a read-only Cheat Engine script:
  it walks the loaded clipmap and writes it to disk, and never writes to the
  game's memory. Set `OUT_DIR` at the top of the script to the directory the
  scanner reads, then follow the instructions in its header. Load the map with
  `devmap <map>` so you are dumping from your own offline session.
* **Compiled map** (`<map>.d3dbsp`), the fallback. Read straight off disk,
  with no need to load the map in-game. Works with `production.py` and
  `characterize.py`, but not with `detect.py` called directly.

**Only one build has been tested: CoD2x 1.4.6.8.** Every dump, every spot and
every figure in this README comes from that build. Stock Call of Duty 2 has
never been tried. `dump_clipmap.lua` reads the clipmap through hardcoded
structure offsets, so a build with a different layout may need `CM_ADDR` set
by hand, or new offsets. On such a build the plausibility checks should reject
every candidate and the script should stop with `could not locate clipMap_t`
rather than write a wrong dump. That is the intended failure, not a guarantee.
Check the counters it prints against the map you loaded before trusting a dump
from any other build. The `.d3dbsp` path does not read game memory and is not
affected.

**The `.d3dbsp` path under-detects.** Measured on `mp_farmhouse`, it finds
only 1 of the 2 known reference spots. Brush `#3455` differs between the two
sources: `mins.z` is 208.0 on disk against 200.0 in the memory dump, and the
disk version carries 4 non-axial planes instead of 3, a bevel plane added at
compile time. That changes its leaf coverage, and the prefilter rejects it.
Prefer the memory dump when you can get one, and read a `.d3dbsp`-only scan
as a lower bound rather than a complete result.

## Commands

### production.py, the recommended entry point

```
python production.py run <map> [--jobs N] [--no-budget] [--maps-dir d]
python production.py run --all [--jobs N] [--no-budget] [--maps-dir d]
python production.py estimate
python production.py budget
python production.py selftest [--maps-dir d]
```

It wraps `characterize.py map-cost` with per-brush resume, a memory ceiling,
and a 10-minute budget per brush. Results are written as they are produced
to `scratch/production_<map>.csv` inside this project directory, one row per
orphan brush. Re-running `run <map>` skips everything already in the CSV, so
a crash or an abort only costs the brush in progress.

| option | meaning |
|---|---|
| `--maps-dir <dir>` | where your dumps and `.d3dbsp` files live. Falls back to `COD2_MAPS_DIR`, then to `maps/` next to the scripts. |
| `--jobs N` | worker processes, default 3. More is faster, but when two columns tie on the same score, which one gets reported is not reproducible. Peak memory is measured across the scan's own processes: 154 MB on `mp_farmhouse` (1686 brushes, 3 workers). Heavier maps use more, against a 2.5 GB ceiling that stops the run; re-running resumes from the CSV. Windows only, see Known limitations. |
| `--all` | every map found in the maps directory instead of a single one. |
| `--no-budget` | catch-up pass. Re-runs only the brushes previously marked `TOO_BIG` or `BUDGET_EXCEEDED`, with no time budget. Run `budget` first to see what is waiting. |

`estimate` prints the cost of a full run without launching anything.
`budget` lists the brushes that were skipped for being too expensive, sorted
by increasing cost.

Set `COD2_OUT_DIR` to write results somewhere other than `scratch/`.

### detect.py, the underlying detector

```
python detect.py check <map> <x> <y> <z> [--maps-dir d]
python detect.py scan <map> [options]
python detect.py selftest [--maps-dir d]
```

`check` answers a single question: does this exact position climb, and by
how much. It is the ground-truth predicate, and the right tool for verifying
a spot someone reported to you.

**These three commands need a memory dump.** Unlike `production.py` and
`characterize.py`, `detect.py` called directly does not fall back to a
`.d3dbsp`, and will stop with "Dump not found". If all you have is a
compiled map, use `production.py run` instead, which reads it fine.

`scan` sweeps a whole map on a regular grid. Be aware that it is a **weaker
instrument** than `production.py`: on `jm_temple`, a full-mode grid scan
found 1 of the 8 spots that `production.py` finds. Spots often sit on the
exact edge of a brush's overlap region, which `production.py` targets
explicitly and a regular grid almost always steps over.

| option | default | meaning |
|---|---|---|
| `--step <n>` | 8 | grid spacing in units. Smaller finds more and costs proportionally more. |
| `--min-step <n>` | 10 | minimum climb height to report. A real climb makes about 50 steps, an artifact makes 1. |
| `--mode ground\|air\|full` | ground | `ground` only tests positions standing on a surface, `air` adds 5 height levels per brush, `full` sweeps the whole Z interval where the player box can touch the brush. On `jm_temple` that is 0.5M, 2.7M and 24M positions respectively. |
| `--posture standing\|crouching` | standing | player box used for the test. |
| `--mask 0x...` | `0x02810011` | collision mask. The default is the multiplayer client's value. A singleplayer binary may differ, and a wrong mask silently invalidates everything. `dump_clipmap.lua` prints the right one for your build. |
| `--jobs N` | cores minus 1 (so 5 on a 6-core machine) | worker processes. Note this differs from `production.py`, which caps the default at 3. |
| `--out <file>` | none | writes every triggering position to a file. |
| `--relax` | off | drops the "crouched is not allsolid" condition. Diagnostic use. |

### characterize.py, the per-brush engine

```
python characterize.py map-cost <map> [--jobs N] [--no-budget] [--maps-dir d]
```

`production.py` calls this for you. Call it directly only if you want a
single map with no launcher around it.

## Reading the results

The CSV has one row per orphan brush. The columns that matter:

| column | meaning |
|---|---|
| `brush_id` | the brush's index in the map's collision data |
| `spot` | `True` for a confirmed spot, `DIVERGENCE` when the pipeline and `detect.py` disagree, `TOO_BIG` or `BUDGET_EXCEEDED` when the brush was skipped as too expensive |
| `run_delta0_max` | height of the climb, in units. This is the predicate. |
| `x_col`, `y_col`, `z_col` | the position that triggers it |
| `n_survivors` | columns that passed the prefilter for this brush |

`TOO_BIG` and `BUDGET_EXCEEDED` are **not** negatives. Those brushes were
never measured. Use `budget` to list them and `run <map> --no-budget` to
work through them.

## Known limitations

* **Tested on CoD2x 1.4.6.8 only.** Stock Call of Duty 2 has never been
  tried. The memory dump depends on hardcoded clipmap offsets, so another
  build may need `CM_ADDR` set by hand, or new offsets. See "Bring your own
  map files".
* **The memory ceiling is Windows-only.** `memwatch.py` reads process
  memory through `tasklist`. On any other OS the ceiling that aborts a
  runaway `production.py` run is inactive. The scan still works, it just
  runs without that safety net.
* **Tie-breaking between equal-scoring columns is not reproducible.** Out of
  roughly 189 rows, 2 or 3 can differ in the decimal precision of `x_col`
  and `y_col` between two runs on the same input, even with `--jobs 1`.
  That is optimizer noise on near-equal candidates, not a correctness
  problem. The reported column is still a valid spot.
* **Only brushes with at least one non-axial plane are examined.** The
  candidate enumeration skips purely axial brushes. Every spot found so far
  has a non-axial plane, but that is partly a consequence of what was
  looked at, so a purely axial spot would not be found by this tool as it
  stands.
* **A result is only ever a candidate.** Confirm it in-game before treating
  it as real.
