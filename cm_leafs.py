"""Brush selection for a trace — the engine's TWO paths.

Replacing the tree walk with a list of neighboring brushes is NOT neutral: a
brush can be seen by one path and not the other, and that is exactly the
elevator's mechanism.

The engine takes one of two paths DEPENDING ON WHETHER start == end:

  swept : CM_TraceThroughTree   (cm_trace.cpp:1980)
          segment descent, offset = size[axis] + SURFACE_CLIP_EPSILON,
          and 2048 on non-axial planes  ->  VERY permissive
  point : CM_PositionTest       (cm_trace.cpp:2094)
          -> CM_BoxLeafnums_r   (cm_test.cpp:84) + BoxOnPlaneSide
          EXACT box test on extents.start +- size +- 1

So a brush can be seen by one and not the other: that is exactly what
produces the elevator.

The memory dump (`ClipMap._load`) is read DIRECTLY from the engine's memory.
The compiled `.d3dbsp` (fallback, `disk_clipmap.DiskClipMap`) is NOT an
equivalent source: its geometry can differ from what the engine actually
sees in-game (cf. `disk_clipmap.py`, leaf bounds section).
"""
import os
from bisect import bisect_right

SURFACE_CLIP_EPSILON = 0.125


class MapLoadError(Exception):
    """A map's dump/.d3dbsp could not be located or read (missing file,
    unreadable, bad --maps-dir). Caught at the CLI entry points for a
    one-line message instead of a raw traceback -- never raised for an
    actual bug (parsing/format errors keep propagating as-is)."""


_MAPS_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'maps')
_maps_dir_override = None


def set_maps_dir(path):
    """Called by the CLIs (--maps-dir) before any map is loaded.

    Also exported to the environment: worker processes are created with the
    `spawn` start method on Windows, so they re-import this module from
    scratch and do NOT inherit `_maps_dir_override`. The environment IS
    inherited, so this is what makes --maps-dir reach the workers -- and the
    workers are where all the actual scanning happens.
    """
    global _maps_dir_override
    _maps_dir_override = path
    os.environ['COD2_MAPS_DIR'] = path


def maps_dir():
    """--maps-dir (CLI, set_maps_dir) > COD2_MAPS_DIR (env) > maps/ next to this module."""
    return _maps_dir_override or os.environ.get('COD2_MAPS_DIR') or _MAPS_DIR_DEFAULT


def has_maps_dir_override():
    """True if --maps-dir or COD2_MAPS_DIR explicitly sets the directory."""
    return bool(_maps_dir_override or os.environ.get('COD2_MAPS_DIR'))


DEFAULT_DUMP = os.path.join(_MAPS_DIR_DEFAULT, 'mp_farmhouse.txt')


class ClipMap(object):
    """Collision geometry as the engine sees it."""

    def __init__(self, path=DEFAULT_DUMP):
        self.name = None
        self.nodes = []      # (normal, dist, type, child0, child1)
        self.leafs = []      # (mins, maxs, brushContents, leafBrushNode)
        self.lbn = []        # ('L', contents, [brushnums]) | ('N', contents, axis, dist, range, off0, off1)
        self.brushes = []    # (mins, maxs, contents, [(normal, dist), ...])
        self._load(path)

    def _load(self, path):
        with open(path, 'r') as fh:
            lines = fh.read().splitlines()
        i = 0
        self.name = lines[0].split(' ', 1)[1]
        i = 1
        while i < len(lines):
            head = lines[i].split()
            kind, count = head[0], int(head[1])
            i += 1
            if kind == 'NODES':
                for k in range(count):
                    f = lines[i + k].split()
                    self.nodes.append(((float(f[1]), float(f[2]), float(f[3])),
                                       float(f[4]), int(f[5]), int(f[6]), int(f[7])))
            elif kind == 'LEAFS':
                for k in range(count):
                    f = lines[i + k].split()
                    self.leafs.append(((float(f[1]), float(f[2]), float(f[3])),
                                       (float(f[4]), float(f[5]), float(f[6])),
                                       int(f[7]), int(f[8])))
            elif kind == 'LBN':
                for k in range(count):
                    f = lines[i + k].split()
                    if f[1] == 'L':
                        self.lbn.append(('L', int(f[2]), int(f[3]),
                                         [int(x) for x in f[4:]]))
                    else:
                        # ('N', leafBrushCount, contents, axis, dist, range, off0, off1)
                        self.lbn.append(('N', int(f[2]), int(f[3]), int(f[4]),
                                         float(f[5]), float(f[6]),
                                         int(f[7]), int(f[8])))
            elif kind == 'BRUSHES':
                for k in range(count):
                    f = lines[i + k].split()
                    npl = int(f[8])
                    planes = []
                    for j in range(npl):
                        b = 9 + j * 4
                        planes.append(((float(f[b]), float(f[b + 1]), float(f[b + 2])),
                                       float(f[b + 3])))
                    self.brushes.append(((float(f[1]), float(f[2]), float(f[3])),
                                         (float(f[4]), float(f[5]), float(f[6])),
                                         int(f[7]), planes))
            i += count

    # ---------------------------------------------------------------- leafs

    def box_on_plane_side(self, bmin, bmax, node):
        """BoxOnPlaneSide: 1 front, 2 back, 3 both."""
        normal, dist, ptype = node[0], node[1], node[2]
        if ptype < 3:                                  # fast axial case
            if dist <= bmin[ptype]:
                return 1
            if dist >= bmax[ptype]:
                return 2
            return 3
        d1 = d2 = 0.0
        for j in range(3):
            if normal[j] > 0:
                d1 += normal[j] * bmax[j]
                d2 += normal[j] * bmin[j]
            else:
                d1 += normal[j] * bmin[j]
                d2 += normal[j] * bmax[j]
        s = 0
        if d1 >= dist:
            s = 1
        if d2 < dist:
            s += 2
        return s

    def box_leafnums(self, bmin, bmax):
        """CM_BoxLeafnums_r (cm_test.cpp:84) — POINT path."""
        out = []
        stack = [0]
        while stack:
            num = stack.pop()
            while num >= 0:
                node = self.nodes[num]
                s = self.box_on_plane_side(bmin, bmax, node)
                if s == 1:
                    num = node[3]
                else:
                    if s != 2:
                        stack.append(node[3])
                    num = node[4]
            out.append(-1 - num)
        return out

    # ------------------------------------------------------------- caches

    def build_cache(self, mask):
        """Precomputes everything static. Call before a scan.

        - `_lb[leafnum]`  : the leaf's brush indices, already filtered by `mask`
        - `_pl[brushnum]` : the 9 planes (6 axial + non-axial), ready to use
        - `_stamp`        : marking array for `_collect_filtered`'s dedup
                            (see the note there)

        `leaf_brushes()` and `brush_planes()` used to be recomputed on every
        call (21787 calls for 121 positions) even though they only depend on
        geometry. Direct gain, no change in semantics."""
        self._mask = mask
        self._pl = [self.brush_planes(b) for b in range(len(self.brushes))]
        self._stamp = [0] * len(self.brushes)
        self._tok = 0
        self._lb = []
        self._lbz = []       # mins.z of _lb's brushes, sorted ascending
        for li in range(len(self.leafs)):
            seen, keep = set(), []
            for b in self.leaf_brushes(li):
                if b not in seen:
                    seen.add(b)
                    if self.brushes[b][2] & mask:
                        keep.append(b)
            keep.sort(key=lambda b: self.brushes[b][0][2])
            self._lb.append(tuple(keep))
            self._lbz.append(tuple(self.brushes[b][0][2] for b in keep))
        return self

    def trace_leafnums(self, p1, p2, size):
        """CM_TraceThroughTree (cm_trace.cpp:1980-2086) — SWEPT path.

        COMPLETE transposition, including the segment split (l. 2048-2082).
        Descending into both children with p1/p2 unchanged would return a
        SUPERSET of the leafs, hence brushes the engine never submits —
        a source of false positives.

            diff = t2 - t1
            frac2 = (fsel(diff, -t1, t1) - offset) / |diff|
            frac  = (fsel(diff, -t1, t1) + offset) / |diff|
            side  = 1 if diff >= 0 else 0
            mid = p1 + (p2 - p1) * min(frac, 1)
            recurse(children[side], p1, mid)
            p1 = p1 + (p2 - p1) * max(frac2, 0)
            num = children[side ^ 1]

        `p1[3]`/`p2[3]` carry the fraction (0 -> 1); the test
        `if (p1[3] >= trace->fraction) return` is applied with the initial
        fraction at 1.0 (no brush traces are done during the walk, so it
        never decreases — conservative)."""
        out = []
        a = (p1[0], p1[1], p1[2], 0.0)
        b = (p2[0], p2[1], p2[2], 1.0)
        self._tree_r(0, a, b, size, out)
        return out

    def _tree_r(self, num, p1, p2, size, out, depth=0):
        if depth > 128:
            return
        while num >= 0:
            normal, dist, ptype, c0, c1 = self.nodes[num]
            if ptype < 3:
                t1 = p1[ptype] - dist
                t2 = p2[ptype] - dist
                offset = size[ptype] + SURFACE_CLIP_EPSILON
            else:
                n0, n1, n2 = normal
                t1 = n0 * p1[0] + n1 * p1[1] + n2 * p1[2] - dist
                t2 = n0 * p2[0] + n1 * p2[1] + n2 * p2[2] - dist
                offset = 2048.0                # cm_trace.cpp:2027 "this is silly"
            if t1 < t2:
                tmin, tmax = t1, t2
            else:
                tmin, tmax = t2, t1
            if tmin >= offset:
                num = c0
                continue
            if -offset >= tmax:
                num = c1
                continue
            if p1[3] >= 1.0:                   # l. 2043, initial fraction
                return
            diff = t2 - t1
            absDiff = diff if diff >= 0.0 else -diff
            if absDiff > 0.00000047683716:
                base = -t1 if diff >= 0.0 else t1
                inv = 1.0 / absDiff
                frac2 = (base - offset) * inv
                frac = (base + offset) * inv
                side = 1 if diff >= 0.0 else 0
            else:
                side = 0
                frac = 1.0
                frac2 = 0.0
            if frac > 1.0:
                frac = 1.0
            mid = (p1[0] + (p2[0] - p1[0]) * frac,
                   p1[1] + (p2[1] - p1[1]) * frac,
                   p1[2] + (p2[2] - p1[2]) * frac,
                   p1[3] + (p2[3] - p1[3]) * frac)
            self._tree_r(c0 if side == 0 else c1, p1, mid, size, out, depth + 1)
            if frac2 < 0.0:
                frac2 = 0.0
            p1 = (p1[0] + (p2[0] - p1[0]) * frac2,
                  p1[1] + (p2[1] - p1[1]) * frac2,
                  p1[2] + (p2[2] - p1[2]) * frac2,
                  p1[3] + (p2[3] - p1[3]) * frac2)
            num = c1 if side == 0 else c0
        out.append(-1 - num)

    # --------------------------------------------------------------- brushes

    def leaf_brushes(self, leafnum):
        """Brushes of a leaf's leafBrushNodes subtree.

        Engine structure (CM_TraceThroughLeafBrushNode_r, cm_trace.cpp:1618):
            if (node->leafBrushCount) {
                if (node->leafBrushCount > 0) { ...brushes...; return; }
                recurse(node + 1);              // leafBrushCount < 0
            }
            ... then internal-node handling via childOffset[2]

        Conservative: the ENTIRE subtree is collected (a superset); the
        engine prunes it against the box/segment. Conservative, so no false
        negative."""
        root = self.leafs[leafnum][3]
        out = []
        stack = [root]
        seen = set()
        while stack:
            ni = stack.pop()
            if ni in seen or not (0 <= ni < len(self.lbn)):
                continue
            seen.add(ni)
            node = self.lbn[ni]
            if node[0] == 'L':                 # leafBrushCount > 0: leaf
                out.extend(node[3])
                continue
            if node[1] < 0:                    # leafBrushCount < 0: sub-node at ni+1
                stack.append(ni + 1)
            stack.append(ni + node[6])         # childOffset[0]
            stack.append(ni + node[7])         # childOffset[1]
        return out

    def brushes_for_point(self, origin, mins, maxs, mask):
        """Brushes submitted to a POINT trace (the 26 deltas)."""
        offset = tuple((mins[j] + maxs[j]) * 0.5 for j in range(3))
        size = tuple(maxs[j] - offset[j] for j in range(3))
        start = tuple(origin[j] + offset[j] for j in range(3))
        bmin = tuple(start[j] - size[j] - 1.0 for j in range(3))
        bmax = tuple(start[j] + size[j] + 1.0 for j in range(3))
        radius = size[2] if size[0] > size[2] else size[0]
        ro = (radius, radius, radius + (size[2] - radius))
        leafnums = self.box_leafnums(bmin, bmax)
        if getattr(self, '_lb', None) is None:
            return self._bbox_filter(self._collect(leafnums, mask), start, start, ro)
        return self._collect_filtered(leafnums, start, start, ro)

    def brushes_for_sweep(self, start_o, end_o, mins, maxs, mask):
        """Brushes submitted to a SWEPT trace (the +-0.25 of PM_GroundTrace)."""
        offset = tuple((mins[j] + maxs[j]) * 0.5 for j in range(3))
        size = tuple(maxs[j] - offset[j] for j in range(3))
        p1 = tuple(start_o[j] + offset[j] for j in range(3))
        p2 = tuple(end_o[j] + offset[j] for j in range(3))
        radius = size[2] if size[0] > size[2] else size[0]
        ro = (radius, radius, radius + (size[2] - radius))
        pmin = (min(p1[0], p2[0]), min(p1[1], p2[1]), min(p1[2], p2[2]))
        pmax = (max(p1[0], p2[0]), max(p1[1], p2[1]), max(p1[2], p2[2]))
        leafnums = self.trace_leafnums(p1, p2, size)
        if getattr(self, '_lb', None) is None:
            return self._bbox_filter(self._collect(leafnums, mask), pmin, pmax, ro)
        return self._collect_filtered(leafnums, pmin, pmax, ro)

    def _collect(self, leafnums, mask):
        lb = getattr(self, '_lb', None)
        if lb is None:                         # no cache: original path
            seen, out = set(), []
            for li in leafnums:
                for b in self.leaf_brushes(li):
                    if b in seen:
                        continue
                    seen.add(b)
                    if self.brushes[b][2] & mask:
                        out.append(b)
            return out
        seen, out = set(), []
        add = out.append
        for li in leafnums:
            for b in lb[li]:
                if b not in seen:
                    seen.add(b)
                    add(b)
        return out

    def _collect_filtered(self, leafnums, pmin, pmax, ro):
        """Collect + bbox rejection in ONE single pass (see `_bbox_filter`).

        Dedup via a MARKING ARRAY, not a `set`: `_stamp[b]` carries the
        number of the last call that saw brush `b`, and `_tok` increments on
        every call — no reset needed. A `set` cost 1,181,860 hashes here for
        5676 traces on `demolition` (41% of scan time); the array avoids the
        hashing.

        The returned list is UNCHANGED, order included: same walk, same bbox
        test, same first-occurrence criterion. Verified on 3000 traces of
        `demolition` (strict identity), then by a full scan of `mp_farmhouse`
        before/after, identical output line by line.

        Duplicates are real (12.2% of entries on `demolition`): the dedup
        cannot simply be dropped. `trace_leafnums`, however, never returns a
        duplicate leaf (0.0% on 51610), so there is nothing to gain there."""
        lb = self._lb
        lbz = self._lbz
        B = self.brushes
        stamp = self._stamp
        self._tok += 1
        tok = self._tok
        out = []
        add = out.append
        r0, r1, r2 = ro
        x0, y0, z0 = pmin
        x1, y1, z1 = pmax
        zmax = z1 + r2
        for li in leafnums:
            brs = lb[li]
            # `_lb` is sorted by mins.z: any brush past z1+r2 fails the Z test
            # below anyway, so the walk is cut short in advance. Decisive on
            # maps with a flat tree (mp_downtown: 527 leafs for 9725 brushes,
            # 282 entries walked per trace).
            fin = bisect_right(lbz[li], zmax)
            for b in brs[:fin]:
                if stamp[b] == tok:
                    continue
                stamp[b] = tok
                e = B[b]
                mn = e[0]
                mx = e[1]
                if x0 > mx[0] + r0 or x1 < mn[0] - r0:
                    continue
                if y0 > mx[1] + r1 or y1 < mn[1] - r1:
                    continue
                if z0 > mx[2] + r2 or z1 < mn[2] - r2:
                    continue
                add(b)
        return out

    def _bbox_filter(self, nums, pmin, pmax, ro):
        """Discards brushes whose bbox, expanded by `ro`, does not intersect
        the segment's box. This is EXACTLY the rejection performed by
        CM_TraceThroughBrush's axial loop (`d1 > 0` on an axial plane <=>
        the center is outside the expanded bbox), in a conservative version:
        the epsilon is ignored, so a bit more is kept. No false negative
        possible."""
        B = self.brushes
        out = []
        add = out.append
        r0, r1, r2 = ro
        x0, y0, z0 = pmin
        x1, y1, z1 = pmax
        for b in nums:
            e = B[b]
            mn = e[0]
            mx = e[1]
            if x0 > mx[0] + r0 or x1 < mn[0] - r0:
                continue
            if y0 > mx[1] + r1 or y1 < mn[1] - r1:
                continue
            if z0 > mx[2] + r2 or z1 < mn[2] - r2:
                continue
            add(b)
        return out


    def column_z(self, brushnum, x, y, hz, margin=2.0):
        """Z intervals where the player box, placed at (x, y, z), can TOUCH
        the brush, and where it is ENTIRELY BURIED inside it.

        Returns (zlo, zhi, zlo_in, zhi_in); None if the column does not touch.

        Separating-axis test on the brush's normals only: this is
        CONSERVATIVE (it can wrongly say "touches", never the reverse), so
        the generated set remains a SUPERSET of the triggering positions. No
        spot can be lost by this pruning.

        The box is [-15,15] x [-15,15] x [0,hz] around the origin; its
        center is (x, y, z + hz/2) and its half-extents are (15, 15, hz/2).

        Being entirely buried implies all 26 deltas stay inside the brush,
        so `PM_CorrectAllSolid` fails: these positions never trigger and are
        removed (safety margin `margin`)."""
        hh = hz * 0.5
        zlo, zhi = -1e30, 1e30
        zlo_in, zhi_in = -1e30, 1e30
        for n, d in self.brush_planes(brushnum):
            n0, n1, n2 = n
            sup = 15.0 * (abs(n0) + abs(n1)) + hh * abs(n2)
            a = n0 * x + n1 * y + n2 * hh
            # intersection: n.C - sup <= d      ->  n2*z <= d + sup - a
            r = d + sup - a
            # buried    : n.C + sup <= d - margin -> n2*z <= d - sup - margin - a
            r_in = d - sup - margin - a
            if n2 > 1e-9:
                zhi = min(zhi, r / n2)
                zhi_in = min(zhi_in, r_in / n2)
            elif n2 < -1e-9:
                zlo = max(zlo, r / n2)
                zlo_in = max(zlo_in, r_in / n2)
            else:
                if r < 0.0:
                    return None
                if r_in < 0.0:
                    zlo_in, zhi_in = 1e30, -1e30
        if zlo > zhi:
            return None
        return zlo, zhi, zlo_in, zhi_in

    def planes_of(self, brushnums):
        """Plane lists ready for `cm_faithful.player_trace`."""
        pl = getattr(self, '_pl', None)
        if pl is None:
            return [self.brush_planes(b) for b in brushnums]
        return [pl[b] for b in brushnums]

    def brush_planes(self, brushnum):
        """6 axial planes (from mins/maxs) + the non-axial ones, as
        `cm_faithful.brush_planes()` expects them."""
        mins, maxs, _, planes = self.brushes[brushnum]
        axial = [((1.0, 0.0, 0.0), maxs[0]), ((-1.0, 0.0, 0.0), -mins[0]),
                 ((0.0, 1.0, 0.0), maxs[1]), ((0.0, -1.0, 0.0), -mins[1]),
                 ((0.0, 0.0, 1.0), maxs[2]), ((0.0, 0.0, -1.0), -mins[2])]
        return axial + planes


if __name__ == '__main__':
    cm = ClipMap()
    print("map       : %s" % cm.name)
    print("nodes %d  leafs %d  lbn %d  brushes %d"
          % (len(cm.nodes), len(cm.leafs), len(cm.lbn), len(cm.brushes)))
    # walk non-regression checks (values measured on the engine)
    allmap = cm.box_leafnums((-6000., -6000., -500.), (5000., 6000., 1500.))
    print("box_leafnums(whole map) = %d leafs   [expected 1589]" % len(allmap))
    b = cm.box_leafnums((269.871063, 91.856026, 200.125),
                        (301.871063, 123.856026, 272.125))
    print("box_leafnums(delta box) = %s   [expected [267]]" % b)
