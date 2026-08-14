"""
ClipMap built PURELY from the .d3dbsp (bsp_read + bsp_ext), no memory dump,
pluggable into elevator_detector.Detector as-is.

Subclass of cm_leafs.ClipMap: entirely skips _load() (which reads a text
dump) and fills nodes/brushes directly from bsp_ext.load_ext(). leaf_brushes()
is overridden to read the flat LUMP_LEAFBRUSHES slice instead of descending
an LBN (which does not exist on disk) — a reduction proven by set equality in
validate_lbn_reduction.py (1607/1607 leafs mp_farmhouse, 1412/1412 leafs
tankhunt, 0 mismatch).

self.leafs is ACTUALLY FILLED (cf. _build_leafs), not just sized for its
length: the reference's build_cache() iterates `range(len(self.leafs))`, but
characterize.leafs_overlapping also reads the bounds. Bounds reconstructed like
the engine (union of the leaf's brushes' AABBs, inflated by
SURFACE_CLIP_EPSILON) -- so CONTENT bounds, not partition cells.
"""
import os

from cm_leafs import ClipMap as MemClipMap
import bsp_ext


class DiskClipMap(MemClipMap):
    def __init__(self, bsp_path):
        # do NOT call MemClipMap.__init__ (it would read a text dump)
        self.name = os.path.basename(bsp_path)
        ext = bsp_ext.load_ext(bsp_path)
        self._ext = ext

        self.nodes = ext['nodes']
        self.lbn = []  # unused (leaf_brushes is overridden)

        self.brushes = []
        for b in ext['brushes']:
            mn, mx = self._bbox(b)
            contents = ext['materials'][b['shader_num']]['content_flags'] \
                if 0 <= b['shader_num'] < len(ext['materials']) else 0
            planes = [(s['normal'], s['dist']) for s in b['sides'][6:]]
            self.brushes.append((mn, mx, contents, planes))

        self.leafs = self._build_leafs(len(ext['leaf_slices']))

    def _build_leafs(self, count):
        """Leaf bounds reconstructed the way the ENGINE does at load time.

        They do NOT exist in the .d3dbsp: `CMod_PartionLeafBrushes`
        (`cm_load_obj.cpp:659-707`) computes them on the fly as the UNION of
        the leaf's brushes' AABBs, then writes them INFLATED by
        SURFACE_CLIP_EPSILON (`:698-702`). That is reproduced here, exactly.

        /!\\ These are CONTENT bounds, not partition cells: two leafs can
        perfectly well overlap.

        Format aligned with cm_leafs.ClipMap: (mins, maxs, brushContents,
        leafBrushNode). The 4th field is never read here, `leaf_brushes`
        being overridden; it is 0. A brushless leaf gets degenerate bounds
        (mins > maxs), which no overlap test satisfies.
        """
        EPS = 0.125  # SURFACE_CLIP_EPSILON (cm_local.h:20)
        INF = float('inf')
        out = []
        for li in range(count):
            mn = [INF] * 3
            mx = [-INF] * 3
            contents = 0
            for bi in self.leaf_brushes(li):
                bmn, bmx, bct, _ = self.brushes[bi]
                contents |= bct
                for a in range(3):
                    if bmn[a] < mn[a]:
                        mn[a] = bmn[a]
                    if bmx[a] > mx[a]:
                        mx[a] = bmx[a]
            if mn[0] is INF or mn[0] == INF:
                out.append(((INF, INF, INF), (-INF, -INF, -INF), 0, 0))
                continue
            out.append((tuple(v - EPS for v in mn),
                        tuple(v + EPS for v in mx), contents, 0))
        return out

    @staticmethod
    def _bbox(brush):
        mins = [None, None, None]
        maxs = [None, None, None]
        for local_idx in range(6):
            side = brush['sides'][local_idx]
            axis = local_idx // 2
            if local_idx % 2 == 0:
                mins[axis] = -side['dist']
            else:
                maxs[axis] = side['dist']
        return tuple(mins), tuple(maxs)

    def leaf_brushes(self, leafnum):
        return bsp_ext.leaf_brushes_flat(self._ext, leafnum)
