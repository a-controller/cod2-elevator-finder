r"""
Extends bsp_read.py to the NODES/LEAFS/LEAFBRUSHES lumps.

Separate file: bsp_read.py stays read-only. It already provides
planes/brushes/materials via bsp_read.load(); this module adds parsing for
the 3 remaining lumps.

Layouts verified against external/cod2map-master/cod2src (not guessed):

- LUMP_NODES=25: BspNode_disk_t (cod2map.h:678-683), 36 bytes/entry:
    int planeNum; int children[2]; int mins[3]; int maxs[3];
  mins/maxs are floats TRUNCATED to int on write (writebsp.c:424-429), not
  bit-patterns. No `type` field on disk: the memory dump derives it from the
  resolved plane's normal via planeNum (cardinal axis -> 0/1/2, else 3) —
  reproduced here by axial_type().

- LUMP_LEAFS=26: BspLeaf_disk_t (cod2map.h:686-696), 36 bytes/entry:
    int cluster, area, firstCollisionAABB, numCollisionAABBs,
        firstLeafBrush, numLeafBrushes, cellnum, reserved28, reserved32;
  No mins/maxs, no leafBrushNode on disk — and cm_leafs.py (the reference
  Python port, `grep 'self.leafs\['`) only ever reads the leafBrushNode
  field (an index into the LBN the engine builds at runtime). mins/maxs/
  brushContents are used by NO function of the port; no need to reconstruct
  them.

- LUMP_LEAFBRUSHES=27: flat array of `int` (cod2map.h:3383
  `extern int bspLeafBrushes[...]`), filled in EmitLeaf (writebsp.c:369-380)
  as a contiguous slice [firstLeafBrush : firstLeafBrush+numLeafBrushes].

REDUCTION, DO NOT READ AS "FAITHFUL TO THE ENGINE":
The LBN (LeafBrushNode) from the memory dump is an acceleration structure
the engine builds at runtime from this same flat array, to spatially prune
(axis/dist/range) a leaf's brushes before testing them. This structure does
NOT exist on disk and is NOT reconstructed here. What IS reconstructed is
the reference Python port (cm_leafs.ClipMap.leaf_brushes), which does NO
spatial pruning when descending the LBN: it collects the whole subtree
unfiltered (its own comment: "collects the ENTIRE subtree"). The superset it
produces is, by construction, the union of every 'L' leaf of the LBN for
that leaf — exactly the set given by the flat slice
LUMP_LEAFBRUSHES[first:first+num]. That is a SET equality verified
empirically (cf validate_lbn_reduction.py), not a format property assumed
without proof.

Consequence: `leaf_brushes_flat()` below reproduces the PYTHON PORT (our
reference, validated by selftest + production runs), not the engine, which
prunes spatially and never visits the full LBN subtree for a given trace.
Do not present this module as "exact to the engine".
"""
import struct
import sys

import bsp_read

LUMP_NODES = 25
LUMP_LEAFS = 26
LUMP_LEAFBRUSHES = 27


def _lump_table(data):
    magic, version = struct.unpack_from('<ii', data, 0)
    if magic != bsp_read.BSP_IDENT or version != bsp_read.BSP_VERSION:
        raise ValueError("bad magic/version")
    lumps = []
    off = 8
    for _ in range(bsp_read.BSP_LUMP_COUNT):
        size, offset = struct.unpack_from('<ii', data, off)
        lumps.append((size, offset))
        off += 8
    return lumps


def axial_type(normal, eps=1e-6):
    """Plane type as derived by the memory dump: cardinal axis -> 0/1/2
    (X/Y/Z), else 3. Reproduces the classification used by
    box_on_plane_side (ptype < 3 = fast axial path)."""
    ax, ay, az = abs(normal[0]), abs(normal[1]), abs(normal[2])
    if abs(ax - 1.0) < eps and ay < eps and az < eps:
        return 0
    if abs(ay - 1.0) < eps and ax < eps and az < eps:
        return 1
    if abs(az - 1.0) < eps and ax < eps and ay < eps:
        return 2
    return 3


def load_ext(path):
    """Returns {'planes','brushes','materials'} (bsp_read.load) plus
    'nodes' (normal,dist,type,child0,child1), 'leaf_slices'
    [(firstLeafBrush,numLeafBrushes), ...], and 'leafbrushes' (flat array)."""
    base = bsp_read.load(path)

    with open(path, 'rb') as f:
        data = f.read()
    lumps = _lump_table(data)

    def lump_bytes(idx):
        size, offset = lumps[idx]
        return data[offset:offset + size]

    planes = base['planes']

    nodes_raw = lump_bytes(LUMP_NODES)
    n_nodes = len(nodes_raw) // 36
    nodes = []
    for i in range(n_nodes):
        planeNum, c0, c1 = struct.unpack_from('<3i', nodes_raw, i * 36)
        nx, ny, nz, dist = planes[planeNum]
        ptype = axial_type((nx, ny, nz))
        nodes.append(((nx, ny, nz), dist, ptype, c0, c1))

    leafs_raw = lump_bytes(LUMP_LEAFS)
    n_leafs = len(leafs_raw) // 36
    leaf_slices = []
    for i in range(n_leafs):
        vals = struct.unpack_from('<9i', leafs_raw, i * 36)
        firstLeafBrush, numLeafBrushes = vals[4], vals[5]
        leaf_slices.append((firstLeafBrush, numLeafBrushes))

    lb_raw = lump_bytes(LUMP_LEAFBRUSHES)
    n_lb = len(lb_raw) // 4
    leafbrushes = list(struct.unpack_from('<%di' % n_lb, lb_raw, 0)) if n_lb else []

    base['nodes'] = nodes
    base['leaf_slices'] = leaf_slices
    base['leafbrushes'] = leafbrushes
    return base


def leaf_brushes_flat(bsp_ext, leafnum):
    first, num = bsp_ext['leaf_slices'][leafnum]
    return bsp_ext['leafbrushes'][first:first + num]


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("usage: bsp_ext.py <file.d3dbsp>")
        sys.exit(1)
    b = load_ext(sys.argv[1])
    print(f"nodes: {len(b['nodes'])}")
    print(f"leafs (leaf_slices): {len(b['leaf_slices'])}")
    print(f"leafbrushes (flat): {len(b['leafbrushes'])}")
