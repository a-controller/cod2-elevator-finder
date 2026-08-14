"""Minimal read-only parser for CoD2 .d3dbsp (IBSP v4) — planes/brushsides/brushes only.

Format derived from external/cod2map-master/cod2src/cod2map.h and writebsp.c
(EmitBrushes), not assumed from Q3. Key facts from the compiler source:

- Header: magic(int)='IBSP' (0x50534249), version(int)=4, then 39 lump
  entries of {size:int, offset:int} (cod2map.h:658-669).
- LUMP_PLANES=4: {normal[3]:float, dist:float} = 16 bytes/entry.
- LUMP_BRUSHSIDES=5: {union{planeNum:int, dist:float}, shaderNum:int} = 8 bytes/entry.
- LUMP_BRUSHES=6: {numSides:short, shaderNum:short} = 4 bytes/entry, no
  firstSide field — a brush's sides are contiguous, offset = cumulative sum
  of numSides of preceding brushes (writebsp.c:149,161).
- Each brush's first BRUSH_AXIAL_SIDES=6 sides (cod2map.h:298) are the
  implicit bbox planes: normal is implied by local side index
  (0/1=X, 2/3=Y, 4/5=Z; even=negative axis, odd=positive axis), and the
  union stores `dist` directly, already signed so that the half-space is
  `dot(normal, point) <= dist` (writebsp.c:163-178). Sides from index 6
  onward store a real `planeNum` into LUMP_PLANES instead.
"""

import struct

BSP_IDENT = 0x50534249
BSP_VERSION = 4
BSP_LUMP_COUNT = 39

LUMP_MATERIALS = 0
LUMP_PLANES = 4
LUMP_BRUSHSIDES = 5
LUMP_BRUSHES = 6

BRUSH_AXIAL_SIDES = 6
ON_EPSILON = 0.1  # cod2map.h:163, for later steps

AXIAL_NORMALS = [
    (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, -1.0), (0.0, 0.0, 1.0),
]


def load(path):
    """Parse a .d3dbsp file. Returns {'planes': [...], 'brushes': [...], 'materials': [...]}.

    planes[i] = (nx, ny, nz, dist)

    materials[i] = {'name': str, 'surface_flags': int, 'content_flags': int}
    (Dmaterial_t, cod2map.h:932-937). A brush's own contentFlags — used by
    FindBrushNeighbors (brush.c:1111) to require matching contentFlags
    between a candidate donor and its receiver — is
    materials[brush['shader_num']]['content_flags'].

    brushes[i] = {
        'shader_num': int,       # brush-level material index (LUMP_MATERIALS, not resolved here)
        'num_sides': int,
        'sides': [
            {
                'normal': (nx, ny, nz),
                'dist': float,          # half-space: dot(normal, point) <= dist
                'shader_num': int,      # side material index (LUMP_MATERIALS, not resolved here)
                'axial': bool,
                'plane_idx': int|None,  # index into planes[], only for non-axial sides
            }, ...
        ],
    }
    """
    with open(path, 'rb') as f:
        data = f.read()

    magic, version = struct.unpack_from('<ii', data, 0)
    if magic != BSP_IDENT:
        raise ValueError(f"{path}: bad magic 0x{magic:08x} (expected 0x{BSP_IDENT:08x})")
    if version != BSP_VERSION:
        raise ValueError(f"{path}: bad version {version} (expected {BSP_VERSION})")

    lumps = []
    off = 8
    for _ in range(BSP_LUMP_COUNT):
        size, offset = struct.unpack_from('<ii', data, off)
        lumps.append((size, offset))
        off += 8

    def lump_bytes(idx):
        size, offset = lumps[idx]
        return data[offset:offset + size]

    materials_raw = lump_bytes(LUMP_MATERIALS)
    n_materials = len(materials_raw) // 72  # Dmaterial_t: char[64] + int surfaceFlags + int contentFlags
    materials = []
    for i in range(n_materials):
        off_m = i * 72
        name = materials_raw[off_m:off_m + 64].split(b'\x00', 1)[0].decode('ascii', 'replace')
        surface_flags, content_flags = struct.unpack_from('<ii', materials_raw, off_m + 64)
        materials.append({'name': name, 'surface_flags': surface_flags, 'content_flags': content_flags})

    planes_raw = lump_bytes(LUMP_PLANES)
    n_planes = len(planes_raw) // 16
    planes = [struct.unpack_from('<ffff', planes_raw, i * 16) for i in range(n_planes)]

    sides_raw = lump_bytes(LUMP_BRUSHSIDES)
    n_sides = len(sides_raw) // 8

    brushes_raw = lump_bytes(LUMP_BRUSHES)
    n_brushes = len(brushes_raw) // 4
    brush_hdrs = [struct.unpack_from('<hh', brushes_raw, i * 4) for i in range(n_brushes)]

    brushes = []
    side_cursor = 0
    for num_sides, shader_num in brush_hdrs:
        brush_sides = []
        for local_idx in range(num_sides):
            gi = side_cursor + local_idx
            if gi >= n_sides:
                raise ValueError(f"brush side index {gi} out of range (n_sides={n_sides})")
            raw = sides_raw[gi * 8: gi * 8 + 8]
            side_shader_num = struct.unpack_from('<i', raw, 4)[0]

            if local_idx < BRUSH_AXIAL_SIDES:
                raw_dist = struct.unpack_from('<f', raw, 0)[0]
                # On-disk value is the raw axis coordinate (min for even, max
                # for odd). Negative-axis (even) sides need the sign flipped
                # to get a proper dot(normal, point) <= dist half-space,
                # consistent with non-axial sides. Verified empirically
                # against CoD2\maps\mp_farmhouse.map brush 1 ground truth.
                dist = -raw_dist if local_idx % 2 == 0 else raw_dist
                normal = AXIAL_NORMALS[local_idx]
                brush_sides.append({
                    'normal': normal, 'dist': dist,
                    'shader_num': side_shader_num, 'axial': True, 'plane_idx': None,
                })
            else:
                plane_idx = struct.unpack_from('<i', raw, 0)[0]
                if not (0 <= plane_idx < n_planes):
                    raise ValueError(f"brush side {gi}: plane_idx {plane_idx} out of range (n_planes={n_planes})")
                nx, ny, nz, dist = planes[plane_idx]
                brush_sides.append({
                    'normal': (nx, ny, nz), 'dist': dist,
                    'shader_num': side_shader_num, 'axial': False, 'plane_idx': plane_idx,
                })
        brushes.append({
            'shader_num': shader_num,
            'num_sides': num_sides,
            'sides': brush_sides,
        })
        side_cursor += num_sides

    return {'planes': planes, 'brushes': brushes, 'materials': materials}


def brush_bbox(brush):
    """Bounding box (mins, maxs) derived from the brush's 6 implicit axial sides."""
    mins = [None, None, None]
    maxs = [None, None, None]
    for local_idx in range(BRUSH_AXIAL_SIDES):
        side = brush['sides'][local_idx]
        axis = local_idx // 2
        if local_idx % 2 == 0:
            mins[axis] = -side['dist']
        else:
            maxs[axis] = side['dist']
    return tuple(mins), tuple(maxs)


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        print("usage: bsp_read.py <file.d3dbsp>")
        sys.exit(1)

    bsp = load(sys.argv[1])
    print(f"planes: {len(bsp['planes'])}")
    print(f"brushes: {len(bsp['brushes'])}")
    n_axial_only = sum(1 for b in bsp['brushes'] if b['num_sides'] <= BRUSH_AXIAL_SIDES)
    print(f"brushes with only 6 axial sides (pure box, no bevel): {n_axial_only}")
