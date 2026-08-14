--[[----------------------------------------------------------------------
  dump_clipmap.lua — exports CoD2 collision geometry to a text file readable
  by detect.py

  HOW TO USE
    1. Start CoD2MP_s.exe and load the map with `devmap <map>`, so you are
       dumping from your own offline session.
    2. In Cheat Engine, attach to the CoD2MP_s.exe process.
    3. Table -> Show Cheat Table Lua Script, paste this file, Execute.
    4. The dump is written to OUT_DIR .. <map_name> .. ".txt"

  Set OUT_DIR below before running: Cheat Engine's working directory is not
  predictable, so a relative path would land somewhere unexpected.

  This script is READ-ONLY: it never writes to the game's memory.

  IMPORTANT — the address of `cm` changes with every build and every launch.
  It is located automatically below. If auto-detection fails, set CM_ADDR by
  hand (see README).

  Why this path matters: reading the collision data from memory is the
  accurate source. Reading the compiled .d3dbsp instead is supported by the
  scanner but under-detects, because the compiler adds bevel planes and
  adjusts brush bounds — see the README for measured figures.
------------------------------------------------------------------------]]

-- Absolute path of the directory where dumps are written, with a trailing
-- separator. Must exist. This is the `maps/` directory the scanner reads,
-- i.e. the one given by --maps-dir / COD2_MAPS_DIR.
local OUT_DIR = [[C:\path\to\your\maps\]]

-- clipMap_t offsets (32-bit build), verified against memory
local O_NUM_NODES, O_NODES   = 0x1C, 0x20
local O_NUM_LEAFS, O_LEAFS   = 0x24, 0x28
local O_LBN_COUNT, O_LBN     = 0x2C, 0x30
local O_NUM_BRUSH, O_BRUSHES = 0x7C, 0x80

--- Validates a candidate `&cm`: the counters must be plausible AND the plane
--- normal of node 0 must be a unit vector.
---
--- This check is required: matching `cm.name` alone is NOT enough. On
--- mp_trainstation the string "maps/mp/....d3dbsp" is referenced from 7 places
--- in the module; 6 of them yield nonsensical counters (numNodes ~ 3.9e9) and
--- crash the read.
local function plausible(cm)
  local numNodes = readInteger(cm + O_NUM_NODES)
  local nodes    = readInteger(cm + O_NODES)
  local numLeafs = readInteger(cm + O_NUM_LEAFS)
  local leafs    = readInteger(cm + O_LEAFS)
  local lbnCount = readInteger(cm + O_LBN_COUNT)
  local lbn      = readInteger(cm + O_LBN)
  local numBrush = readSmallInteger(cm + O_NUM_BRUSH)
  local brushes  = readInteger(cm + O_BRUSHES)
  if not (numNodes and nodes and numLeafs and leafs
          and lbnCount and lbn and numBrush and brushes) then return false end
  -- counters: a few thousand, never zero
  if numNodes < 1 or numNodes > 65535 then return false end
  if numLeafs < 1 or numLeafs > 65535 then return false end
  if lbnCount < 1 or lbnCount > 500000 then return false end
  if numBrush < 1 then return false end
  -- pointers: on the heap, not inside the module
  for _, p in ipairs({nodes, leafs, lbn, brushes}) do
    if p < 0x00100000 or p > 0x7FFFFFFF then return false end
  end
  -- decisive test: the plane of node 0 must have a unit normal
  local pl = readInteger(nodes)
  if not pl or pl < 0x00100000 then return false end
  local x, y, z = readFloat(pl), readFloat(pl + 4), readFloat(pl + 8)
  if not (x and y and z) then return false end
  local n = x * x + y * y + z * z
  if n < 0.98 or n > 1.02 then return false end
  return true
end

--- Host module: multiplayer or singleplayer. SP maps do not load in the MP
--- client, so both binaries must be supported.
local function findModule()
  for _, m in ipairs({"CoD2MP_s.exe", "CoD2SP_s.exe", "CoD2MP.exe", "CoD2SP.exe"}) do
    local ok, base = pcall(getAddress, m)
    if ok and base and base > 0 then
      return m, base, (getModuleSize(m) or 0x1000000)
    end
  end
  return nil
end

--- Gives Cheat Engine's main thread a chance to pump its message queue.
---
--- This matters on large maps. The script runs ON that thread, so a long loop
--- freezes CE completely; past a few seconds Windows marks it Not Responding
--- and swaps in a ghost window, which then vanishes -- the script finishes fine
--- but its output is gone with the window. Call this every few thousand
--- iterations and CE stays alive throughout.
local function pump()
  if processMessages then processMessages() end
end

--- Tests one candidate `a` == &cm.brushes.
local function consider(a, v, ok)
  if v <= 0x10000000 then return false end
  local nm = readInteger(a - O_BRUSHES)             -- cm.name if a == &cm.brushes
  if not nm or nm <= 0x10000 then return false end
  -- 64, not 32: "maps/mp/jm_ladder_hell_easy.d3dbsp" is 34 characters. Truncated
  -- at 32 the ".d3dbsp" suffix is cut off, the test fails and `cm` is never
  -- found -- a silent, map-dependent failure reporting zero candidates.
  local s = readString(nm, 64)
  if not (s and s:find("maps/") == 1 and s:find("%.d3dbsp")) then return false end
  local cm = a - O_BRUSHES
  if plausible(cm) then ok[#ok+1] = cm return false end
  return true                                        -- counted as rejected
end

--- Locates `&cm`: a pointer to a cbrush_t array whose `cm.name` looks like
--- "maps/.../xxx.d3dbsp" AND whose structure is plausible.
---
--- The module is read in 64 KB blocks rather than one `readInteger` per address:
--- 22 MB / 4 is ~5.6 million cross-process reads, minutes of frozen UI, versus a
--- few hundred block reads recombined in Lua. Same addresses examined, same
--- results -- only the number of round trips changes.
local function findClipMap()
  local ok, rejected = {}, 0
  local mod, ms, sz = findModule()
  if not mod then
    print("FAILED: no CoD2 module found (neither MP nor SP).")
    return nil
  end
  print(string.format("module: %s (base 0x%08X, size 0x%X)", mod, ms, sz))

  local CHUNK  = 0x10000                             -- multiple of 4: keeps alignment
  local last   = ms + sz - 4
  local blocks = 0
  local a = ms
  while a <= last do
    local n = math.min(CHUNK, last - a + 4)
    local b = readBytes(a, n, true)                  -- table of bytes, or nil
    if b then
      for o = 1, n - 3, 4 do
        local b3 = b[o + 3]
        -- cheapest possible pre-filter: only the top byte decides whether the
        -- little-endian dword can exceed 0x10000000 at all.
        if b3 and b3 >= 0x10 then
          local v = b[o] | (b[o + 1] << 8) | (b[o + 2] << 16) | (b3 << 24)
          if consider(a + o - 1, v, ok) then rejected = rejected + 1 end
        end
      end
    else
      -- unreadable page inside the block: fall back address by address so a
      -- single bad page cannot hide `cm` behind a 64 KB hole.
      for x = a, a + n - 4, 4 do
        local v = readInteger(x)
        if v and consider(x, v, ok) then rejected = rejected + 1 end
      end
    end
    a = a + n
    blocks = blocks + 1
    if blocks % 8 == 0 then pump() end
  end
  if rejected > 0 then
    print(string.format("  %d candidate(s) rejected: nonsensical counters", rejected))
  end
  if #ok > 1 then
    print(string.format("  WARNING: %d plausible candidates, using the first", #ok))
    for i, c in ipairs(ok) do print(string.format("    #%d 0x%08X", i, c)) end
  end
  return ok[1]
end

local CM_ADDR = nil                        -- <- force the address here if needed
local cm = CM_ADDR or findClipMap()
if not cm then
  print("FAILED: could not locate clipMap_t. Set CM_ADDR by hand.")
  return
end

local name      = readString(readInteger(cm), 64)
local numNodes  = readInteger(cm + O_NUM_NODES)
local nodes     = readInteger(cm + O_NODES)
local numLeafs  = readInteger(cm + O_NUM_LEAFS)
local leafs     = readInteger(cm + O_LEAFS)
local lbnCount  = readInteger(cm + O_LBN_COUNT)
local lbn       = readInteger(cm + O_LBN)
local numBrush  = readSmallInteger(cm + O_NUM_BRUSH)
local brushes   = readInteger(cm + O_BRUSHES)

print(string.format("cm = 0x%08X  map = %s", cm, name))
print(string.format("nodes=%d leafs=%d lbn=%d brushes=%d",
                    numNodes, numLeafs, lbnCount, numBrush))

local short = name:match("([^/\\]+)%.d3dbsp") or "map"
local path  = OUT_DIR .. short .. ".txt"
local f, err = io.open(path, "w")
if not f then print("FAILED to open " .. path .. ": " .. tostring(err)) return end

local F = string.format
f:write(F("MAP %s\n", name))

-- cNode_t : 8 bytes { cplane_t *plane; int16 children[2] }
-- cplane_t : 20 bytes { vec3 normal; float dist; byte type; ... }
f:write(F("NODES %d\n", numNodes))
for i = 0, numNodes - 1 do
  local n  = nodes + i * 8
  local p  = readInteger(n)
  local c0 = readSmallInteger(n + 4); if c0 >= 32768 then c0 = c0 - 65536 end
  local c1 = readSmallInteger(n + 6); if c1 >= 32768 then c1 = c1 - 65536 end
  f:write(F("%d %.9g %.9g %.9g %.9g %d %d %d\n", i,
    readFloat(p), readFloat(p + 4), readFloat(p + 8), readFloat(p + 0x0C),
    readBytes(p + 0x10, 1, false), c0, c1))
  if i % 4096 == 0 then pump() end
end

-- cLeaf_t : 44 bytes, mins +0x0C, maxs +0x18, brushContents +0x04,
--           leafBrushNode +0x24
f:write(F("LEAFS %d\n", numLeafs))
for i = 0, numLeafs - 1 do
  local a = leafs + i * 44
  f:write(F("%d %.9g %.9g %.9g %.9g %.9g %.9g %d %d\n", i,
    readFloat(a + 0x0C), readFloat(a + 0x10), readFloat(a + 0x14),
    readFloat(a + 0x18), readFloat(a + 0x1C), readFloat(a + 0x20),
    readInteger(a + 4), readInteger(a + 0x24)))
  if i % 4096 == 0 then pump() end
end

-- cLeafBrushNode_t : 20 bytes, pack(2)
--   axis +0x00 | leafBrushCount int16 +0x02 | contents +0x04
--   union +0x08 : leaf{uint16 *brushes} | children{float dist, range; uint16 off[2]}
f:write(F("LBN %d\n", lbnCount))
for i = 0, lbnCount - 1 do
  local n   = lbn + i * 20
  local lbc = readSmallInteger(n + 2); if lbc >= 32768 then lbc = lbc - 65536 end
  local ct  = readInteger(n + 4)
  if lbc > 0 then
    local p, t = readInteger(n + 8), {}
    for k = 0, lbc - 1 do t[#t+1] = tostring(readSmallInteger(p + k * 2)) end
    f:write(F("%d L %d %d %s\n", i, lbc, ct, table.concat(t, " ")))
  else
    f:write(F("%d N %d %d %d %.9g %.9g %d %d\n", i, lbc, ct,
      readBytes(n, 1, false), readFloat(n + 8), readFloat(n + 0x0C),
      readSmallInteger(n + 0x10), readSmallInteger(n + 0x12)))
  end
  if i % 4096 == 0 then pump() end
end

-- cbrush_t : 48 bytes  mins +0x00 | contents +0x0C | maxs +0x10
--                      numsides +0x1C (non-axial only) | sides* +0x20
-- cbrushside_t : 8 bytes { cplane_t *plane; int materialNum }
f:write(F("BRUSHES %d\n", numBrush))
for i = 0, numBrush - 1 do
  local a  = brushes + i * 48
  local ns = readInteger(a + 0x1C)
  local sd = readInteger(a + 0x20)
  local t  = {}
  if ns and ns > 0 and ns < 256 and sd and sd > 0x1000000 then
    for k = 0, ns - 1 do
      local pp = readInteger(sd + k * 8)
      if pp and pp > 0x1000000 then
        t[#t+1] = F("%.9g %.9g %.9g %.9g",
          readFloat(pp), readFloat(pp + 4), readFloat(pp + 8), readFloat(pp + 0x0C))
      end
    end
  end
  f:write(F("%d %.9g %.9g %.9g %.9g %.9g %.9g %d %d %s\n", i,
    readFloat(a), readFloat(a + 4), readFloat(a + 8),
    readFloat(a + 0x10), readFloat(a + 0x14), readFloat(a + 0x18),
    readInteger(a + 0x0C), #t, table.concat(t, " ")))
  if i % 2048 == 0 then pump() end
end
f:close()

print("OK -> " .. path)

--[[ player tracemask ------------------------------------------------------
  detect.py defaults to MASK = 0x02810011, a value read from the MULTIPLAYER
  client. On another binary (singleplayer) it can differ, and a wrong mask
  silently invalidates EVERYTHING. It is recovered here through
  `pm->mins` = (-15,-15,0):
      pm->mins at pm+0xC4, pm->maxs at pm+0xD0, pm->tracemask at pm+0x3C
--------------------------------------------------------------------------]]
--- A valid collision mask contains CONTENTS_SOLID (bit 0) and has only a few
--- bits set. A false `pm` typically yields a COORDINATE reinterpreted as an
--- integer: no bit 0, and a dozen or so scattered bits.
local function maskPlausible(mk)
  if not mk or mk == 0 then return false end
  if (mk & 1) ~= 1 then return false end            -- CONTENTS_SOLID required
  local n = 0
  for i = 0, 31 do if (mk >> i) & 1 == 1 then n = n + 1 end end
  return n <= 12
end

-- This whole block runs AFTER the dump is closed, so a failure here costs only
-- the mask, never the dump. It is wrapped in pcall for exactly that reason: the
-- scan runs against live memory, its results shift from one run to the next, and
-- an unguarded error used to kill the script window with no message at all.
local maskOk, maskErr = pcall(function()
  local sig  = "00 00 70 C1 00 00 70 C1 00 00 00 00"  -- -15, -15, 0
  local hits = AOBScan(sig, "+W")
  if not hits or hits.getCount() == 0 then
    print("  tracemask: pm->mins pattern not found (be in-game, with an active player)")
    if hits then hits.destroy() end
    return
  end
  local seen, good = {}, {}
  for i = 0, math.min(hits.getCount() - 1, 60) do
    -- hits[i] can come back nil: the result list is a snapshot of memory that
    -- keeps moving under us. Skip, never dereference.
    local a = hits[i] and tonumber(hits[i], 16)
    if a then
      local pm = a - 0xC4
      local mk = readInteger(pm + 0x3C)
      local mz = readFloat(pm + 0xD8)               -- maxs.z : 70 / 50 / 30
      if mk and mz and (mz == 70 or mz == 50 or mz == 30) then
        local k = string.format("0x%08X", mk)
        if not seen[k] then
          seen[k] = true
          if maskPlausible(mk) then
            good[#good+1] = k
            print(string.format("  tracemask candidate %s  (pm=0x%08X, maxs.z=%.0f)", k, pm, mz))
          else
            print(string.format("  [rejected] 0x%08X: no SOLID bit, or too many bits"
                                .. " (pm=0x%08X) -- probably a coordinate", mk, pm))
          end
        end
      end
    end
  end
  hits.destroy()
  if #good == 1 then
    print("  -> MASK SELECTED: " .. good[1])
    if good[1] ~= "0x02810011" then
      print("  -> run detect.py with --mask " .. good[1])
    end
  elseif #good == 0 then
    print("  -> NO plausible mask found. Do not guess: report the anomaly.")
  else
    print(string.format("  -> %d plausible masks, AMBIGUOUS: report the anomaly.", #good))
  end
end)
if not maskOk then
  print("  tracemask: recovery FAILED (" .. tostring(maskErr) .. ")")
  print("  -> the dump above is complete and usable; only the mask is missing.")
end
