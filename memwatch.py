import re
import subprocess


def memory_mb(pids):
    """Total RSS in MB of the given process IDs, via tasklist (no psutil).

    `pids`: an iterable of process IDs. Pass the scan's own processes -- the
    parent plus every worker. `None` falls back to summing every python.exe on
    the machine, which is what this function used to do unconditionally; that
    reading is inflated by any unrelated Python process the user happens to be
    running, so it can trip a memory ceiling that the scan alone would never
    reach. Prefer the explicit PID set.

    Only the listed processes are counted, never their children. The workers
    do not spawn any, so the sum is exact -- but it would silently stop being
    exact if that ever changed.

    Returns 0 when tasklist is unavailable (any non-Windows system), which
    disables the ceiling rather than aborting the run.
    """
    wanted = None if pids is None else {int(p) for p in pids}
    # Filtering by image name is only needed for the `None` path. When PIDs are
    # given it would just make the reading fragile: an interpreter launched as
    # py.exe or pythonw.exe would be missed, silently reporting 0 and disabling
    # the ceiling.
    cmd = ['tasklist', '/FO', 'CSV', '/NH']
    if wanted is None:
        cmd[1:1] = ['/FI', 'IMAGENAME eq python.exe']
    try:
        o = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode('latin-1')
    except Exception:
        return 0
    total = 0
    for l in o.splitlines():
        m = re.findall(r'"([^"]*)"', l)
        # tasklist CSV columns: name, PID, session name, session #, mem usage
        if len(m) >= 5:
            if wanted is not None:
                try:
                    if int(re.sub(r'[^\d]', '', m[1]) or -1) not in wanted:
                        continue
                except ValueError:
                    continue
            total += int(re.sub(r'[^\d]', '', m[4]) or 0)
    return total // 1024
