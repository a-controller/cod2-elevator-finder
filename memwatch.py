import re
import subprocess


def memory_mb(pid):
    """Total RSS of the process and its children, via tasklist (no psutil)."""
    try:
        o = subprocess.check_output(['tasklist', '/FI', 'IMAGENAME eq python.exe',
                                     '/FO', 'CSV', '/NH'],
                                    stderr=subprocess.DEVNULL).decode('latin-1')
    except Exception:
        return 0
    total = 0
    for l in o.splitlines():
        m = re.findall(r'"([^"]*)"', l)
        if len(m) >= 5:
            total += int(re.sub(r'[^\d]', '', m[4]) or 0)
    return total // 1024
