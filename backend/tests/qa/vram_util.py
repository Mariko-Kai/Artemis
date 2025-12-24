import subprocess
import re
import logging

logger = logging.getLogger(__name__)

def get_vram_usage():
    """Returns (used, total) in MB using nvidia-smi."""
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"], encoding="utf-8")
        used, total = map(int, output.strip().split(","))
        return used, total
    except Exception as e:
        logger.error(f"Failed to get VRAM usage: {e}")
        return 0, 0

def check_vram_safe(threshold_mb=3800):
    """Checks if VRAM usage is below threshold."""
    used, total = get_vram_usage()
    if used > threshold_mb:
        logger.warning(f"VRAM usage HIGH: {used}MB / {total}MB")
        return False
    return True
