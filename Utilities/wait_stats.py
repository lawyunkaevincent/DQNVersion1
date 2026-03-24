from __future__ import annotations
from pathlib import Path
import numpy as np
import re
import xml.etree.ElementTree as ET

def _xml_parse_waits(xml_text: str) -> tuple[list[float], list[str]]:
    waits, ids = [], []
    # Try XML parse first
    try:
        root = ET.fromstring(xml_text)
        for tag in ("personinfo", "tripinfo"):
            for el in root.findall(f".//{tag}"):
                wt = el.get("waitingTime")
                pid = el.get("id")
                if wt is None:
                    continue
                waits.append(float(wt))
                ids.append(pid)
        if waits:
            return waits, ids
    except Exception:
        pass
    # Regex fallback (fixed escapes)
    for tag in ("personinfo", "tripinfo"):
        for m in re.findall(rf"<{tag}\b[^>]*?>", xml_text, flags=re.IGNORECASE|re.DOTALL):
            m_w = re.search(r'waitingTime="([\d.]+)"', m)
            m_i = re.search(r'id="([^"]+)"', m)
            if m_w:
                waits.append(float(m_w.group(1)))
                ids.append(m_i.group(1) if m_i else None)
    return waits, ids

def compute_wait_stats(xml_path: str | Path):
    xml_path = Path(xml_path)
    text = xml_path.read_text(encoding="utf-8", errors="ignore")
    waits, ids = _xml_parse_waits(text)
    arr = np.array(waits, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), arr, ids
    avg = float(np.mean(arr))
    # If you’re on older NumPy (<1.22), use interpolation='linear' instead of method=
    p95 = float(np.percentile(arr, 98, method="linear"))
    return avg, p95, arr, ids

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="Compute waiting-time stats from SUMO tripinfos.xml")
    ap.add_argument("xml", help="Path to tripinfos.xml")
    args = ap.parse_args()
    avg, p95, series, ids = compute_wait_stats(args.xml)
    out = {
        "avg_wait_seconds": avg,
        "p95_wait_seconds": p95,
        "num_passengers": int(series.size),
    }
    print(json.dumps(out, indent=2))
