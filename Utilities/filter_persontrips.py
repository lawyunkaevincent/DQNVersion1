
#!/usr/bin/env python3
"""Filter SUMO person trips by presence/absence of <ride>.

Usage:
  python filter_persontrips.py input.xml [input2.xml ...] [-o OUTDIR] [--keep with-ride|no-ride]

Defaults:
  --keep with-ride   (drop persons that have no <ride>)
  -o same directory as input, filename gets suffixed with .filtered

Examples:
  python filter_persontrips.py routes.rou.xml
  python filter_persontrips.py routes.rou.xml --keep no-ride -o filtered/
"""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import sys

def has_ride(person_el):
    # Child tags might have namespace, but in these files they don't.
    for child in list(person_el):
        tag = child.tag
        if tag.endswith('ride'):
            return True
    return False

def filter_file(in_path: Path, out_dir: Path, keep: str):
    tree = ET.parse(in_path)
    root = tree.getroot()

    # SUMO routes root is usually <routes ...>
    if root.tag.endswith('routes') is False:
        print(f"[WARN] {in_path}: Root tag is {root.tag}, expected 'routes'")

    kept = 0
    dropped = 0
    for person in list(root.findall('person')):
        ride_present = has_ride(person)
        # keep with-ride => drop if not ride
        should_keep = ride_present if keep == 'with-ride' else (not ride_present)
        if should_keep:
            kept += 1
        else:
            root.remove(person)
            dropped += 1

    # Prepare output path
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = in_path.stem + ('.with_ride' if keep=='with-ride' else '.no_ride') + in_path.suffix
    out_path = out_dir / out_name

    # Preserve XML declaration and write
    ET.register_namespace('', "http://sumo.dlr.de/xsd/routes_file.xsd")  # harmless if not used
    tree.write(out_path, encoding='utf-8', xml_declaration=True)
    return kept, dropped, out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', nargs='+', help='Input SUMO routes/persontrips XML files')
    ap.add_argument('-o', '--outdir', default=None, help='Output directory (default: same as input)')
    ap.add_argument('--keep', choices=['with-ride', 'no-ride'], default='with-ride',
                    help='Keep persons that have a ride (default) or those with no ride')
    args = ap.parse_args()

    total_kept = total_dropped = 0
    out_paths = []
    for inp in args.inputs:
        in_path = Path(inp)
        if not in_path.exists():
            print(f"[ERROR] Input not found: {inp}", file=sys.stderr)
            continue
        out_dir = Path(args.outdir) if args.outdir else in_path.parent
        kept, dropped, out_path = filter_file(in_path, out_dir, args.keep)
        total_kept += kept
        total_dropped += dropped
        out_paths.append((in_path, out_path, kept, dropped))

    # Summary
    for in_path, out_path, kept, dropped in out_paths:
        print(f"[OK] {in_path.name} -> {out_path.name} | kept={kept}, dropped={dropped}")
    print(f"[SUMMARY] Files processed: {len(out_paths)}, total kept={total_kept}, total dropped={total_dropped}")

if __name__ == '__main__':
    main()


