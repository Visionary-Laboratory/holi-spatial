#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diff Boxes3D keys between an existing rerun .rrd and a bbox json.

Why you may see "add" in replacement:
- json has (label, ins_id) keys that are not present as Boxes3D entities in the rrd.
- or the entity-path parsing fails because label contains path separators, etc.

This tool helps you inspect:
- json keys count
- rrd boxes keys count (from entity paths where Boxes3D exists)
- missing keys (json - rrd)
- extra keys (rrd - json)
- whitespace/special-char diagnostics for labels

Supports:
- single scene: --rrd + --json, or --scene with --old_dir/--new_json_dir
- batch scenes: iterate all .rrd in --old_dir, compare to json in --new_json_dir,
  and write a summary CSV + per-scene JSON reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import rerun.dataframe as rr_df


Key = Tuple[str, str]  # (label, ins_id)


def _iter_json_instances(obj: Any) -> Iterable[dict]:
    if isinstance(obj, list):
        for it in obj:
            if isinstance(it, dict):
                yield it
        return
    if isinstance(obj, dict):
        for k in ("instances", "objects", "annotations", "data"):
            v = obj.get(k)
            if isinstance(v, list):
                for it in v:
                    if isinstance(it, dict):
                        yield it
                return
    raise ValueError("Unsupported JSON layout: expected list or dict containing a list of instances.")


def load_json_keys(json_path: Path) -> Set[Key]:
    with json_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    keys: Set[Key] = set()
    for inst in _iter_json_instances(obj):
        label = str(inst.get("label", ""))
        ins_id = str(inst.get("ins_id", ""))
        if label and ins_id:
            keys.add((label, ins_id))
    return keys


def parse_instance_key_from_entity_path(entity_path: str) -> Optional[Key]:
    """
    Parse /.../instances/{label}/{ins_id}/bbox
    Robust to label containing '/', which becomes multiple segments.
    """
    parts = list(PurePosixPath(entity_path).parts)
    for i, p in enumerate(parts):
        if p != "instances":
            continue
        bbox_idx = None
        for j in range(i + 1, len(parts)):
            if parts[j] == "bbox":
                bbox_idx = j
        if bbox_idx is None:
            continue
        if bbox_idx - (i + 1) < 2:
            continue
        ins_id = str(parts[bbox_idx - 1])
        label_parts = [str(x) for x in parts[i + 1 : bbox_idx - 1]]
        if not label_parts:
            continue
        label = "/".join(label_parts)
        return (label, ins_id)
    return None


def rr_unescape_component(s: str) -> str:
    """
    Undo rerun path escaping for a component string.
    Example: 'cleaning\\ supplies' -> 'cleaning supplies'
    """
    out: List[str] = []
    i = 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            out.append(s[i + 1])
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def load_rrd_box_keys(rrd_path: Path) -> Tuple[Set[Key], List[str]]:
    """
    Returns: (keys, unparsed_entity_paths)
    """
    recs = rr_df.load_archive(str(rrd_path)).all_recordings()
    if not recs:
        return set(), []
    rec = recs[0]
    schema = rec.schema()
    keys: Set[Key] = set()
    unparsed: List[str] = []
    for d in schema.component_columns():
        # We only need one component per Boxes3D entity; centers is enough.
        if d.component != "Boxes3D:centers":
            continue
        ent = d.entity_path
        k = parse_instance_key_from_entity_path(ent)
        if k is None:
            unparsed.append(ent)
        else:
            # Unescape label/ins_id because rerun entity paths may contain backslash escapes.
            keys.add((rr_unescape_component(k[0]), rr_unescape_component(k[1])))
    return keys, unparsed


def _label_diag(label: str) -> Dict[str, Any]:
    ws = bool(re.search(r"\s", label))
    leading = label[:1].isspace()
    trailing = label[-1:].isspace()
    multi_space = "  " in label
    has_tab = "\t" in label
    has_nl = "\n" in label or "\r" in label
    has_slash = "/" in label
    has_backslash = "\\" in label
    return {
        "repr": repr(label),
        "len": len(label),
        "has_whitespace": ws,
        "leading_space": leading,
        "trailing_space": trailing,
        "multi_space": multi_space,
        "has_tab": has_tab,
        "has_newline": has_nl,
        "has_slash": has_slash,
        "has_backslash": has_backslash,
    }


def _norm_strip_collapse_ws(label: str) -> str:
    return " ".join(label.split())


def _norm_lower_strip_collapse_ws(label: str) -> str:
    return _norm_strip_collapse_ws(label).casefold()


@dataclass
class DiffReport:
    scene: str
    rrd_path: str
    json_path: str
    json_count: int
    rrd_count: int
    missing_count: int
    extra_count: int
    unparsed_rrd_entity_paths_count: int
    missing: List[Dict[str, Any]]
    extra: List[Dict[str, Any]]
    diagnostics: Dict[str, Any]


def diff_one(rrd_path: Path, json_path: Path) -> DiffReport:
    scene = rrd_path.stem
    json_keys = load_json_keys(json_path)
    rrd_keys, unparsed = load_rrd_box_keys(rrd_path)

    missing = sorted(json_keys - rrd_keys)
    extra = sorted(rrd_keys - json_keys)

    missing_rows: List[Dict[str, Any]] = []
    for label, ins_id in missing[:2000]:
        missing_rows.append({"label": label, "ins_id": ins_id, "label_diag": _label_diag(label)})
    extra_rows: List[Dict[str, Any]] = []
    for label, ins_id in extra[:2000]:
        extra_rows.append({"label": label, "ins_id": ins_id, "label_diag": _label_diag(label)})

    # Heuristics: do missing keys match if we normalize whitespace?
    json_norm = {( _norm_lower_strip_collapse_ws(l), i) for (l, i) in json_keys}
    rrd_norm = {( _norm_lower_strip_collapse_ws(l), i) for (l, i) in rrd_keys}
    missing_norm = sorted(json_norm - rrd_norm)

    # Heuristics: if we only unescape rerun-style backslashes, does it fix it?
    # (This should be 0 if the only difference was backslash-escaped spaces.)
    # Note: rrd_keys already unescaped in load_rrd_box_keys(), so this is just a placeholder
    # to show in the report that unescape was applied.

    # Per-ins_id label mismatches
    by_id_json: Dict[str, Set[str]] = defaultdict(set)
    by_id_rrd: Dict[str, Set[str]] = defaultdict(set)
    for l, i in json_keys:
        by_id_json[i].add(l)
    for l, i in rrd_keys:
        by_id_rrd[i].add(l)
    ids_all = set(by_id_json.keys()) | set(by_id_rrd.keys())
    id_label_conflicts = []
    for iid in sorted(ids_all):
        lj = by_id_json.get(iid, set())
        lr = by_id_rrd.get(iid, set())
        if lj and lr and lj != lr:
            id_label_conflicts.append(
                {"ins_id": iid, "json_labels": sorted(lj)[:20], "rrd_labels": sorted(lr)[:20]}
            )

    # Label diagnostics summary (only for missing+extra)
    diag_counter = Counter()
    for (l, _) in missing:
        d = _label_diag(l)
        if d["has_whitespace"]:
            diag_counter["missing_has_whitespace"] += 1
        if d["leading_space"]:
            diag_counter["missing_leading_space"] += 1
        if d["trailing_space"]:
            diag_counter["missing_trailing_space"] += 1
        if d["multi_space"]:
            diag_counter["missing_multi_space"] += 1
        if d["has_slash"]:
            diag_counter["missing_has_slash"] += 1
    for (l, _) in extra:
        d = _label_diag(l)
        if d["has_whitespace"]:
            diag_counter["extra_has_whitespace"] += 1
        if d["leading_space"]:
            diag_counter["extra_leading_space"] += 1
        if d["trailing_space"]:
            diag_counter["extra_trailing_space"] += 1
        if d["multi_space"]:
            diag_counter["extra_multi_space"] += 1
        if d["has_slash"]:
            diag_counter["extra_has_slash"] += 1

    diagnostics = {
        "missing_preview_limit": 2000,
        "extra_preview_limit": 2000,
        "unparsed_rrd_entity_paths_preview": unparsed[:200],
        "unparsed_rrd_entity_paths_note": "If this is non-empty, your rrd has Boxes3D entities not following /instances/{label}/{id}/bbox shape.",
        "whitespace_normalization": {
            "missing_count_after_norm(label=strip+collapse_ws+casefold, keep ins_id)": len(missing_norm),
            "missing_preview_after_norm": [{"label_norm": l, "ins_id": i} for (l, i) in missing_norm[:50]],
            "note": "If this number drops a lot vs missing_count, then whitespace differences are likely the culprit.",
        },
        "rerun_unescape": {
            "applied": True,
            "note": "rrd entity labels may contain backslash-escaped spaces. This tool unescapes them before diffing.",
        },
        "ins_id_label_conflicts_preview": id_label_conflicts[:50],
        "summary_counters": dict(diag_counter),
    }

    return DiffReport(
        scene=scene,
        rrd_path=str(rrd_path),
        json_path=str(json_path),
        json_count=len(json_keys),
        rrd_count=len(rrd_keys),
        missing_count=len(missing),
        extra_count=len(extra),
        unparsed_rrd_entity_paths_count=len(unparsed),
        missing=missing_rows,
        extra=extra_rows,
        diagnostics=diagnostics,
    )


def _write_report_json(report: DiffReport, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report.__dict__, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rrd", type=Path, default=None, help="Path to a single .rrd")
    ap.add_argument("--json", type=Path, default=None, help="Path to a single bbox .json")
    ap.add_argument("--scene", type=str, default=None, help="Scene id/stem (e.g. 0a5c013435)")
    ap.add_argument("--old_dir", type=Path, default=Path("output_scannetppv2_new"), help="Dir containing .rrd files")
    ap.add_argument("--new_json_dir", type=Path, default=Path("output_scannetppv2_new_aabb"), help="Dir containing updated .json files")
    ap.add_argument("--out_dir", type=Path, default=Path("rrd_json_diff_reports"), help="Where to write reports")
    ap.add_argument("--no_write", action="store_true", help="Do not write per-scene report json files")
    ap.add_argument("--limit", type=int, default=0, help="Batch mode: only process first N scenes (0=all)")
    args = ap.parse_args()

    # Single-scene mode
    if args.rrd or args.json or args.scene:
        if args.scene is not None:
            rrd_path = args.old_dir / f"{args.scene}.rrd"
            json_path = args.new_json_dir / f"{args.scene}.json"
        else:
            if args.rrd is None or args.json is None:
                raise SystemExit("Single mode requires --rrd and --json, or --scene with --old_dir/--new_json_dir")
            rrd_path = args.rrd
            json_path = args.json
        if not rrd_path.exists():
            raise SystemExit(f"rrd not found: {rrd_path}")
        if not json_path.exists():
            raise SystemExit(f"json not found: {json_path}")

        report = diff_one(rrd_path, json_path)
        print(
            f"[{report.scene}] json={report.json_count}, rrd={report.rrd_count}, "
            f"missing(json-rrd)={report.missing_count}, extra(rrd-json)={report.extra_count}, "
            f"unparsed_rrd_paths={report.unparsed_rrd_entity_paths_count}"
        )
        if report.missing_count:
            print("missing preview (first 30):")
            for row in report.missing[:30]:
                print(f"  - ins_id={row['ins_id']}, label={row['label_diag']['repr']}")
        if report.extra_count:
            print("extra preview (first 30):")
            for row in report.extra[:30]:
                print(f"  - ins_id={row['ins_id']}, label={row['label_diag']['repr']}")
        if not args.no_write:
            out_path = args.out_dir / f"{report.scene}_diff.json"
            _write_report_json(report, out_path)
            print(f"wrote report: {out_path}")
        return

    # Batch mode: diff all rrd in old_dir against new_json_dir
    rrd_files = sorted(args.old_dir.rglob("*.rrd"))
    if args.limit and args.limit > 0:
        rrd_files = rrd_files[: args.limit]
    if not rrd_files:
        raise SystemExit(f"No .rrd under: {args.old_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.out_dir / "summary.csv"
    rows: List[Dict[str, Any]] = []

    for rrd_path in rrd_files:
        scene = rrd_path.stem
        json_path = args.new_json_dir / f"{scene}.json"
        if not json_path.exists():
            rows.append(
                {
                    "scene": scene,
                    "rrd_path": str(rrd_path),
                    "json_path": str(json_path),
                    "status": "json_missing",
                    "json_count": "",
                    "rrd_count": "",
                    "missing_count": "",
                    "extra_count": "",
                    "unparsed_rrd_paths": "",
                }
            )
            continue
        rep = diff_one(rrd_path, json_path)
        if not args.no_write:
            _write_report_json(rep, args.out_dir / f"{scene}_diff.json")
        rows.append(
            {
                "scene": scene,
                "rrd_path": rep.rrd_path,
                "json_path": rep.json_path,
                "status": "ok",
                "json_count": rep.json_count,
                "rrd_count": rep.rrd_count,
                "missing_count": rep.missing_count,
                "extra_count": rep.extra_count,
                "unparsed_rrd_paths": rep.unparsed_rrd_entity_paths_count,
            }
        )

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "scene",
                "status",
                "json_count",
                "rrd_count",
                "missing_count",
                "extra_count",
                "unparsed_rrd_paths",
                "rrd_path",
                "json_path",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"wrote batch summary: {summary_csv}")


if __name__ == "__main__":
    main()

