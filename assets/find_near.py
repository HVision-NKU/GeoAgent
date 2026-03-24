#!/usr/bin/env python3
import argparse
import bisect
import csv
import json
import math
import os
import re
from typing import Dict, List, Sequence, Tuple

EARTH_RADIUS_KM = 6371.0088
Coord = Tuple[float, float]

LAT_CANDIDATES = [
    "ground_truth_lat",
    "lat",
    "latitude",
    "LAT",
    "predict_lat",
]
LON_CANDIDATES = [
    "ground_truth_lon",
    "lon",
    "lng",
    "longitude",
    "LON",
    "predict_lon",
]


def normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def detect_coord_keys(keys: Sequence[str]) -> Tuple[str, str]:
    norm_to_raw = {normalize_key(k): k for k in keys}

    lat_key = None
    lon_key = None

    for cand in LAT_CANDIDATES:
        c = normalize_key(cand)
        if c in norm_to_raw:
            lat_key = norm_to_raw[c]
            break
    for cand in LON_CANDIDATES:
        c = normalize_key(cand)
        if c in norm_to_raw:
            lon_key = norm_to_raw[c]
            break

    if lat_key is None or lon_key is None:
        raise ValueError(
            "  None " + ", ".join(keys[:50])
        )
    return lat_key, lon_key


def load_records(path: str) -> Tuple[List[Dict[str, object]], List[str]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            records = list(reader)
            keys = reader.fieldnames or []
        return records, list(keys)

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{path}  None ")
        records = [x for x in data if isinstance(x, dict)]
        keys = list(records[0].keys()) if records else []
        return records, keys

    raise ValueError(f" None {path}")


def load_coords(path: str, max_rows: int = 0) -> Tuple[List[Coord], str, str, int]:
    records, keys = load_records(path)
    lat_key, lon_key = detect_coord_keys(keys)

    coords: List[Coord] = []
    total_rows = 0
    for row in records:
        if max_rows > 0 and total_rows >= max_rows:
            break
        total_rows += 1
        lat = row.get(lat_key)
        lon = row.get(lon_key)
        if lat is None or lon is None:
            continue
        try:
            lat_f = float(lat)
            lon_f = float(lon)
        except (TypeError, ValueError):
            continue
        if -90.0 <= lat_f <= 90.0 and -180.0 <= lon_f <= 180.0:
            coords.append((lat_f, lon_f))

    return coords, lat_key, lon_key, total_rows


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.asin(min(1.0, math.sqrt(a)))
    return EARTH_RADIUS_KM * c


def compute_min_distances_with_cap(
    source_points: Sequence[Coord], target_points: Sequence[Coord], max_threshold_km: float
) -> List[float]:
    if max_threshold_km <= 0:
        raise ValueError("max_threshold_km must > 0")

    if not source_points:
        return []
    if not target_points:
        return [math.inf for _ in source_points]

    target_sorted = sorted((lat, lon) for lat, lon in target_points)
    target_lats = [x[0] for x in target_sorted]
    delta_lat_deg = math.degrees(max_threshold_km / EARTH_RADIUS_KM)

    min_distances: List[float] = []
    for s_lat, s_lon in source_points:
        low = s_lat - delta_lat_deg
        high = s_lat + delta_lat_deg
        left = bisect.bisect_left(target_lats, low)
        right = bisect.bisect_right(target_lats, high)

        best = math.inf
        for i in range(left, right):
            t_lat, t_lon = target_sorted[i]
            d = haversine_km(s_lat, s_lon, t_lat, t_lon)
            if d < best:
                best = d
        min_distances.append(best)
    return min_distances


def parse_thresholds_km(text: str) -> List[float]:
    vals = []
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        v = float(p)
        if v <= 0:
            raise ValueError(f"threshold must > 0, received {v}")
        vals.append(v)
    if not vals:
        raise ValueError("at least one threshold must be provided")
    return sorted(set(vals))


def summarize_for_pair(distances: Sequence[float], thresholds_km: Sequence[float], source_total: int):
    rows = []
    for th in thresholds_km:
        c = sum(1 for d in distances if d <= th)
        rows.append(
            {
                "threshold_km": th,
                "near_count": c,
                "near_ratio": c / source_total if source_total else 0.0,
            }
        )
    return rows


def sample_points(points: Sequence[Coord], max_points: int) -> List[Coord]:
    if max_points <= 0 or len(points) <= max_points:
        return list(points)
    step = max(1, len(points) // max_points)
    sampled = [points[i] for i in range(0, len(points), step)]
    return sampled[:max_points]


def plot_spatial_single(dataset: dict, ref_dataset: dict, output_path: str, max_plot_points: int) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6), dpi=150)

    ref_pts = sample_points(ref_dataset["coords"], max_plot_points)
    if ref_pts:
        ref_lats, ref_lons = zip(*ref_pts)
        plt.scatter(
            ref_lons,
            ref_lats,
            s=6,
            alpha=0.25,
            c="#ff7f0e",
            label=f"{ref_dataset['name']} (n={len(ref_dataset['coords'])}, plot={len(ref_pts)})",
        )

    pts = sample_points(dataset["coords"], max_plot_points)
    if pts:
        lats, lons = zip(*pts)
        plt.scatter(
            lons,
            lats,
            s=6,
            alpha=0.35,
            c="#1f77b4",
            label=f"{dataset['name']} (n={len(dataset['coords'])}, plot={len(pts)})",
        )

    plt.title(f"Spatial Distribution: {dataset['name']} vs {ref_dataset['name']}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.grid(alpha=0.2)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_trend(pair_rows: Dict[str, List[dict]], output_path: str) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6), dpi=150)
    for pair_name, rows in pair_rows.items():
        x = [r["threshold_km"] for r in rows]
        y = [r["near_count"] for r in rows]
        plt.plot(x, y, marker="o", linewidth=1.8, label=pair_name)

    plt.xscale("log")
    plt.xlabel("Distance threshold (km)")
    plt.ylabel("Near count")
    plt.title("Near-count Trend vs im2gps (Haversine)")
    plt.grid(alpha=0.25)
    plt.legend(loc="best", ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_ratio_trend(pair_rows: Dict[str, List[dict]], output_path: str) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6), dpi=150)
    for pair_name, rows in pair_rows.items():
        x = [r["threshold_km"] for r in rows]
        y = [r["near_ratio"] for r in rows]
        plt.plot(x, y, marker="o", linewidth=1.8, label=pair_name)

    plt.xscale("log")
    plt.xlabel("Distance threshold (km)")
    plt.ylabel("Near ratio")
    plt.title("Near-ratio Trend vs im2gps (Haversine)")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Support JSON/CSV three-dataset spherical distance statistics and trend visualization."
    )
    parser.add_argument(
        "--dataset-a",
        default="/nfs/sunboyuan/Geobench/dataset/test_loc_img/GeoSeek_Loc.json",
    )
    parser.add_argument(
        "--dataset-b",
        default="/nfs/sunboyuan/Geobench/dataset/test_loc_img/gre_partaa.csv",
    )
    parser.add_argument(
        "--dataset-c",
        default="/nfs/sunboyuan/Geobench/dataset/test_loc_img/mp16-reason-train.csv",
    )
    parser.add_argument(
        "--dataset-ref",
        default="/nfs/sunboyuan/Geobench/dataset/test_loc_img/im2gps3k_test_data_resize.json",
    )
    parser.add_argument("--name-a", default="GeoSeek")
    parser.add_argument("--name-b", default="GRE")
    parser.add_argument("--name-c", default="MP16")
    parser.add_argument("--name-ref", default="im2gps")
    parser.add_argument("--thresholds-km", default="1,2,5,10,20,50,100")
    parser.add_argument(
        "--summary-path",
        default="/nfs/sunboyuan/Geobench/dataset/test_loc_img/near_summary.json",
    )
    parser.add_argument(
        "--spatial-plot-path",
        default="/nfs/sunboyuan/Geobench/dataset/test_loc_img/spatial_distribution_3sets.png",
    )
    parser.add_argument(
        "--trend-plot-path",
        default="/nfs/sunboyuan/Geobench/dataset/test_loc_img/near_trend_3sets.png",
    )
    parser.add_argument(
        "--ratio-trend-plot-path",
        default="/nfs/sunboyuan/Geobench/dataset/test_loc_img/near_ratio_trend_3sets.png",
    )
    parser.add_argument(
        "--max-rows-per-dataset",
        type=int,
        default=0,
        help="Maximum number of rows to read per dataset, 0 means all",
    )
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=30000,
        help="Maximum number of points to plot per dataset",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Only output statistics, no plotting",
    )
    args = parser.parse_args()

    thresholds_km = parse_thresholds_km(args.thresholds_km)
    max_threshold_km = max(thresholds_km)

    datasets = []
    for path, name in [
        (args.dataset_a, args.name_a),
        (args.dataset_b, args.name_b),
        (args.dataset_c, args.name_c),
    ]:
        coords, lat_key, lon_key, rows_read = load_coords(path, max_rows=args.max_rows_per_dataset)
        datasets.append(
            {
                "path": path,
                "name": name,
                "coords": coords,
                "lat_key": lat_key,
                "lon_key": lon_key,
                "rows_read": rows_read,
            }
        )

    ref_coords, ref_lat_key, ref_lon_key, ref_rows_read = load_coords(
        args.dataset_ref, max_rows=args.max_rows_per_dataset
    )
    ref_dataset = {
        "path": args.dataset_ref,
        "name": args.name_ref,
        "coords": ref_coords,
        "lat_key": ref_lat_key,
        "lon_key": ref_lon_key,
        "rows_read": ref_rows_read,
    }

    pair_summaries: Dict[str, List[dict]] = {}
    for src in datasets:
        pair_name = f"{src['name']} -> {ref_dataset['name']}"
        dists = compute_min_distances_with_cap(src["coords"], ref_dataset["coords"], max_threshold_km)
        pair_summaries[pair_name] = summarize_for_pair(
            distances=dists,
            thresholds_km=thresholds_km,
            source_total=len(src["coords"]),
        )

    os.makedirs(os.path.dirname(args.summary_path), exist_ok=True)
    summary = {
        "metric": "haversine_km",
        "thresholds_km": thresholds_km,
        "datasets": [
            {
                "name": ds["name"],
                "path": ds["path"],
                "rows_read": ds["rows_read"],
                "valid_coord_count": len(ds["coords"]),
                "lat_key": ds["lat_key"],
                "lon_key": ds["lon_key"],
            }
            for ds in datasets
        ],
        "reference_dataset": {
            "name": ref_dataset["name"],
            "path": ref_dataset["path"],
            "rows_read": ref_dataset["rows_read"],
            "valid_coord_count": len(ref_dataset["coords"]),
            "lat_key": ref_dataset["lat_key"],
            "lon_key": ref_dataset["lon_key"],
        },
        "pairwise_summary": pair_summaries,
        "spatial_plot_paths": None if args.no_plot else {},
        "near_count_trend_plot_path": None if args.no_plot else args.trend_plot_path,
        "near_ratio_trend_plot_path": None if args.no_plot else args.ratio_trend_plot_path,
    }

    if not args.no_plot:
        os.makedirs(os.path.dirname(args.spatial_plot_path), exist_ok=True)
        os.makedirs(os.path.dirname(args.trend_plot_path), exist_ok=True)
        os.makedirs(os.path.dirname(args.ratio_trend_plot_path), exist_ok=True)

        spatial_base, spatial_ext = os.path.splitext(args.spatial_plot_path)
        for ds in datasets:
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", ds["name"])
            ds_path = f"{spatial_base}_{safe_name}{spatial_ext or '.png'}"
            plot_spatial_single(
                dataset=ds,
                ref_dataset=ref_dataset,
                output_path=ds_path,
                max_plot_points=args.max_plot_points,
            )
            summary["spatial_plot_paths"][ds["name"]] = ds_path

        plot_trend(pair_rows=pair_summaries, output_path=args.trend_plot_path)
        plot_ratio_trend(pair_rows=pair_summaries, output_path=args.ratio_trend_plot_path)

    with open(args.summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
