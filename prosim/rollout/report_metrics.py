import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

BKT_METRICS = {
  "Kinematic": [
    "linear_speed_likelihood",
    "linear_acceleration_likelihood",
    "angular_speed_likelihood",
    "angular_acceleration_likelihood",
  ],
  "Interactive": [
    "distance_to_nearest_object_likelihood",
    "collision_indication_likelihood",
    "time_to_collision_likelihood",
  ],
  "Map": [
    "distance_to_road_edge_likelihood",
    "offroad_indication_likelihood",
  ],
}


def maybe_float(value):
  try:
    numeric_value = float(str(value).strip().strip('"'))
    if not math.isfinite(numeric_value):
      return None
    return numeric_value
  except Exception:
    return None


def summarize_metrics(root):
  root = Path(root)
  metrics_dir = root / "metrics"
  metric_files = sorted(glob.glob(str(metrics_dir / "scene_*.json")))

  if len(metric_files) == 0:
    raise FileNotFoundError(f"No metric files found in {metrics_dir}")

  values = defaultdict(list)
  scenario_ids = []

  for metric_file in metric_files:
    with open(metric_file, "r") as f:
      data = json.load(f)

    if "scenario_id" in data:
      scenario_ids.append(str(data["scenario_id"]).strip().strip('"'))

    for key, value in data.items():
      numeric_value = maybe_float(value)
      if numeric_value is not None:
        values[key].append(numeric_value)

  report = {
    "num_scenes": len(metric_files),
    "num_unique_scenarios": len(set(scenario_ids)),
    "aggregation": "simple_mean_over_scenes",
    "metrics": {},
    "bucket_metrics": {},
  }

  for key in sorted(values.keys()):
    report["metrics"][key] = {
      "mean": sum(values[key]) / len(values[key]),
      "count": len(values[key]),
    }

  for bucket_name, metric_names in BKT_METRICS.items():
    bucket_values = [report["metrics"][metric_name]["mean"] for metric_name in metric_names if metric_name in report["metrics"]]
    if len(bucket_values) == 0:
      continue

    report["bucket_metrics"][bucket_name] = {
      "mean": sum(bucket_values) / len(bucket_values),
      "metrics": metric_names,
    }

  report_path = root / "metrics_report.json"
  with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

  summary_lines = [
    f"root: {root}",
    f"num_scenes: {report['num_scenes']}",
    f"num_unique_scenarios: {report['num_unique_scenarios']}",
    "aggregation: simple mean over per-scene metric JSON files",
    "note: metametric is the official Waymo per-scenario metametric; this report averages those per-scene values without extra weighting",
    "",
  ]

  if len(report["bucket_metrics"]) > 0:
    summary_lines.append("bucket_metrics:")
    for bucket_name in ["Kinematic", "Interactive", "Map"]:
      if bucket_name not in report["bucket_metrics"]:
        continue
      metric_list = ", ".join(report["bucket_metrics"][bucket_name]["metrics"])
      summary_lines.append(f"{bucket_name}: {report['bucket_metrics'][bucket_name]['mean']:.6f} [{metric_list}]")
    summary_lines.append("")

  for key in sorted(report["metrics"].keys()):
    summary_lines.append(
      f"{key}: {report['metrics'][key]['mean']:.6f} (n={report['metrics'][key]['count']})"
    )

  summary_path = root / "metrics_summary.txt"
  with open(summary_path, "w") as f:
    f.write("\n".join(summary_lines) + "\n")

  print(f"Saved report to {report_path}")
  print(f"Saved summary to {summary_path}")
  print(f"num_scenes: {report['num_scenes']}")
  print(f"num_unique_scenarios: {report['num_unique_scenarios']}")
  for bucket_name in ["Kinematic", "Interactive", "Map"]:
    if bucket_name in report["bucket_metrics"]:
      print(f"{bucket_name}: {report['bucket_metrics'][bucket_name]['mean']:.6f}")
  for key in sorted(report["metrics"].keys()):
    print(f"{key}: {report['metrics'][key]['mean']:.6f} (n={report['metrics'][key]['count']})")

  return report_path, summary_path, report


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=str, required=True, help="Rollout root containing metrics/")
  args = parser.parse_args()
  summarize_metrics(args.root)


if __name__ == "__main__":
  main()
