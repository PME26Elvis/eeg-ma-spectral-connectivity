"""Reproducible resource benchmarks for handwritten RF and LightGBM.

No EEG participants or datasets are loaded. Fits run in independent subprocesses
to get process-level peak RSS instead of cumulative maxrss within one interpreter.
"""
from __future__ import annotations

import argparse
import cProfile
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import pstats
import resource
import statistics
import subprocess
import sys
import time

import numpy as np

SCHEMA = "eeg-handwritten-performance-1"
PRESETS = {
    "ci": [(300, 21, "dense"), (300, 126, "sparse")],
    "full": [
        (300, 21, "dense"), (300, 126, "sparse"),
        (300, 168, "correlated"), (300, 312, "constant"),
        (600, 21, "sparse"), (600, 126, "dense"),
        (600, 168, "correlated"), (600, 312, "sparse"),
        (3000, 21, "dense"), (3000, 126, "sparse"),
        (3000, 168, "correlated"), (3000, 312, "dense"),
        (6000, 21, "sparse"), (6000, 126, "dense"),
        (6000, 168, "correlated"), (6000, 312, "sparse"),
    ],
}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def machine():
    cpu = platform.processor()
    if sys.platform == "linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    return {
        "machine": platform.machine(), "system": platform.system(),
        "cpu_model": cpu or "unknown", "available_cpus": os.cpu_count(),
        "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "python": ".".join(str(x) for x in sys.version_info[:3]),
        "numpy": np.__version__,
        "blas_thread_env": {k: os.environ.get(k) for k in
                            ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
    }


def fixture(rows, columns, kind, seed):
    rng = np.random.default_rng(seed)
    if kind == "dense":
        X = rng.normal(size=(rows, columns))
    elif kind == "sparse":
        X = np.zeros((rows, columns), dtype=float)
        indices = np.arange(rows)
        a = rng.integers(0, max(1, columns // 2), rows)
        b = rng.integers(max(1, columns // 2), columns, rows)
        X[indices, a] = rng.uniform(-3, 3, rows)
        X[indices, b] = rng.uniform(.1, 3, rows)
    elif kind == "correlated":
        latent = rng.normal(size=(rows, 5))
        X = latent[:, np.arange(columns) % 5] + .06 * rng.normal(size=(rows, columns))
    elif kind == "constant":
        X = np.zeros((rows, columns), dtype=float)
        X[:, :min(columns, 3)] = rng.normal(size=(rows, min(columns, 3)))
    else:
        raise ValueError(kind)
    # Scores are engineering fixtures only. Never use scores as EEG results.
    score = X[:, 0] + .35 * X[:, min(1, columns-1)] - .2 * X[:, min(2, columns-1)]
    y = (score > np.median(score)).astype(np.int64)
    assert set(y.tolist()) == {0, 1} and np.isfinite(X).all()
    return X, y


def model_parameters(name, jobs):
    if name == "RandomForest":
        return {"n_estimators": 6, "max_depth": 5, "min_samples_leaf": 2,
                "max_features": "sqrt", "oob_score": True, "random_state": 2601,
                "n_jobs": jobs}
    return {"n_estimators": 6, "max_depth": 4, "num_leaves": 5,
            "max_bin": 31, "min_data_in_leaf": 3, "learning_rate": .06,
            "random_state": 2601, "enable_bundle": True, "n_jobs": jobs}


def workload_units(name, rows, features, params):
    if name == "RandomForest":
        sampled = max(1, int(math.sqrt(features)))
        return int(math.ceil(params["n_estimators"] * sampled * rows * math.log2(rows + 1)))
    return int(math.ceil(params["n_estimators"] * features * rows *
                         (1 + math.log2(rows + 1)) * (1 + math.log2(params["num_leaves"] + 1))))


def measured_case(args):
    import eeg_ma
    from eeg_ma.random_forest import HandcraftedRandomForestClassifier
    from eeg_ma.lightgbm import HandcraftedLightGBMClassifier
    from eeg_ma.provenance import _verify_forest_training_counts
    from eeg_ma.lightgbm_audit import _verify_training

    with np.load(args.input, allow_pickle=False) as data:
        X, y = data["X"], data["y"]
    if X.shape != (args.rows, args.features):
        raise ValueError("Worker input shape differs from requested benchmark")
    cls = HandcraftedRandomForestClassifier if args.model == "RandomForest" else HandcraftedLightGBMClassifier
    p = model_parameters(args.model, args.jobs)
    model = cls(**p)
    profiler = cProfile.Profile() if args.profile else None
    if profiler is not None:
        profiler.enable()
    cpu_start, tick = time.process_time(), time.perf_counter()
    model.fit(X, y)
    fit_wall, fit_cpu = time.perf_counter() - tick, time.process_time() - cpu_start
    if profiler is not None:
        profiler.disable()
    tick = time.perf_counter()
    proba = model.predict_proba(X[:min(80, len(X))])
    predict_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    serialized = model.to_dict()
    raw = canonical(serialized)
    restored = cls.from_dict(serialized)
    json_seconds = time.perf_counter() - tick
    if not np.array_equal(proba, restored.predict_proba(X[:len(proba)])):
        raise AssertionError("Serialization changed exact model predictions")
    tick = time.perf_counter()
    if args.model == "RandomForest":
        _verify_forest_training_counts(model, X, y)
        audit_scope = "source_bound_final_forest_counts"
    else:
        _verify_training(model, X, y)
        audit_scope = "source_bound_boosting_derivatives_and_routing"
    audit_seconds = time.perf_counter() - tick
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_mib = usage.ru_maxrss / 1024 if sys.platform == "linux" else usage.ru_maxrss / 2**20
    output = {
        "schema_version": SCHEMA, "model": args.model, "rows": len(X), "features": X.shape[1],
        "kind": args.kind, "jobs": args.jobs, "params": p,
        "source_module": str(Path(eeg_ma.__file__).resolve()),
        "source_module_sha256": digest(Path(sys.modules[cls.__module__].__file__)),
        "input_sha256": digest(args.input), "host": machine(),
        "fit_wall_s": fit_wall, "fit_cpu_s": fit_cpu, "predict_wall_s": predict_seconds,
        "serialize_roundtrip_s": json_seconds, "audit_wall_s": audit_seconds,
        "audit_scope": audit_scope, "peak_rss_mib": peak_mib,
        "model_json_bytes": len(raw.encode("utf-8")),
        "model_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "prediction_sha256": hashlib.sha256(proba.tobytes()).hexdigest(),
        "units": workload_units(args.model, len(X), X.shape[1], p),
    }
    if profiler is not None:
        buffer = io.StringIO()
        stats = pstats.Stats(profiler, stream=buffer).sort_stats("cumulative")
        stats.print_stats(30)
        Path(args.profile).write_text(buffer.getvalue(), encoding="utf-8")
    Path(args.output).write_text(json.dumps(output, sort_keys=True, indent=2,
                                           allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "measured", "fit_wall_s": fit_wall,
                      "peak_rss_mib": peak_mib}, sort_keys=True))


def percentile(values, pct):
    xs = sorted(values)
    if not xs:
        raise ValueError("empty measurements")
    return float(np.quantile(xs, pct))


def suite(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    inputs = Path(args.inputs_dir) if args.inputs_dir else out / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    observed = []
    for rows, features, kind in PRESETS[args.preset]:
        if args.rows_only is not None and rows != args.rows_only:
            continue
        if args.case_only and f"r{rows}-f{features}-{kind}" not in args.case_only:
            continue
        name = f"r{rows}-f{features}-{kind}"
        source = inputs / f"{name}.npz"
        if not source.is_file():
            X, y = fixture(rows, features, kind, args.seed)
            np.savez(source, X=X, y=y)
        elif args.inputs_dir:
            with np.load(source, allow_pickle=False) as archive:
                want_X, want_y = fixture(rows, features, kind, args.seed)
                if not np.array_equal(archive["X"], want_X) or not np.array_equal(archive["y"], want_y):
                    raise AssertionError("Shared input bytes do not represent declared deterministic fixture")
        for model in ("RandomForest", "LightGBM"):
            for jobs in args.jobs:
                group = f"{name}-{model}-j{jobs}"
                for rep in range(-args.warmups, args.repeats):
                    output = out / f"{group}-rep{rep}.json"
                    cmd = [sys.executable, __file__, "case", "--input", str(source),
                           "--rows", str(rows), "--features", str(features),
                           "--kind", kind, "--model", model, "--jobs", str(jobs),
                           "--output", str(output)]
                    if args.profile and rep == 0:
                        cmd += ["--profile", str(out / f"{group}.profile.txt")]
                    try:
                        subprocess.run(cmd, check=True, timeout=args.timeout,
                                       capture_output=True, text=True)
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                        error = {"case": group, "rep": rep, "error": str(exc),
                                 "stdout": getattr(exc, "stdout", ""),
                                 "stderr": getattr(exc, "stderr", "")}
                        (out / "failed_case.json").write_text(json.dumps(error, default=str) + "\n")
                        raise
                    if rep >= 0:
                        result = json.loads(output.read_text())
                        result["repeat"] = rep
                        observed.append(result)
                    else:
                        output.unlink()
    if not observed:
        raise ValueError("Empty benchmark matrix after workload filters")
    grouped = {}
    for row in observed:
        key = f"{row['rows']}x{row['features']}/{row['kind']}/{row['model']}/j{row['jobs']}"
        grouped.setdefault(key, []).append(row)
    summaries = []
    for key, rows in grouped.items():
        if len(rows) < 3:
            raise AssertionError("At least three non-warmup repeats required")
        values = [x["fit_wall_s"] for x in rows]
        first = rows[0]
        if len({(x["input_sha256"], x["model_sha256"], x["prediction_sha256"]) for x in rows}) != 1:
            raise AssertionError("Identical repeated input/models produced different fitted bytes")
        summaries.append({
            "key": key, "model": first["model"], "rows": first["rows"],
            "features": first["features"], "kind": first["kind"],
            "jobs": first["jobs"], "n_repeats": len(rows),
            "fit_wall_median_s": statistics.median(values),
            "fit_wall_p95_s": percentile(values, .95),
            "fit_wall_min_s": min(values), "fit_wall_max_s": max(values),
            "fit_cpu_median_s": statistics.median(x["fit_cpu_s"] for x in rows),
            "peak_rss_max_mib": max(x["peak_rss_mib"] for x in rows),
            "predict_median_s": statistics.median(x["predict_wall_s"] for x in rows),
            "audit_median_s": statistics.median(x["audit_wall_s"] for x in rows),
            "model_bytes": first["model_json_bytes"], "input_sha256": first["input_sha256"],
            "model_sha256": first["model_sha256"],
            "prediction_sha256": first["prediction_sha256"],
            "units": first["units"],
        })
    max_ratio = max(x["fit_wall_s"] / x["units"] for x in observed)
    calibration = {
        "schema_version": "eeg-benchmark-calibration-1",
        "host": machine(), "seconds_per_unit": max_ratio,
        "measurements": len(observed), "safety_factor": 2.0,
        "scope": "same_host_handwritten_tree_units_only_no_SFS_or_non_tree_runtime",
    }
    report = {
        "schema_version": SCHEMA, "preset": args.preset, "status": "complete",
        "host": machine(), "seed": args.seed,
        "n_measurements": len(observed), "warmups": args.warmups,
        "repeats": args.repeats, "cases": summaries,
        "scope": "SYNTHETIC_ENGINEERING_ONLY_NOT_EEG_MODEL_ACCURACY",
        "warnings": ["Wall time is host- and workload-specific; do not infer training speed from a different runner.",
                     "Peak RSS is process-level and includes imports, not net allocation from fit.",
                     "P95 with three repeats is descriptive, not a statistically precise tail percentile.",
                     "Tree-work calibration excludes LDA SFS and SVM/KFDA compute."],
    }
    (out / "summary.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(12, max(4, 0.30 * len(summaries) + 2)))
    ordering = sorted(summaries, key=lambda x: x["key"])
    ax.barh(range(len(ordering)), [row["fit_wall_median_s"] for row in ordering])
    ax.set_yticks(range(len(ordering)), [row["key"] for row in ordering])
    ax.invert_yaxis()
    ax.set_xlabel("Median fit wall seconds (3 independent measured repeats)")
    ax.set_title("Synthetic handwritten-tree fit benchmarks")
    fig.tight_layout()
    fig.savefig(out / "fit_medians.png", dpi=125)
    plt.close(fig)
    with (out / "measurements.jsonl").open("w", encoding="utf-8") as stream:
        for row in observed:
            stream.write(canonical(row) + "\n")
    # Validate calibration provenance against the actual measured observations.
    observations = out / "measurements.jsonl"
    calibration["measurements_sha256"] = digest(observations)
    calibration["learner_sha256"] = {
        learner: next(row["source_module_sha256"] for row in observed if row["model"] == learner)
        for learner in ("RandomForest", "LightGBM")
    }
    for learner, source_sha in calibration["learner_sha256"].items():
        if any(row["source_module_sha256"] != source_sha for row in observed if row["model"] == learner):
            raise AssertionError("Learner source changed within one benchmark suite")
    (out / "calibration.json").write_text(json.dumps(
        calibration, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "out_dir": str(out),
                      "measurements": len(observed), "cases": len(summaries)}))


def compare(args):
    base = json.loads((Path(args.baseline) / "summary.json").read_text())
    target = json.loads((Path(args.candidate) / "summary.json").read_text())
    if base["host"] != target["host"] or base["preset"] != target["preset"]:
        raise ValueError("Comparisons must use the same host, environment, preset and input definitions")
    a, b = ({row["key"]: row for row in doc["cases"]} for doc in (base, target))
    if set(a) != set(b):
        raise ValueError("Benchmark matrix differs across baseline and candidate")
    rows = []
    for key in sorted(a):
        old, new = a[key], b[key]
        if any(old[f] != new[f] for f in
               ("input_sha256", "model_sha256", "prediction_sha256", "units")):
            raise AssertionError(f"Numerical/source identity drift: {key}")
        speed = old["fit_wall_median_s"] / new["fit_wall_median_s"]
        rows.append({"key": key, "baseline_median_s": old["fit_wall_median_s"],
                     "candidate_median_s": new["fit_wall_median_s"],
                     "speedup": speed, "median_improvement_pct": 100 * (1 - 1 / speed),
                     "baseline_peak_rss_mib": old["peak_rss_max_mib"],
                     "candidate_peak_rss_mib": new["peak_rss_max_mib"],
                     "candidate_memory_nonregression":
                         new["peak_rss_max_mib"] <= old["peak_rss_max_mib"] * 1.05})
    result = {"schema_version": SCHEMA, "status": "verified_numerically",
              "same_runner": True, "cases": rows,
              "performance_threshold_is_advisory_not_a_flaky_ci_gate": True}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(12, max(4, .36 * len(rows) + 2)))
    ax.barh(range(len(rows)), [row["median_improvement_pct"] for row in rows])
    ax.axvline(0, color="gray", linewidth=.8)
    ax.set_yticks(range(len(rows)), [row["key"] for row in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Same-runner median fit time change (%) vs baseline")
    ax.set_title("Synthetic workloads; no EEG inference")
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=125)
    plt.close(fig)

    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    case = sub.add_parser("case")
    for name, typ in (("input", str), ("model", str), ("kind", str), ("output", str)):
        case.add_argument(f"--{name}", required=True, type=typ)
    for name in ("rows", "features", "jobs"):
        case.add_argument(f"--{name}", required=True, type=int)
    case.add_argument("--profile", default=None)
    run = sub.add_parser("suite")
    run.add_argument("--out-dir", required=True)
    run.add_argument("--preset", choices=PRESETS, default="ci")
    run.add_argument("--seed", type=int, default=260126)
    run.add_argument("--inputs-dir")
    run.add_argument("--jobs", nargs="+", type=int, default=[1, 2, 4])
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--warmups", type=int, default=1)
    run.add_argument("--timeout", type=float, default=180)
    run.add_argument("--rows-only", type=int, choices=[300, 600, 3000, 6000])
    run.add_argument("--case-only", nargs="+", help="Restrict to listed prespecified workload IDs")
    run.add_argument("--profile", action="store_true")
    comp = sub.add_parser("compare")
    comp.add_argument("--baseline", required=True)
    comp.add_argument("--candidate", required=True)
    comp.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "case":
        measured_case(args)
    elif args.command == "suite":
        if (args.repeats < 3 or args.warmups < 0 or
                not args.jobs or any(j not in (1, 2, 4) for j in args.jobs)
                or args.timeout <= 0):
            raise ValueError("Bounded benchmark requires >=3 repeats, jobs 1/2/4 and positive timeout")
        suite(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
