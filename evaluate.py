from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Iterable, Iterator, Protocol, runtime_checkable

# Local imports (this file lives at repo root, eval/ is a sibling package).
from eval.schema import (
    ALLOWED_DATASETS,
    ALLOWED_LANGUAGES,
    ALLOWED_TASKS,
    PARSE_ERROR_TOKEN,
    validate_predictions_file,
)
from eval.score import score_predictions, write_metrics

PARSE_ERROR = PARSE_ERROR_TOKEN

DATASET_SLUGS = {
    "SemEval-2014-Restaurant": "semeval14_rest",
    "SemEval-2014-Laptop": "semeval14_lap",
    "UIT-VSFC": "vsfc",
}


@runtime_checkable
class Predictor(Protocol):
    """Protocol every teammate's model wrapper must implement.

    Required class-level attributes
    -------------------------------
    - ``method``    (str): canonical method name as it should appear in the
      final tables, e.g. ``"LCF-BERT"`` or ``"InstructABSA"``.
    - ``paradigm``  (str): one of ``"Discriminative"`` /
      ``"Instruction-Tuning"`` / ``"Graph"`` / ``"Generative-Seq2Seq"`` /
      ``"LLM-Reasoning"``.
    - ``backbone``  (str): the actual HF / model identifier used.

    Required method
    ---------------
    - ``predict(text, aspect=None) -> (pred_aspect, pred_sentiment, raw_output)``

      * As of SPEC v1.2 the model must predict **both** the aspect and
        the sentiment for both datasets. The runner does not provide
        the gold aspect — ``aspect`` is currently always ``None``.
        - For SemEval (ATSC): predict an aspect-term string (open
          vocabulary; ideally a substring of ``text``). It is compared
          to ``gold.aspect`` by exact string match.
        - For UIT-VSFC (ACSA): predict one of the 4 closed topic
          categories: ``lecturer`` / ``training_program`` /
          ``facility`` / ``others``.
      * If the model output cannot be parsed into a valid
        ``(aspect, sentiment)`` pair, return
        ``("__PARSE_ERROR__", "__PARSE_ERROR__", raw_output)`` — the
        runner will count this as a wrong prediction (locked decision
        #2).

    Optional methods
    ----------------
    - ``warmup(text)``: called ``--warmup`` times before timing starts so JIT
      compilation / first-token latency does not skew the numbers.
    """

    method: str
    paradigm: str
    backbone: str

    def predict(
        self, text: str, aspect: str | None = None
    ) -> tuple[str, str, str]:  # pragma: no cover - protocol stub
        ...


# --- Dataset loader --------------------------------------------------------


@dataclass
class TestSample:
    id: str
    dataset: str
    language: str
    task: str
    text: str
    gold_aspect: str
    gold_sentiment: str


def load_test_set(path: str | Path) -> list[TestSample]:
    """Load a unified test JSONL file. See SPEC §4 for the input schema."""
    samples: list[TestSample] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            try:
                samples.append(
                    TestSample(
                        id=rec["id"],
                        dataset=rec["dataset"],
                        language=rec["language"],
                        task=rec["task"],
                        text=rec["text"],
                        gold_aspect=rec["gold"]["aspect"],
                        gold_sentiment=rec["gold"]["sentiment"],
                    )
                )
            except KeyError as exc:
                raise ValueError(
                    f"{path}:{lineno}: missing required field {exc!s}"
                ) from exc

    if not samples:
        raise ValueError(f"empty test set: {path}")

    # Sanity check: a test file must be a single (dataset, task) pair.
    datasets = {s.dataset for s in samples}
    tasks = {s.task for s in samples}
    if len(datasets) != 1:
        raise ValueError(f"mixed datasets in {path}: {sorted(datasets)}")
    if len(tasks) != 1:
        raise ValueError(f"mixed tasks in {path}: {sorted(tasks)}")
    if next(iter(datasets)) not in ALLOWED_DATASETS:
        raise ValueError(
            f"unknown dataset {next(iter(datasets))!r} (see SPEC §4 for the allowed set)"
        )
    if next(iter(tasks)) not in ALLOWED_TASKS:
        raise ValueError(f"unknown task {next(iter(tasks))!r}")

    return samples


# --- Timing utilities ------------------------------------------------------


def _timed_predict(
    predictor: Predictor, text: str, aspect: str | None
) -> tuple[str, str, str, float]:
    """Run ``predictor.predict`` and return (aspect, sentiment, raw, latency_ms)."""
    t0 = time.perf_counter_ns()
    out = predictor.predict(text, aspect=aspect)
    elapsed_ms = (time.perf_counter_ns() - t0) / 1e6

    if not (isinstance(out, tuple) and len(out) == 3):
        raise TypeError(
            f"{predictor.__class__.__name__}.predict must return a 3-tuple "
            "(aspect, sentiment, raw_output); got " + repr(out)
        )
    pred_aspect, pred_sentiment, raw = out
    return str(pred_aspect), str(pred_sentiment), str(raw), elapsed_ms


def _gpu_mem_peak_gb() -> float | None:
    """Return the peak GPU memory in GB since the last reset, or None if no CUDA."""
    try:
        import torch  # type: ignore
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024**3)


def _reset_gpu_mem() -> None:
    try:
        import torch  # type: ignore
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# --- Main runner -----------------------------------------------------------


@dataclass
class RunConfig:
    test_set: Path
    output_dir: Path
    warmup: int = 5
    hardware: str = "1x T4 16GB"
    precision: str = "fp16"
    batch_size_inference: int = 1
    params_million: float | None = None
    training_hours: float | None = None
    extra_efficiency: dict[str, Any] = field(default_factory=dict)


def _slugify_method(method: str) -> str:
    return (
        method.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def _make_record(
    sample: TestSample,
    predictor: Predictor,
    pred_aspect: str,
    pred_sentiment: str,
    raw: str,
    parse_ok: bool,
    latency_ms: float,
) -> dict[str, Any]:
    return {
        "id": sample.id,
        "dataset": sample.dataset,
        "language": sample.language,
        "task": sample.task,
        "text": sample.text,
        "gold": {"aspect": sample.gold_aspect, "sentiment": sample.gold_sentiment},
        "pred": {"aspect": pred_aspect, "sentiment": pred_sentiment},
        "raw_output": raw,
        "parse_ok": parse_ok,
        "method": predictor.method,
        "paradigm": predictor.paradigm,
        "backbone": predictor.backbone,
        "latency_ms": round(latency_ms, 3),
    }


def run_evaluation(
    predictor: Predictor,
    config: RunConfig,
    progress_every: int = 100,
) -> dict[str, Any]:
    """Run ``predictor`` over a test set, writing predictions + metrics.

    Returns the metrics dict (also written to disk).
    """
    if not isinstance(predictor, Predictor):
        raise TypeError(
            "predictor does not implement the Predictor protocol "
            "(missing method / paradigm / backbone / predict?)"
        )

    samples = load_test_set(config.test_set)
    dataset = samples[0].dataset
    task = samples[0].task
    if samples[0].language not in ALLOWED_LANGUAGES:
        raise ValueError(f"unknown language {samples[0].language!r}")

    method_slug = _slugify_method(predictor.method)
    dataset_slug = DATASET_SLUGS.get(dataset, dataset.lower())

    pred_path = config.output_dir / "predictions" / f"{method_slug}_{dataset_slug}.jsonl"
    metrics_path = config.output_dir / "metrics" / f"{method_slug}_{dataset_slug}.json"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    # Warmup: run a few predictions but DO NOT time them.
    warmup_n = min(config.warmup, len(samples))
    if warmup_n > 0:
        _warmup_fn = getattr(predictor, "warmup", None)
        for sample in samples[:warmup_n]:
            # As of SPEC v1.2 the model predicts both aspect and sentiment
            # for both datasets, so the runner does not provide the gold
            # aspect.
            if callable(_warmup_fn):
                _warmup_fn(sample.text)
            else:
                predictor.predict(sample.text, aspect=None)

    # Reset GPU mem stats AFTER warmup so warmup allocations don't dominate.
    _reset_gpu_mem()
    overall_t0 = time.perf_counter_ns()
    latencies_ms: list[float] = []

    with pred_path.open("w", encoding="utf-8") as fh:
        for i, sample in enumerate(samples, start=1):
            # SPEC v1.2: the model predicts both aspect and sentiment.
            try:
                pa, ps, raw, latency_ms = _timed_predict(
                    predictor, sample.text, None
                )
            except Exception as exc:  # noqa: BLE001 -- fail-soft per-sample
                # Per SPEC §2 decision #2: a failure is a wrong prediction,
                # not a skip. Latency is undefined → record as 0.
                pa, ps, raw, latency_ms = (
                    PARSE_ERROR,
                    PARSE_ERROR,
                    f"<<predict raised {type(exc).__name__}: {exc}>>",
                    0.0,
                )
            parse_ok = pa != PARSE_ERROR and ps != PARSE_ERROR
            # If a predictor returns a partial parse error (one field is
            # PARSE_ERROR, the other isn't), normalise both to PARSE_ERROR so
            # the resulting record passes schema validation. The schema
            # requires both pred.aspect and pred.sentiment to equal the
            # token when parse_ok is false.
            if not parse_ok:
                pa, ps = PARSE_ERROR, PARSE_ERROR
            # SPEC v1.2: do not force pred.aspect back to gold — the model
            # is responsible for predicting both fields and the evaluator
            # scores aspect and joint accuracy from what was returned.
            rec = _make_record(sample, predictor, pa, ps, raw, parse_ok, latency_ms)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            latencies_ms.append(latency_ms)

            if progress_every and i % progress_every == 0:
                print(
                    f"[evaluate] {i}/{len(samples)} samples "
                    f"({i / len(samples):.0%}) — last latency {latency_ms:.1f} ms",
                    file=sys.stderr,
                )

    overall_elapsed_s = (time.perf_counter_ns() - overall_t0) / 1e9
    gpu_peak_gb = _gpu_mem_peak_gb()

    # Re-validate the file we just wrote, then score it.
    schema_errors = validate_predictions_file(pred_path)
    if schema_errors:
        raise RuntimeError(
            f"runner produced an invalid predictions file at {pred_path}:\n  "
            + "\n  ".join(schema_errors)
        )
    metrics = score_predictions(pred_path)

    efficiency: dict[str, Any] = {
        "params_million": config.params_million,
        "gpu_mem_peak_gb": round(gpu_peak_gb, 3) if gpu_peak_gb is not None else None,
        "avg_latency_ms": round(mean(latencies_ms), 3),
        "median_latency_ms": round(median(latencies_ms), 3),
        "p95_latency_ms": round(_percentile(latencies_ms, 95), 3),
        "stddev_latency_ms": (
            round(pstdev(latencies_ms), 3) if len(latencies_ms) > 1 else 0.0
        ),
        "throughput_qps": (
            round(len(samples) / overall_elapsed_s, 3) if overall_elapsed_s > 0 else None
        ),
        "training_hours": config.training_hours,
        "hardware": config.hardware,
        "precision": config.precision,
        "batch_size_inference": config.batch_size_inference,
        "warmup_samples_skipped_in_timing": False,  # warmup runs aren't timed at all
        "warmup_samples_run": warmup_n,
    }
    efficiency.update(config.extra_efficiency)
    metrics["efficiency"] = efficiency

    write_metrics(metrics, metrics_path)

    print(
        f"[evaluate] wrote {pred_path} ({len(samples)} samples) and {metrics_path}",
        file=sys.stderr,
    )
    return metrics


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


# --- CLI -------------------------------------------------------------------


def _import_predictor(spec: str, kwargs_json: str | None) -> Predictor:
    """Resolve ``module.path:ClassName`` into an instantiated Predictor."""
    if ":" not in spec:
        raise ValueError(
            "--predictor must be of the form 'module.path:ClassName' "
            f"(got {spec!r})"
        )
    module_path, class_name = spec.split(":", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    kwargs: dict[str, Any] = json.loads(kwargs_json) if kwargs_json else {}
    instance = cls(**kwargs)
    if not isinstance(instance, Predictor):
        raise TypeError(
            f"{class_name} does not implement the Predictor protocol "
            "(needs method/paradigm/backbone attributes and a predict method)"
        )
    return instance


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Run a Predictor against a unified test set, time it, and emit "
            "predictions JSONL + metrics JSON conforming to docs/SURVEY_SPEC.md."
        )
    )
    p.add_argument(
        "--predictor",
        required=True,
        help="Predictor class spec, e.g. 'predictors.lcf_bert:LCFBertPredictor'.",
    )
    p.add_argument(
        "--predictor-kwargs",
        default=None,
        help="Optional JSON dict of kwargs to pass to the predictor constructor.",
    )
    p.add_argument(
        "--test-set",
        required=True,
        type=Path,
        help="Path to the unified test JSONL file (see SPEC §4).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Where to write predictions/ and metrics/ subdirs (default: results/).",
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--hardware", default="1x T4 16GB")
    p.add_argument("--precision", default="fp16")
    p.add_argument("--batch-size-inference", type=int, default=1)
    p.add_argument("--params-million", type=float, default=None)
    p.add_argument("--training-hours", type=float, default=None)
    p.add_argument("--progress-every", type=int, default=100)
    args = p.parse_args(list(argv) if argv is not None else None)

    predictor = _import_predictor(args.predictor, args.predictor_kwargs)
    config = RunConfig(
        test_set=args.test_set,
        output_dir=args.output_dir,
        warmup=args.warmup,
        hardware=args.hardware,
        precision=args.precision,
        batch_size_inference=args.batch_size_inference,
        params_million=args.params_million,
        training_hours=args.training_hours,
    )
    metrics = run_evaluation(predictor, config, progress_every=args.progress_every)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())