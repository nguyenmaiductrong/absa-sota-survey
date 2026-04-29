from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Protocol, runtime_checkable

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
    method: str
    paradigm: str
    backbone: str

    def predict(self, text: str, aspect: str | None = None) -> tuple[str, str, str]:
        ...


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
    samples: list[TestSample] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
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
                raise ValueError(f"{path}:{lineno}: missing required field {exc!s}") from exc

    if not samples:
        raise ValueError(f"empty test set: {path}")

    datasets = {s.dataset for s in samples}
    tasks = {s.task for s in samples}
    languages = {s.language for s in samples}
    if len(datasets) != 1:
        raise ValueError(f"mixed datasets in {path}: {sorted(datasets)}")
    if len(tasks) != 1:
        raise ValueError(f"mixed tasks in {path}: {sorted(tasks)}")
    if len(languages) != 1:
        raise ValueError(f"mixed languages in {path}: {sorted(languages)}")
    if next(iter(datasets)) not in ALLOWED_DATASETS:
        raise ValueError(f"unknown dataset {next(iter(datasets))!r}")
    if next(iter(tasks)) not in ALLOWED_TASKS:
        raise ValueError(f"unknown task {next(iter(tasks))!r}")
    if next(iter(languages)) not in ALLOWED_LANGUAGES:
        raise ValueError(f"unknown language {next(iter(languages))!r}")
    return samples


def _timed_predict(
    predictor: Predictor,
    text: str,
    aspect: str | None,
) -> tuple[str, str, str, float]:
    t0 = time.perf_counter_ns()
    out = predictor.predict(text, aspect=aspect)
    elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
    if not (isinstance(out, tuple) and len(out) == 3):
        raise TypeError(
            f"{predictor.__class__.__name__}.predict must return "
            "(aspect, sentiment, raw_output)"
        )
    pred_aspect, pred_sentiment, raw = out
    return str(pred_aspect), str(pred_sentiment), str(raw), elapsed_ms


def _call_warmup(predictor: Predictor, text: str, aspect: str | None) -> None:
    warmup_fn = getattr(predictor, "warmup", None)
    if callable(warmup_fn):
        try:
            warmup_fn(text, aspect=aspect)
            return
        except TypeError:
            warmup_fn(text)
            return
    predictor.predict(text, aspect=aspect)


def _gpu_mem_peak_gb() -> float | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024**3)


def _reset_gpu_mem() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


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
    given_aspect: bool = False
    extra_efficiency: dict[str, Any] = field(default_factory=dict)


def _slugify_method(method: str) -> str:
    return method.lower().replace(" ", "_").replace("-", "_").replace("/", "_")


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
    if not isinstance(predictor, Predictor):
        raise TypeError(
            "predictor does not implement the Predictor protocol "
            "(needs method/paradigm/backbone and predict)"
        )

    samples = load_test_set(config.test_set)
    dataset = samples[0].dataset
    method_slug = _slugify_method(predictor.method)
    dataset_slug = DATASET_SLUGS.get(dataset, dataset.lower())

    pred_path = config.output_dir / "predictions" / f"{method_slug}_{dataset_slug}.jsonl"
    metrics_path = config.output_dir / "metrics" / f"{method_slug}_{dataset_slug}.json"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    warmup_n = min(config.warmup, len(samples))
    for sample in samples[:warmup_n]:
        aspect_arg = sample.gold_aspect if config.given_aspect else None
        _call_warmup(predictor, sample.text, aspect_arg)

    _reset_gpu_mem()
    latencies_ms: list[float] = []

    with pred_path.open("w", encoding="utf-8") as fh:
        for i, sample in enumerate(samples, start=1):
            aspect_arg = sample.gold_aspect if config.given_aspect else None
            try:
                pa, ps, raw, latency_ms = _timed_predict(predictor, sample.text, aspect_arg)
            except Exception as exc:  # noqa: BLE001
                pa, ps, raw, latency_ms = (
                    PARSE_ERROR,
                    PARSE_ERROR,
                    f"<<predict raised {type(exc).__name__}: {exc}>>",
                    0.0,
                )

            parse_ok = pa != PARSE_ERROR and ps != PARSE_ERROR
            if not parse_ok:
                pa, ps = PARSE_ERROR, PARSE_ERROR

            rec = _make_record(sample, predictor, pa, ps, raw, parse_ok, latency_ms)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            latencies_ms.append(latency_ms)

            if progress_every and i % progress_every == 0:
                print(
                    f"[evaluate] {i}/{len(samples)} samples "
                    f"({i / len(samples):.0%}) - last latency {latency_ms:.1f} ms",
                    file=sys.stderr,
                )

    gpu_peak_gb = _gpu_mem_peak_gb()

    schema_errors = validate_predictions_file(pred_path)
    if schema_errors:
        raise RuntimeError(
            f"runner produced an invalid predictions file at {pred_path}:\n  "
            + "\n  ".join(schema_errors)
        )

    metrics = score_predictions(pred_path)
    metrics["evaluation_protocol"] = "given_aspect_absc" if config.given_aspect else "text_only_joint"
    metrics["given_aspect"] = bool(config.given_aspect)
    metrics["avg_latency_ms"] = round(mean(latencies_ms), 3) if latencies_ms else 0.0

    efficiency: dict[str, Any] = {
        "params_million": config.params_million,
        "gpu_mem_peak_gb": round(gpu_peak_gb, 3) if gpu_peak_gb is not None else None,
        "training_hours": config.training_hours,
        "hardware": config.hardware,
        "precision": config.precision,
        "batch_size_inference": config.batch_size_inference,
    }
    efficiency.update(config.extra_efficiency)
    metrics["efficiency"] = efficiency

    write_metrics(metrics, metrics_path)
    print(f"[evaluate] wrote {pred_path} and {metrics_path}", file=sys.stderr)
    return metrics


def _import_predictor(spec: str, kwargs_json: str | None) -> Predictor:
    if ":" not in spec:
        raise ValueError(f"--predictor must be 'module.path:ClassName', got {spec!r}")
    module_path, class_name = spec.split(":", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    kwargs: dict[str, Any] = json.loads(kwargs_json) if kwargs_json else {}
    instance = cls(**kwargs)
    if not isinstance(instance, Predictor):
        raise TypeError(f"{class_name} does not implement the Predictor protocol")
    return instance


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a Predictor against a unified JSONL test set and write predictions + metrics."
    )
    parser.add_argument("--predictor", required=True)
    parser.add_argument("--predictor-kwargs", default=None)
    parser.add_argument("--test-set", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--hardware", default="1x T4 16GB")
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--batch-size-inference", type=int, default=1)
    parser.add_argument("--params-million", type=float, default=None)
    parser.add_argument("--training-hours", type=float, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--given-aspect",
        action="store_true",
        help="Pass gold aspect/topic to predictor. Use this for LCF-BERT given-aspect/given-topic ABSC.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

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
        given_aspect=args.given_aspect,
    )
    metrics = run_evaluation(predictor, config, progress_every=args.progress_every)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
