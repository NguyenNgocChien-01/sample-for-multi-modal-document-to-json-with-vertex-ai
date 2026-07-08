"""
utils/inference_report.py
Inference performance report: timing / tokens / RAM-GPU.
Import from notebook 04 and call functions in the order shown at the bottom.
"""

import re
import time
import subprocess
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoProcessor
from google.cloud import aiplatform
from google.cloud import logging as gcp_logging
from google.cloud import monitoring_v3
from google.api_core.exceptions import ResourceExhausted

JOB_NAME_PREFIX = "infer-"

_TQDM_LINE = re.compile(
    r"(?P<percent>\d+)%\|.*?\|\s*(?P<done>\d+)/(?P<total>\d+)\s*"
    r"\[(?P<elapsed>[\d:]+)<[\d:]+,\s*[\d.]+(?:it/s|s/it)\]"
)
_LOG_MARKERS = {
    "env_ready":        re.compile(r"^torch = "),
    "checkpoint_ready": re.compile(r"^Best checkpoint:"),
    "inference_done":   re.compile(r"Inference completed:"),
    "upload_done":      re.compile(r"Uploaded to"),
}


# ── Job lookup ────────────────────────────────────────────────────────────────

def find_latest_inference_job(project: str, location: str, gcs_results_dir: str,
                               name_prefix: str = JOB_NAME_PREFIX):
    """Return the most recent CustomJob whose display_name belongs to this bucket."""
    bucket_name = gcs_results_dir.split("/")[2]
    all_jobs = aiplatform.CustomJob.list(project=project, location=location)
    matching = [
        j for j in all_jobs
        if j.display_name.startswith(name_prefix)
        and bucket_name in j.display_name.replace(":--", "-").replace("--", "-")
    ]
    if not matching:
        raise RuntimeError(f"No CustomJob found for bucket '{bucket_name}' with prefix '{name_prefix}'")
    return max(matching, key=lambda j: j.create_time)


def find_job_by_id(project: str, location: str, job_id: str):
    resource_name = f"projects/{project}/locations/{location}/customJobs/{job_id}"
    return aiplatform.CustomJob.get(resource_name)


def download_job_results(job_display_name: str, gcs_results_dir: str, local_path: Path) -> Path:
    gcs_uri = f"{gcs_results_dir}/{job_display_name}/results.jsonl"
    subprocess.run(f'gsutil cp "{gcs_uri}" "{local_path}"', shell=True, check=True)
    return local_path


# ── Cloud Logging → phase timings ────────────────────────────────────────────

def _hms_to_seconds(hms: str) -> float:
    parts = [float(p) for p in hms.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, s = parts
    return h * 3600 + m * 60 + s


def fetch_job_log_lines(job_id: str, project: str, start_time, end_time,
                         max_retries: int = 6) -> list:
    client = gcp_logging.Client(project=project)
    log_filter = (
        f'resource.type="ml_job" AND resource.labels.job_id="{job_id}" '
        f'AND timestamp>="{start_time.isoformat()}" AND timestamp<="{end_time.isoformat()}"'
    )
    for attempt in range(max_retries):
        try:
            entries = client.list_entries(
                filter_=log_filter,
                order_by=gcp_logging.ASCENDING,
                page_size=1000,
            )
            lines = []
            for entry in entries:
                payload = entry.payload
                text = payload.get("message") if isinstance(payload, dict) else str(payload)
                lines.append((entry.timestamp, text or json.dumps(payload, ensure_ascii=False)))
            return lines
        except ResourceExhausted:
            wait = 20 * (attempt + 1)
            print(f"Cloud Logging quota exceeded, retrying in {wait}s ({attempt + 1}/{max_retries})")
            time.sleep(wait)
    raise RuntimeError("Cloud Logging quota still exceeded after all retries — wait a few minutes")


def parse_phase_timings(log_lines: list) -> dict:
    """Time between print() markers in infer.py: env_ready → checkpoint → done → upload."""
    marker_time = {}
    for timestamp, text in log_lines:
        for name, pattern in _LOG_MARKERS.items():
            if name not in marker_time and pattern.search(text):
                marker_time[name] = timestamp

    def _gap(a, b):
        if a in marker_time and b in marker_time:
            return (marker_time[b] - marker_time[a]).total_seconds()
        return None

    return {
        "setup_to_checkpoint_seconds":       _gap("env_ready", "checkpoint_ready"),
        "checkpoint_to_infer_done_seconds":  _gap("checkpoint_ready", "inference_done"),
        "infer_done_to_upload_seconds":      _gap("inference_done", "upload_done"),
    }


def parse_per_sample_timings(log_lines: list) -> dict:
    """Per-sample inference time derived from tqdm progress lines SWIFT prints."""
    elapsed_at_step = {}
    for _, text in log_lines:
        m = _TQDM_LINE.search(text)
        if m:
            elapsed_at_step[int(m.group("done"))] = _hms_to_seconds(m.group("elapsed"))

    steps = sorted(elapsed_at_step.items())
    if len(steps) < 2:
        return {}

    per_sample = []
    for (step_a, time_a), (step_b, time_b) in zip(steps, steps[1:]):
        n = step_b - step_a
        if n > 0:
            per_sample.extend([(time_b - time_a) / n] * n)

    if not per_sample:
        return {}

    arr = np.array(per_sample)
    return {
        "sample_time_seconds_min":  float(arr.min()),
        "sample_time_seconds_max":  float(arr.max()),
        "sample_time_seconds_mean": float(arr.mean()),
        "sample_time_seconds_std":  float(arr.std()),
        "total_samples_timed":      len(arr),
    }


# ── Token counting ────────────────────────────────────────────────────────────

def load_tokenizer(model_id: str, base_model_id: str):
    """LoRA fine-tuning doesn't change the tokenizer; gs:// models use the base tokenizer."""
    if model_id.startswith("gs://"):
        return AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    try:
        return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"Could not load tokenizer for '{model_id}' ({e}), falling back to base model")
        return AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)


def read_results(jsonl_path: Path) -> list:
    if not jsonl_path.exists():
        raise FileNotFoundError(f"{jsonl_path} not found")
    with jsonl_path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _summarize(values: list) -> dict:
    arr = np.array(values, dtype=float)
    return {"min": float(arr.min()), "max": float(arr.max()),
            "mean": float(arr.mean()), "std": float(arr.std())}


def compute_output_token_stats(records: list, tokenizer) -> dict:
    counts = [len(tokenizer.encode(r.get("response", ""), add_special_tokens=False)) for r in records]
    stats = _summarize(counts)
    stats["total"] = float(sum(counts))
    stats["n_samples"] = len(records)
    return stats


def compute_input_token_stats(records: list, model_id: str,
                               dataset_gcs_uri: str, local_dir: Path) -> dict:
    """
    Estimate input token counts using the model processor.
    Requires torchvision for Qwen2-VL image processing.
    Falls back to text-only token count if torchvision is not installed.
    """
    local_dir.mkdir(exist_ok=True)
    subprocess.run(f'gsutil -m cp -r "{dataset_gcs_uri}/*" "{local_dir}"', shell=True, check=True)

    try:
        import torchvision  # noqa: F401 — required by Qwen2VLProcessor
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        use_images = True
    except ImportError:
        print("torchvision not found — falling back to text-only token count (install torchvision for accurate stats)")
        from transformers import AutoTokenizer as _Tok
        processor = _Tok.from_pretrained(model_id, trust_remote_code=True)
        use_images = False

    counts = []
    for record in records:
        if use_images:
            prompt_text = processor.apply_chat_template(
                record.get("messages", []), tokenize=False, add_generation_prompt=False
            )
            image_paths = [str(local_dir / Path(p).name) for p in record.get("images", [])]
            inputs = processor(text=[prompt_text], images=image_paths or None, return_tensors="pt")
            counts.append(inputs["input_ids"].shape[1])
        else:
            # text-only fallback: join all message content
            text = " ".join(
                m.get("content", "") for m in record.get("messages", [])
                if isinstance(m.get("content"), str)
            )
            counts.append(len(processor.encode(text, add_special_tokens=False)))

    stats = _summarize(counts)
    stats["text_only_fallback"] = not use_images
    return stats


# ── Cloud Monitoring → GPU / CPU ──────────────────────────────────────────────

def fetch_gpu_cpu_stats(job_id: str, project: str, start_time, end_time) -> dict:
    if start_time is None or end_time is None:
        print("Missing start_time/end_time, skipping Cloud Monitoring")
        return {}

    client = monitoring_v3.MetricServiceClient()
    interval = monitoring_v3.TimeInterval({
        "start_time": {"seconds": int(start_time.timestamp())},
        "end_time":   {"seconds": int(end_time.timestamp()) + 1},
    })
    metrics = {
        "cpu_utilization": "ml.googleapis.com/training/cpu/utilization",
        "gpu_utilization": "ml.googleapis.com/training/accelerator/utilization",
        "gpu_memory":      "ml.googleapis.com/training/accelerator/memory/utilization",
    }
    stats = {}
    for name, metric_type in metrics.items():
        try:
            series = client.list_time_series(request={
                "name":   f"projects/{project}",
                "filter": (
                    f'metric.type="{metric_type}" AND resource.type="cloudml_job" '
                    f'AND resource.labels.job_id="{job_id}"'
                ),
                "interval": interval,
                "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
            })
            values = [float(p.value.double_value or p.value.int64_value)
                      for ts in series for p in ts.points]
            if values:
                stats[name] = _summarize(values)
            else:
                print(f"No data for metric '{name}'")
        except Exception as e:
            print(f"Skipping metric '{name}': {e}")
    return stats


# ── Assemble + save ───────────────────────────────────────────────────────────

def build_report_row(job, job_id: str, model_id: str, total_job_seconds,
                      tokens_per_sec, phase_timings: dict, sample_timings: dict,
                      output_token_stats: dict, input_token_stats: dict,
                      resource_stats: dict) -> dict:
    row = {
        "run_timestamp":     datetime.now(timezone.utc).isoformat(),
        "job_display_name":  job.display_name,
        "job_id":            job_id,
        "model_id":          model_id,
        "total_job_seconds": total_job_seconds,
        "tokens_per_sec":    tokens_per_sec,
        **phase_timings,
        **sample_timings,
    }
    for k, v in output_token_stats.items():
        row[f"output_tokens_{k}"] = v
    for k, v in input_token_stats.items():
        row[f"input_tokens_{k}"] = v
    for metric_name, stats in resource_stats.items():
        for k, v in stats.items():
            row[f"{metric_name}_{k}"] = v
    return row


def append_to_report_csv(row: dict, csv_path: Path) -> pd.DataFrame:
    new_df = pd.DataFrame([row])
    if csv_path.exists() and csv_path.stat().st_size > 0:
        df = pd.concat([pd.read_csv(csv_path), new_df], ignore_index=True)
    else:
        df = new_df
    df.to_csv(csv_path, index=False)
    return df


# ── Usage (call from notebook 04 in this order) ───────────────────────────────
#
#   from utils.inference_report import *
#
#   job = find_latest_inference_job(PROJECT_ID, region, gcs_results)
#   job_id = job.resource_name.split("/")[-1]
#   total_job_seconds = (job.end_time - job.start_time).total_seconds()
#   download_job_results(job.display_name, gcs_results, RESULTS_JSONL)
#
#   log_lines     = fetch_job_log_lines(job_id, PROJECT_ID, job.start_time, job.end_time)
#   phase_timings = parse_phase_timings(log_lines)
#   sample_timings = parse_per_sample_timings(log_lines)
#
#   results    = read_results(RESULTS_JSONL)
#   tokenizer  = load_tokenizer(model_config.model_id, base_model_config.model_id)
#   output_token_stats = compute_output_token_stats(results, tokenizer)
#   input_token_stats  = compute_input_token_stats(
#       results, base_model_config.model_id, dataset_gcs_uri, LOCAL_DATASET_DIR
#   )
#   tokens_per_sec = (output_token_stats["total"] / phase_timings["checkpoint_to_infer_done_seconds"]
#                     if phase_timings.get("checkpoint_to_infer_done_seconds") else None)
#
#   resource_stats = fetch_gpu_cpu_stats(job_id, PROJECT_ID, job.start_time, job.end_time)
#
#   row = build_report_row(job, job_id, model_config.model_id, total_job_seconds, tokens_per_sec,
#                          phase_timings, sample_timings, output_token_stats,
#                          input_token_stats, resource_stats)
#   history_df = append_to_report_csv(row, REPORT_CSV)