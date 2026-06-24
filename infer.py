
import sys
sys.modules["torch_xla"] = None

import os
import json
import argparse
import subprocess
from pathlib import Path
from typing import Union, Dict, Optional
from datetime import datetime

import torch
import transformers
import swift
from swift.llm import infer_main

print("torch =", torch.__version__)
print("transformers =", transformers.__version__)
print("swift =", swift.__version__)


def find_latest_version_dir(model_dir: Path) -> Path:
    """find dir version with the format vX-YYYYMMDD-HHMMSS."""
    candidates = []
    for d in model_dir.iterdir():
        if not (d.is_dir() and d.name.startswith("v")):
            continue
        try:
            version, date_str, time_str = d.name.split("-", 2)
            ts = datetime.strptime(f"{date_str} {time_str}", "%Y%m%d %H%M%S")
            candidates.append((int(version[1:]), ts, d))
        except ValueError:
            continue
    if not candidates:
        raise FileNotFoundError(f"Not found version dir in {model_dir}")
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[-1][2]


def find_best_checkpoint(model_dir: Path) -> Path:
    """Read logging.jsonl to get best checkpoint path."""
    version_dir = find_latest_version_dir(model_dir)
    log_file = version_dir / "logging.jsonl"
    if not log_file.exists():
        raise FileNotFoundError(f"Not found {log_file}")

    best_checkpoint = None
    with log_file.open() as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("best_model_checkpoint"):
                best_checkpoint = entry["best_model_checkpoint"]
    if not best_checkpoint:
        raise ValueError(f"not found best_model_checkpoint trong {log_file}")

    checkpoint_name = Path(best_checkpoint).name
    ckpt_dir = version_dir / checkpoint_name
    print(f"Best checkpoint: {ckpt_dir}")
    return ckpt_dir


def fix_image_paths(json_path: Path, dataset_dir: Path) -> None:
    """ relative path in JSONL to absolute."""
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)

    fixed = 0
    for record in data:
        for key in ("images", "videos", "audios"):
            if key not in record:
                continue
            paths = []
            for p in record[key]:
                if os.path.isabs(p):
                    paths.append(p)
                else:
                    paths.append(str(dataset_dir / p))
                    fixed += 1
            record[key] = paths

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"[fix_image_paths] {json_path.name}: {fixed} path -> absolute")


def build_guided_decoding(dataset_dir: Path, guided_decoding: Optional[Union[Dict, str]]) -> Optional[Dict]:
    if not guided_decoding:
        return None
    if isinstance(guided_decoding, str):
        with (dataset_dir / guided_decoding).open() as f:
            return {"json": json.load(f)}
    return guided_decoding


def write_generation_config(model_dir: Path, guided_decoding: Optional[Dict]) -> None:
    if not guided_decoding:
        return
    config_path = model_dir / "generation_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    config.setdefault("guided_decoding", {}).update(guided_decoding)
    config_path.write_text(json.dumps(config, indent=2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--model_type", required=True)
    parser.add_argument("--dataset_gcs", required=True)
    parser.add_argument("--test_data_path", default="conversations_test_swift_format.json")
    parser.add_argument("--results_gcs", required=True)
    parser.add_argument("--guided_decoding", default="null")
    args = parser.parse_args()

    dataset_dir = Path("/tmp/dataset")
    output_dir = Path("/tmp/output")
    models_dir = Path("/tmp/models")
    for d in (dataset_dir, output_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)

    # download test dataset (JSON + images/) from GCS
    subprocess.run(["gsutil", "-m", "rsync", "-r", args.dataset_gcs, str(dataset_dir)], check=True)

    test_data_path = dataset_dir / args.test_data_path
    result_path = output_dir / "results.jsonl"
    fix_image_paths(test_data_path, dataset_dir)

    guided_decoding = build_guided_decoding(dataset_dir, json.loads(args.guided_decoding))

    argv = [
        "--result_path", str(result_path),
        "--max_length", "4096",
        "--max_new_tokens", "2048",  
        "--max_pixels", "1048576",     
        "--load_data_args", "false",
        "--val_dataset", str(test_data_path),
        "--use_hf", "true",
        "--infer_backend", "pt",
        "--temperature", "0",
    ]

    #  Load model: GCS (LoRA adapter fine-tuned) or HuggingFace Hub (base model)
    if args.model_id.startswith("gs://"):
        model_dir = models_dir / "finetuned"
        model_dir.mkdir(exist_ok=True)
        subprocess.run(
            ["gsutil", "-m", "cp", "-r", args.model_id.rstrip("/") + "/*", str(model_dir)],
            check=True,
        )
        ckpt_dir = find_best_checkpoint(model_dir)
        write_generation_config(ckpt_dir, guided_decoding)
        argv += ["--adapters", str(ckpt_dir), "--merge_lora", "true"]
    else:
        from huggingface_hub import snapshot_download
        model_dir = models_dir / "base"
        snapshot_download(repo_id=args.model_id, local_dir=str(model_dir))
        write_generation_config(model_dir, guided_decoding)
        argv += ["--model_type", args.model_type, "--model", str(model_dir)]

    # Run inference
    infer_main(argv)
    print(f" Inference completed: {result_path}")

    #  Upload result at GCS
    subprocess.run(["gsutil", "-m", "cp", str(result_path), args.results_gcs.rstrip("/") + "/"], check=True)
    print(f" Uploaded to {args.results_gcs}/results.jsonl")


if __name__ == "__main__":
    main()