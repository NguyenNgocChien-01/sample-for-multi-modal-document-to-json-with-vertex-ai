import sys
sys.modules['torch_xla'] = None

import os
import json
import argparse
import subprocess
import faulthandler
faulthandler.enable() 
import transformers
import torch
import swift

print("torch =", torch.__version__)
print("transformers =", transformers.__version__)
print("swift =", swift.__version__)
from swift.llm import sft_main


def fix_image_paths(json_path: str, dataset_dir: str) -> None:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    fixed_count = 0
    for record in data:
        for key in ("images", "videos", "audios"):
            if key not in record:
                continue
            new_paths = []
            for p in record[key]:
                if os.path.isabs(p):
                    new_paths.append(p)
                else:
                    new_paths.append(os.path.join(dataset_dir, p))
                    fixed_count += 1
            record[key] = new_paths

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"[fix_image_paths] {json_path}: {fixed_count} path "
          f" -> (dataset_dir={dataset_dir})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--training_data_gcs", type=str)
    parser.add_argument("--train_data_path", type=str, default="conversations_train_swift_format.json")
    parser.add_argument("--validation_data_path", type=str, default="conversations_dev_swift_format.json")
    args = parser.parse_args()

    dataset_dir = "/tmp/dataset"
    output_dir = "/tmp/output"
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    subprocess.run(
        ["gsutil", "-m", "cp", "-r", args.training_data_gcs + "/*", dataset_dir],
        check=True
    )

    train_data_local_path = os.path.join(dataset_dir, args.train_data_path)
    validation_data_local_path = os.path.join(dataset_dir, args.validation_data_path)

    fix_image_paths(train_data_local_path, dataset_dir)
    fix_image_paths(validation_data_local_path, dataset_dir)

    argv = [
        "--model_type", args.model_type,
        "--model", args.model_id,
        "--model_revision", "main",
        "--train_type", "lora",
        "--use_dora", "true",
        "--output_dir", output_dir,
        "--max_length", "2048",
        "--max_pixels", "262144",
        "--dataset", train_data_local_path,
        "--val_dataset", validation_data_local_path,
        "--save_steps", "50",
        "--logging_steps", "5",
        "--num_train_epochs", "4",
        "--lora_dtype", "bfloat16",
        "--per_device_train_batch_size", "1",
        "--per_device_eval_batch_size", "1",
        "--gradient_accumulation_steps", "4",
        "--learning_rate", "1e-4",
        "--gradient_checkpointing", "true",
        "--target_modules", "all-linear",
        "--warmup_ratio", "0.05",
        "--save_total_limit", "3",
        "--freeze_vit", "true",
        "--freeze_llm", "false",
        "--freeze_aligner", "true",
        "--dataloader_num_workers", "0"
    ]

    result = sft_main(argv)
    best_checkpoint = result["best_model_checkpoint"]
    print(f"Best checkpoint (local): {best_checkpoint}")

    output_gcs_uri = os.environ.get("AIP_MODEL_DIR", "gs://electric-bill-dataset-gcs/output")
    subprocess.run(
        ["gsutil", "-m", "cp", "-r", output_dir + "/*", output_gcs_uri],
        check=True
    )
    print(f"Uploaded to {output_gcs_uri}")

if __name__ == "__main__":
    main()