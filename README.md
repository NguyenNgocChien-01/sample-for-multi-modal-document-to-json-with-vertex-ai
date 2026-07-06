# Document-to-JSON Vision LLM — Fine-tune & Deploy on GCP

Fine-tunes Qwen2.5-VL to extract structured JSON from documents (electric
bills, Australian identity documents), then deploys to Cloud Run. Replaces
the original SageMaker deployment stage with a GCP-native one.

## Pipeline

```
Dataset → Fine-tune (Vertex AI) → Merge LoRA (Cloud Build)
→ Bake model into image (Cloud Build) → Deploy (Cloud Run)
```

The merged model is baked into the Docker image at build time — no GCS
download at container startup, so cold starts are fast. Each model gets
its own image tag (`inference-bill`, `inference-identity`), since the
weights live inside the image now.

<img src="./images/workflow-overview.svg" alt="Workflow: dataset to fine-tune to merge to bake to deploy" width="600"/>

## Results

<!-- TODO: fill in once evaluation / test runs are available -->

| Model | Exact Match | CER | Notes |
|---|---:|---:|---|
| Electric bill | *TODO* | *TODO* | from `05_evaluate_model.ipynb` |
| Identity document | *TODO* | *TODO* | from `05_evaluate_model.ipynb` |

<!-- TODO: insert a sample input/output screenshot, e.g. document image next
     to extracted JSON, from 07_consume_model_cloudrun.ipynb's display_results() -->
<img src="./images/sample-extraction-result.png" alt="Sample document and extracted JSON side by side" width="700"/>

## Notebooks

| # | Notebook | Purpose |
|---|---|---|
| 01 | `gcs_config.json` | Shared config: project, bucket, region |
| 02 | `02_create_custom_dataset_swift.ipynb` | Build Swift training data |
| 03 | `03_finetune_swift.ipynb` | Fine-tune on Vertex AI |
| 04 | `04_run_batch_inference.ipynb` | Batch inference |
| 05 | `05_evaluate_model.ipynb` | Evaluation metrics |
| 06 | `06_deploy_pipeline_bill.ipynb` / `06_deploy_pipeline_identity.ipynb` | Merge → bake → deploy → test, one per model |
| 07 | `07_consume_model_cloudrun.ipynb` | Call the deployed model via REST |

## Config

`gcs_config.json`:
```json
{
  "project_id": "",
  "bucket_name": "",
  "region": "",
  "gcs_output_prefix": "",
  "gcs_model_dir": ""
}
```

Only `project_id` and `region` are read by the deploy notebooks — the rest
(checkpoint path, service name, model name) is set per-model at the top of
each `06_deploy_pipeline_*.ipynb`.

## Running a deploy notebook

Each cell prints the command instead of running it (`# !{cmd}` is commented
out) — copy the printed command into **Cloud Shell**, not PowerShell.
PowerShell parses quotes/commas differently and breaks
`--set-env-vars`/`--startup-probe`.

1. Merge LoRA into base model (Cloud Build) — skip if merged model exists
2. Bake merged model into the image (Cloud Build)
3. Deploy to Cloud Run — 1x L4 GPU, 8 vCPU, 32GB, `--min-instances 0`
4. Grant public invoker access (testing only)
5. Get service URL
6. `curl .../health` to confirm it's up
7. Cleanup — delete, scale to zero, or list services

## Troubleshooting

- **Run in Cloud Shell**, not PowerShell.
- Model changed → **rebuild the image** (it's baked in, not downloaded at runtime).
- `gsutil cp` fails in Cloud Build → grant `roles/storage.objectViewer` to the Cloud Build service account on the source bucket.
- Startup probe timeout → already handled via `failureThreshold=60, periodSeconds=10` in the deploy script (60 is Cloud Run's hard cap — don't set it higher).

## Links

- Cloud Build: https://console.cloud.google.com/cloud-build/builds?project=first-orc-chien
- Cloud Run: https://console.cloud.google.com/run?project=first-orc-chien
- Artifact Registry: https://console.cloud.google.com/artifacts?project=first-orc-chien