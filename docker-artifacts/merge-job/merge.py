import sys, traceback, torch, json
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel, LoraConfig

ADAPTER_DIR = "/workspace/checkpoint"
OUTPUT_DIR = "/workspace/merged-model"
BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

try:
    # Fix adapter_config: thay target_modules regex thành list cụ thể
    config_path = Path(ADAPTER_DIR) / "adapter_config.json"
    config = json.loads(config_path.read_text())
    print(f"Original target_modules: {config['target_modules']}", flush=True)
    
    # Các module thực sự trong model
    config["target_modules"] = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    config["use_dora"] = False  # peft cũ không support DoRA tốt
    config_path.write_text(json.dumps(config, indent=2))
    print(f"Fixed target_modules: {config['target_modules']}", flush=True)

    print("Loading base model...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    print("Loading adapter...", flush=True)
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    print("Merging...", flush=True)
    model = model.merge_and_unload()
    print("Saving...", flush=True)
    model.save_pretrained(OUTPUT_DIR, safe_serialization=True, max_shard_size="5GB")
    AutoProcessor.from_pretrained(BASE_MODEL).save_pretrained(OUTPUT_DIR)
    print("Done!", flush=True)
except Exception:
    traceback.print_exc()
    sys.exit(1)
