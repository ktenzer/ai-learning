import os, json, pickle, re, time
from datetime import datetime
from pathlib import Path

import torch
from torch import quantization

from accelerate import init_empty_weights
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
import wandb

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

# Config
BASE_MODEL          = "meta-llama/Meta-Llama-3.1-8B"
PROJECT_NAME        = "pricer"
HF_USER             = "ktenzer"
DATASET_NAME        = f"{HF_USER}/pricer-data"
# MAX_SEQUENCE_LENGTH = 182
MAX_SEQUENCE_LENGTH = 182

LORA_R         = 32
LORA_ALPHA     = 64
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
LORA_DROPOUT   = 0.1
EPOCHS         = 1
BATCH_SIZE     = 4
GRAD_ACCUM_STEPS = 1
LEARNING_RATE  = 1e-4
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO   = 0.03
OPTIMIZER      = "paged_adamw_32bit"
LOG_TO_WANDB   = True
SAVE_STEPS     = 500 # added was 2_000
LOG_STEPS      = 50

RUN_NAME         = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
HUB_MODEL_NAME   = f"{HF_USER}/{PROJECT_RUN_NAME}"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Auth
load_dotenv(override=True)
hf_token    = os.getenv("HF_TOKEN")
wandb_token = os.getenv("WANDB_API_KEY")

login(hf_token, add_to_git_credential=True)
if LOG_TO_WANDB:
    wandb.login(key=wandb_token)
    wandb.init(project=PROJECT_NAME, name=RUN_NAME)

# Dataset 
dataset = load_dataset(DATASET_NAME)
train_ds = dataset["train"]

# Tokenizer and fp16 base model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("↓ Loading fp16 base model to CPU …")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)
base_model.to(DEVICE)

# LoRA Config & Trainer
response_template = "Price is $"
collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, response_template=response_template)

lora_cfg = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)

trl_cfg = SFTConfig(
    output_dir=PROJECT_RUN_NAME,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    optim="adamw_torch", 
    max_steps=25, # just to make it complete fast, comment for real training
    save_steps=SAVE_STEPS,
    logging_steps=LOG_STEPS,
    learning_rate=LEARNING_RATE,
    bf16=False,
    fp16=False,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    dataset_text_field="text",
    push_to_hub=True,
    hub_model_id=HUB_MODEL_NAME,
    hub_private_repo=True,
    report_to="wandb" if LOG_TO_WANDB else None,
    run_name=RUN_NAME,
)

trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_ds,
    args=trl_cfg,
    peft_config=lora_cfg,
    data_collator=collator,
)

print("Fine-tuning on MPS …")
trainer.train()
print("Training done")

# Merge LoRA into base weights
print("Merging LoRA adapters into fp16 model …")
merged_model = trainer.model.merge_and_unload()

# Dynamic INT8 quant (CPU)
print("Converting merged model to dynamic INT8 …")
torch.backends.quantized.engine = "qnnpack"             # fix for NoQEngine
merged_model.cpu()                                      # must stay on CPU
int8_model = quantization.quantize_dynamic(
    merged_model, {torch.nn.Linear}, dtype=torch.qint8
)
print(
    f"INT8 model memory: {int8_model.get_memory_footprint()/1e6:.1f} MB "
    f"(was ~{merged_model.get_memory_footprint()/1e6:.1f} MB fp16)"
)

#save_path = Path(PROJECT_RUN_NAME) / "int8-model"
#int8_model.save_pretrained(save_path)

# Ensure save path exists
#save_path = Path(PROJECT_RUN_NAME) / "int8-model"

# Get clean state dict by filtering out non-Tensor entries
#clean_state_dict = {
#    k: v for k, v in int8_model.state_dict().items()
#    if isinstance(v, torch.Tensor)
#}

# Save model config and state dict manually
#int8_model.config.save_pretrained(save_path)
#torch.save(clean_state_dict, save_path / "pytorch_model.bin")


#tokenizer.save_pretrained(save_path)
#print("💾 Saved INT-8 model to", save_path.resolve())

# Push LoRA adapters, not the big fp16 model
trainer.model.push_to_hub(PROJECT_RUN_NAME, private=True)
print(f"Saved to the hub: {PROJECT_RUN_NAME}")


if LOG_TO_WANDB:
    wandb.finish()