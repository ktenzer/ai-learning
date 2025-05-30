import os, time, json, pickle, re
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from items import Item
from testing import Tester

# ─────────────────── Setup ─────────────────── #
load_dotenv(override=True)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "YOUR-KEY")

openai = OpenAI()

# (nothing else in the data-prep section changes)

# ────── when you create the job ──────
wandb_integration = {"type": "wandb", "wandb": {"project": "gpt-pricer"}}

# ─────────────────── Data ─────────────────── #
with open("train.pkl", "rb") as f:
    train = pickle.load(f)
with open("test.pkl", "rb") as f:
    test  = pickle.load(f)

train_items      = train[:200]
validation_items = train[200:250]

# ─────────────────── Helper functions ─────────────────── #
def messages_for(item: Item) -> list[dict]:
    sys_msg  = "You estimate prices of items. Reply only with the price, no explanation"
    user_msg = item.test_prompt().replace(" to the nearest dollar", "").replace("\n\nPrice is $", "")
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user",   "content": user_msg},
        # assistant message ONLY in training file
        # (for inference we'll leave it blank)
    ]

def make_jsonl(items, include_answer: bool) -> str:
    rows = []
    for it in items:
        msgs = messages_for(it)
        if include_answer:
            msgs.append({"role": "assistant", "content": f"Price is ${it.price:.2f}"})
        rows.append(json.dumps({"messages": msgs}))
    return "\n".join(rows)

def dump_jsonl(items, fname, include_answer=True):
    with open(fname, "w") as f:
        f.write(make_jsonl(items, include_answer))

def upload_jsonl(fname) -> str:
    with open(fname, "rb") as f:
        return openai.files.create(file=f, purpose="fine-tune").id

def wait_for_job(job_id: str, poll=5):
    while True:
        job = openai.fine_tuning.jobs.retrieve(job_id)
        if job.status in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(poll)

def get_price(txt: str) -> float:
    txt = txt.replace("$", "").replace(",", "")
    m = re.search(r"[-+]?\d*\.\d+|\d+", txt)
    return float(m.group()) if m else 0.0

# ─────────────────── 1. Build / Upload training files (once) ─────────────────── #
dump_jsonl(train_items,      "train.jsonl", include_answer=True)
dump_jsonl(validation_items, "validation.jsonl", include_answer=True)

train_file_id      = upload_jsonl("train.jsonl")
validation_file_id = upload_jsonl("validation.jsonl")

print("Training file   :", train_file_id)
print("Validation file :", validation_file_id)

# ─────────────────── 2. Launch or reuse fine-tune ─────────────────── #
BASE_MODEL = "gpt-3.5-turbo-0125"   # <-- officially fine-tuneable

# Try to find an existing successful FT job with the same files + suffix
existing = [
    j for j in openai.fine_tuning.jobs.list(limit=20).data
    if j.status == "succeeded"
       and j.training_file == train_file_id
       and j.validation_file == validation_file_id
       and j.model == BASE_MODEL
       and j.suffix == "pricer"
]
if existing:
    job = existing[0]
    print("✓ Reusing existing fine-tuned model:", job.fine_tuned_model)
else:
    job = openai.fine_tuning.jobs.create(
        training_file   = train_file_id,
        validation_file = validation_file_id,
        model           = BASE_MODEL,
        hyperparameters = {"n_epochs": 1},
        seed            = 42,
        suffix          = "pricer",
        integrations    = [wandb_integration],
    )

    print("▶ Started new job:", job.id)
    job = wait_for_job(job.id)              # blocks until finished
    print("✓ Job finished with status:", job.status)

if job.status != "succeeded":
    raise RuntimeError(f"Fine-tune failed: {job.status}")

fine_tuned_model_name = job.fine_tuned_model   # e.g. ft:gpt-3.5-turbo-0125:org:pricer:xyz
print("→ Model ready:", fine_tuned_model_name)

# ─────────────────── 3. Inference helper ─────────────────── #
def prompt_for(item: Item) -> list[dict]:
    # ask without the known answer
    sys_msg  = "You estimate prices of items. Reply only with the price, no explanation"
    user_msg = item.test_prompt().replace(" to the nearest dollar", "").replace("\n\nPrice is $", "")
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user",   "content": user_msg},
    ]

def gpt_fine_tuned(item: Item) -> float:
    resp = openai.chat.completions.create(
        model    = fine_tuned_model_name,
        messages = prompt_for(item),
        max_tokens = 8,
        seed = 42,
    )
    answer = resp.choices[0].message.content
    return get_price(answer)

# ─────────────────── 4. Quick smoke-test ─────────────────── #
print("Sample gt price:", test[0].price, "\nPrompt →", test[0].test_prompt())
print("Model answer   :", gpt_fine_tuned(test[0]))

# run your evaluation
Tester.test(gpt_fine_tuned, test)