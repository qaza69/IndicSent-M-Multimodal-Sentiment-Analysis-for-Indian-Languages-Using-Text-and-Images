"""Train all XLM-RoBERTa text and multimodal models on the full dataset.

Order is optimized for maximum speed:
  1. text Task A        -> Extracts & caches XLM-RoBERTa features (~6-8 min on CPU), then trains Task A
  2. text Task B        -> Reuses cached XLM-RoBERTa features (~2 min)
  3. multimodal Task A  -> Reuses cached XLM-RoBERTa + cached VGG16 features (~10-15 min)
  4. multimodal Task B  -> Reuses cached XLM-RoBERTa + cached VGG16 features (~10-15 min)
"""
import subprocess, sys, time, os
import config

print(f"=== Starting Training with Text Encoder: {config.TEXT_MODEL_NAME} ===", flush=True)

RUNS = [
    ("text",       "A"),
    ("text",       "B"),
    ("multimodal", "A"),
    ("multimodal", "B"),
]

total_start = time.time()
results = []

for i, (model, task) in enumerate(RUNS, 1):
    header = f"\n{'='*60}\n[{i}/{len(RUNS)}] Training: model={model}  task={task}  (Encoder: {config.TEXT_MODEL_NAME})\n{'='*60}"
    print(header, flush=True)
    t0 = time.time()

    cmd = [
        sys.executable, "train.py",
        "--model", model,
        "--task", task,
    ]

    proc = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    elapsed = time.time() - t0
    status = "SUCCESS" if proc.returncode == 0 else f"FAILED (code {proc.returncode})"
    results.append((model, task, status, elapsed))
    print(f"\n>> [{i}/{len(RUNS)}] {model} Task {task}: {status} in {elapsed:.1f}s ({elapsed/60:.1f}m)", flush=True)

total = time.time() - total_start
print(f"\n{'='*60}")
print(f"ALL TRAINING COMPLETE  —  Total time: {total/60:.1f} minutes")
print(f"{'='*60}")
print(f"{'Model':<14} {'Task':<6} {'Status':<20} {'Time':>10}")
print("-" * 54)
for model, task, status, elapsed in results:
    print(f"{model:<14} {task:<6} {status:<20} {elapsed:>8.1f}s")
print(f"\nResults saved in: outputs/")
