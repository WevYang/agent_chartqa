"""
Zero-shot baseline evaluation for ChartQA.

Runs single-pass inference on a val parquet file using vLLM,
scores with the same compute_score() as the GRPO training reward,
and writes a JSON + Markdown report comparable to the training metrics table.

Usage (GPU required):
    python scripts/run_baseline_eval.py \
        --model_path /root/rivermind-data/qwen_2.5_3b_instruct \
        --val_file   datasets/val_small_128.parquet \
        --output_dir docs/baseline_eval \
        --label      baseline_zeroshot

To also eval the fine-tuned v20 checkpoint for a side-by-side comparison:
    python scripts/run_baseline_eval.py \
        --model_path checkpoints/mini_chartQA_toolfixed_growthprompt_modelonly_n4_20step_v20/global_step_21/actor/huggingface \
        --val_file   datasets/val_small_128.parquet \
        --output_dir docs/baseline_eval \
        --label      v20_finetuned

Notes:
- Inference is single-pass (no agentic tool executor loop).
  The model sees the same bbox-augmented prompt used during training.
  tool_score=1 iff the response contains a focus_on_*_with_* call.
- accuracy_score uses the same relaxed numeric matching from refocus.py.
- format_score=1 iff the response contains FINAL ANSWER: or ANSWER:.
- overall = 0.55 * accuracy + 0.15 * format + 0.30 * tool  (same weights as training).
- GPU memory: Qwen2.5-VL-3B-Instruct needs ~8 GB in bf16; runs on RTX 4090.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

# ── repo root on path so we can import reward function ──────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from vllm import LLM, SamplingParams
from vllm.multimodal.utils import encode_image_base64

# reward function (same as training)
from examples.reward_function.refocus import compute_score


# ── helpers ──────────────────────────────────────────────────────────────────

def load_val_data(parquet_path: str):
    df = pd.read_parquet(parquet_path)
    required = {"prompt", "answer", "images"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"parquet missing columns: {missing}")
    return df


def build_vllm_messages(prompt: str, image_bytes: bytes):
    """Convert a ChartQA prompt string + image bytes to vLLM chat messages."""
    import base64
    b64 = base64.b64encode(image_bytes).decode()
    data_url = f"data:image/png;base64,{b64}"
    # Replace the '<image>' placeholder with the actual image token understood by vLLM
    text_content = prompt.replace("<image>", "").strip()
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": text_content},
            ],
        }
    ]


def run_inference(
    model_path: str,
    prompts: List[dict],
    max_new_tokens: int = 512,
    gpu_memory_utilization: float = 0.7,
    max_model_len: int = 4096,
) -> List[str]:
    """Run batch inference with vLLM, return list of generated texts."""
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        limit_mm_per_prompt={"image": 1},
        enforce_eager=True,
    )
    sampling = SamplingParams(
        temperature=0.0,          # greedy for reproducible baseline
        max_tokens=max_new_tokens,
    )
    outputs = llm.chat(prompts, sampling_params=sampling, use_tqdm=True)
    return [o.outputs[0].text for o in outputs]


def safe_mean(vals):
    v = [x for x in vals if x is not None]
    return round(sum(v) / len(v), 4) if v else None


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ChartQA zero-shot baseline eval")
    parser.add_argument("--model_path", required=True,
                        help="HF model dir (baseline = qwen_2.5_3b_instruct, "
                             "finetuned = checkpoints/.../huggingface)")
    parser.add_argument("--val_file", default="datasets/val_small_128.parquet",
                        help="Parquet val set relative to repo root")
    parser.add_argument("--output_dir", default="docs/baseline_eval",
                        help="Directory for JSON + Markdown output")
    parser.add_argument("--label", default="baseline_zeroshot",
                        help="Short label used in filenames and report (e.g. baseline_zeroshot, v20_finetuned)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.70)
    parser.add_argument("--max_model_len", type=int, default=4096)
    args = parser.parse_args()

    # ── resolve paths ────────────────────────────────────────────────────────
    val_path = args.val_file if os.path.isabs(args.val_file) else REPO_ROOT / args.val_file
    model_path = args.model_path if os.path.isabs(args.model_path) else REPO_ROOT / args.model_path
    output_dir = Path(args.output_dir) if os.path.isabs(args.output_dir) else REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[baseline_eval] model  : {model_path}")
    print(f"[baseline_eval] val    : {val_path}")
    print(f"[baseline_eval] label  : {args.label}")
    print(f"[baseline_eval] output : {output_dir}")

    # ── load data ────────────────────────────────────────────────────────────
    df = load_val_data(str(val_path))
    n = len(df)
    print(f"[baseline_eval] {n} samples loaded")

    # ── build vLLM message dicts ─────────────────────────────────────────────
    all_messages = []
    for _, row in df.iterrows():
        img_item = row["images"][0]
        img_bytes = img_item["bytes"] if isinstance(img_item, dict) else bytes(img_item)
        msgs = build_vllm_messages(row["prompt"], img_bytes)
        all_messages.append(msgs)

    # ── inference ────────────────────────────────────────────────────────────
    t0 = time.time()
    predictions = run_inference(
        str(model_path),
        all_messages,
        max_new_tokens=args.max_new_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    elapsed = time.time() - t0
    print(f"[baseline_eval] inference done in {elapsed:.1f}s ({elapsed/n:.2f}s/sample)")

    # ── score ────────────────────────────────────────────────────────────────
    ground_truths = df["answer"].astype(str).tolist()
    scores = compute_score(predictions, ground_truths)   # same as training reward

    # ── per-sample results ────────────────────────────────────────────────────
    records = []
    for i, (row, pred, score) in enumerate(zip(df.itertuples(), predictions, scores)):
        records.append({
            "idx": i,
            "figure_id": getattr(row, "figure_id", ""),
            "query": row.query,
            "ground_truth": row.answer,
            "prediction": pred,
            "reward_overall": score["overall"],
            "reward_accuracy": score["accuracy"],
            "reward_format": score["format"],
            "reward_tool": score["tool"],
        })

    # ── aggregate ─────────────────────────────────────────────────────────────
    overalls  = [r["reward_overall"]  for r in records]
    accs      = [r["reward_accuracy"] for r in records]
    fmts      = [r["reward_format"]   for r in records]
    tools     = [r["reward_tool"]     for r in records]

    half = n // 2
    summary = {
        "label": args.label,
        "model_path": str(model_path),
        "val_file": str(val_path),
        "n_samples": n,
        "inference_elapsed_s": round(elapsed, 2),
        "reward": {
            "overall_mean":           safe_mean(overalls),
            "overall_last_half_mean": safe_mean(overalls[half:]),
            "accuracy_mean":          safe_mean(accs),
            "accuracy_last_half_mean":safe_mean(accs[half:]),
            "format_mean":            safe_mean(fmts),
            "tool_mean":              safe_mean(tools),
        },
    }

    # ── write JSON ────────────────────────────────────────────────────────────
    json_path = output_dir / f"{args.label}_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, ensure_ascii=False, indent=2)
    print(f"[baseline_eval] results -> {json_path}")

    # ── write Markdown ────────────────────────────────────────────────────────
    md_path = output_dir / f"{args.label}_report.md"
    r = summary["reward"]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# ChartQA Baseline Eval — {args.label}\n\n")
        f.write(f"- **Model**: `{model_path}`\n")
        f.write(f"- **Val set**: `{val_path}` ({n} samples)\n")
        f.write(f"- **Inference**: single-pass (no tool executor)\n")
        f.write(f"- **Scoring**: same `compute_score()` weights as GRPO training "
                f"(accuracy×0.55 + format×0.15 + tool×0.30)\n\n")
        f.write("## Aggregate Metrics\n\n")
        f.write("| Metric | Value |\n|--------|-------|\n")
        f.write(f"| reward/overall mean        | {r['overall_mean']:.4f} |\n")
        f.write(f"| reward/overall last half   | {r['overall_last_half_mean']:.4f} |\n")
        f.write(f"| reward/accuracy mean       | {r['accuracy_mean']:.4f} |\n")
        f.write(f"| reward/accuracy last half  | {r['accuracy_last_half_mean']:.4f} |\n")
        f.write(f"| reward/format mean         | {r['format_mean']:.4f} |\n")
        f.write(f"| reward/tool mean           | {r['tool_mean']:.4f} |\n\n")
        f.write("## Comparison with GRPO Training Runs\n\n")
        f.write("| Exp | reward/overall | reward/accuracy | reward/tool | reward/format |\n")
        f.write("|-----|---------------|----------------|------------|---------------|\n")
        f.write(f"| **{args.label}** (baseline) | **{r['overall_mean']:.4f}** | "
                f"**{r['accuracy_mean']:.4f}** | **{r['tool_mean']:.4f}** | **{r['format_mean']:.4f}** |\n")
        f.write("| v19 (GRPO 20-step) | 0.9163 | 0.8478 | 1.0000 | 1.0000 |\n")
        f.write("| v20 (GRPO growthprompt) | 0.9204 | 0.8554 | 1.0000 | 1.0000 |\n\n")
        f.write("> Single-pass inference does not execute the focus tool, so tool_score reflects\n")
        f.write("> whether the model *calls* the tool in its output, not whether it *uses* the result.\n")
        f.write("> GRPO-trained models additionally benefit from the multi-turn observation in the agentic loop.\n\n")
        f.write("## Sample Predictions (first 5)\n\n")
        for rec in records[:5]:
            f.write(f"**Q**: {rec['query']}  \n")
            f.write(f"**GT**: `{rec['ground_truth']}`  \n")
            f.write(f"**Pred**: `{rec['prediction'][:200]}`  \n")
            f.write(f"**Score**: overall={rec['reward_overall']:.3f}, acc={rec['reward_accuracy']:.3f}, "
                    f"tool={rec['reward_tool']:.3f}\n\n")
    print(f"[baseline_eval] report  -> {md_path}")

    # ── console summary ───────────────────────────────────────────────────────
    print("\n=== RESULTS ===")
    print(f"  label            : {args.label}")
    print(f"  n_samples        : {n}")
    print(f"  overall mean     : {r['overall_mean']:.4f}")
    print(f"  accuracy mean    : {r['accuracy_mean']:.4f}")
    print(f"  format mean      : {r['format_mean']:.4f}")
    print(f"  tool mean        : {r['tool_mean']:.4f}")
    print("  (compare: v19 overall=0.9163 acc=0.8478 | v20 overall=0.9204 acc=0.8554)")


if __name__ == "__main__":
    main()
