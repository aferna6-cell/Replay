#!/usr/bin/env bash
# Build the LoRA SFT dataset from the reviewer's graded corpus, then print
# instructions for training it — never here. Per specs/llm-turn-director_spec.md
# (non-goals): "Training never runs on the ThinkPad: rented GPU or a fine-tune
# API." This script only ever does the local, cheap, stdlib-only half (dataset
# build). The training recipe is print-only unless you pass --run-remote, and
# even then this script does not launch anything for you — it hands you a
# ready-to-copy config + commands to run wherever your GPU actually is.
set -euo pipefail
cd "$(dirname "$0")/.."

CORPUS="data/train_corpus.jsonl"
OUT="ml/sft_dataset.jsonl"
MIN_VERDICT="good"
RUN_REMOTE=0

usage() {
  cat <<'EOF'
Usage: scripts/retrain_lora.sh [--run-remote] [--corpus PATH] [--out PATH]
                               [--min-verdict good|questionable]

Always: builds ml/sft_dataset.jsonl from the reviewer's graded corpus
(data/train_corpus.jsonl) via `python -m ml.lora_dataset`. Safe to run with
an empty or missing corpus — you just get an empty dataset.

--run-remote   Also print the LoRA training recipe (rented-GPU or
               fine-tune-API config + commands). Printing only — this
               script never trains anything itself.
--corpus PATH  Graded-decision JSONL to read (default: data/train_corpus.jsonl)
--out PATH     SFT dataset JSONL to write (default: ml/sft_dataset.jsonl)
--min-verdict  Lowest verdict included directly (default: good); "bad"
               decisions only enter via hindsight rewrite regardless.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --run-remote) RUN_REMOTE=1; shift ;;
    --corpus) CORPUS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --min-verdict) MIN_VERDICT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

PY="${PYTHON:-python3}"

echo "Building LoRA SFT dataset: ${CORPUS} -> ${OUT} (min-verdict=${MIN_VERDICT})"
"$PY" -m ml.lora_dataset --corpus "$CORPUS" --out "$OUT" --min-verdict "$MIN_VERDICT"
echo "Dataset build complete."

if [[ "$RUN_REMOTE" -ne 1 ]]; then
  echo
  echo "This machine never trains (specs/llm-turn-director_spec.md, non-goals)."
  echo "Dataset is ready at ${OUT}. Re-run with --run-remote to print the"
  echo "rented-GPU / fine-tune-API training recipe."
  exit 0
fi

# Same model family the live Director talks to (hsbg_coach/llm_client.py
# DEFAULT_MODEL / HSBG_LLM_MODEL) — the LoRA adapter has to match what's
# actually served, local (Ollama) or hosted.
MODEL_TAG="${HSBG_LLM_MODEL:-qwen2.5:3b-instruct}"

echo
echo "=================================================================="
echo " LoRA training recipe — target model family: ${MODEL_TAG}"
echo " Run this on a rented GPU (Runpod/Lambda/Vast) or a fine-tune API,"
echo " never on the ThinkPad. Nothing below is executed by this script."
echo "=================================================================="
cat <<EOF

1) Copy the dataset to wherever training runs:
   scp ${OUT} <gpu-host>:~/hsbg-lora/sft_dataset.jsonl
   # or upload it to your fine-tune API's file endpoint.

2) Install a LoRA trainer. Either of these works for a small Qwen2.5-class
   instruct model:
     pip install unsloth              # CALIBRATE: exact extras/version pin
                                       # for your CUDA version — see
                                       # https://github.com/unslothai/unsloth
   or
     pip install axolotl               # CALIBRATE: same — axolotl's install
                                       # instructions are CUDA/torch-version
                                       # specific and change often.

3) Base model — map the Ollama tag to a Hugging Face repo id.
   # CALIBRATE: confirm this mapping for whatever MODEL_TAG resolves to;
   # "qwen2.5:3b-instruct" (Ollama) is believed to correspond to
   # Qwen/Qwen2.5-3B-Instruct on Hugging Face, but this is NOT verified
   # against a live pull — check \`ollama show ${MODEL_TAG} --modelfile\`
   # for the exact base if you're unsure.
   BASE_MODEL_HF="Qwen/Qwen2.5-3B-Instruct"   # CALIBRATE

4) Axolotl-style LoRA config (adjust paths, then hand this file to your
   trainer of choice — the exact invocation differs between unsloth's
   Python API and axolotl's CLI, both marked CALIBRATE below):

   cat > hsbg_director_lora.yml <<'CFG'
   base_model: Qwen/Qwen2.5-3B-Instruct   # CALIBRATE: see step 3
   model_type: AutoModelForCausalLM
   tokenizer_type: AutoTokenizer

   load_in_4bit: true
   adapter: lora
   lora_r: 16
   lora_alpha: 32
   lora_dropout: 0.05
   lora_target_linear: true

   datasets:
     - path: sft_dataset.jsonl
       type: chat_template          # CALIBRATE: exact dataset "type" key
                                     # for a raw {"messages": [...]} JSONL
                                     # depends on your axolotl/unsloth
                                     # version — check current docs.
       field_messages: messages

   sequence_len: 2048
   sample_packing: false

   num_epochs: 3                    # CALIBRATE: starting point, not tuned
   micro_batch_size: 2
   gradient_accumulation_steps: 4
   learning_rate: 0.0002
   lr_scheduler: cosine
   warmup_steps: 10

   output_dir: ./hsbg_director_lora_out
   CFG

5) Launch training (# CALIBRATE — exact CLI differs by trainer/version):
   axolotl train hsbg_director_lora.yml
   # or, with unsloth's Python API, load BASE_MODEL_HF, apply the same
   # LoRA hyperparameters above, and call trainer.train() directly — see
   # https://github.com/unslothai/unsloth for the current example script.

6) Export + serve the adapter with the SAME backend the live Director
   uses (hsbg_coach/llm_client.py): merge into a GGUF for Ollama, or serve
   the LoRA-merged weights behind an OpenAI-compatible endpoint and point
   HSBG_LLM_URL / HSBG_LLM_MODEL at it. # CALIBRATE: exact merge/export
   command depends on the trainer (axolotl ships \`scripts/merge_lora.py\`
   equivalents; check current docs for the flag names).

EOF
echo "=================================================================="
