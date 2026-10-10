# llm-fingerprints

Identifying which LLM (GPT, Claude, Llama, etc.) wrote a response using CNN and LSTM classifiers. CMPSC 448 project, Penn State.

## Research questions

- **RQ1:** can a CNN / LSTM identify which LLM family generated a response?
- **RQ2:** does the user's prompt help? (input only vs. output only vs. input + output)

## Data

Source: [`lmarena-ai/arena-human-preference-55k`](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k) (Apache-2.0), ~57k Chatbot Arena battles where two anonymous LLMs answer the same user prompt.

`prepare.py` turns this into `(family, model, prompt, response)` rows:

1. each battle gives two examples (model A's and model B's response); only the **first turn** of each conversation is kept
2. model names are mapped to families by prefix (`gpt-3.5*/gpt-4*` → gpt, `claude*` → claude, `llama*` → llama, `mistral*/mixtral*` → mistral, `gemini*` → gemini); fine-tunes trained on other models' outputs (vicuna, koala, wizardlm, …) are dropped
3. empty responses and exact duplicates are removed
4. classes are **balanced** by downsampling every family to the same size (default ≤ 3000)
5. each prompt gets a rough task tag (coding / math / writing / other) by keyword rules
6. train / val / test = 70 / 15 / 15, split **by prompt** (`GroupShuffleSplit`) so the same prompt never appears in two splits

Model and company names (OpenAI, Claude, Llama, Gemini, …) are replaced with `<org>` before training so the classifier can't just read off self-identification; `--nomask` turns this off for an ablation.

## Models

- **TextCNN** (Kim, 2014): embedding (200) → Conv1d with widths 3/4/5 × 128 filters → ReLU → masked max-pool → dropout 0.5 → linear
- **BiLSTM**: embedding (200) → 2-layer bidirectional LSTM (128 per direction, packed sequences) → masked max-pool over time → dropout 0.5 → linear
- **Baselines**: majority class; TF-IDF (1–2 grams) + logistic regression

Training: Adam (lr 1e-3), cross-entropy, batch 64, up to 15 epochs, early stopping on validation macro-F1 (patience 3), gradient clipping 1.0, 3 seeds. Tokens: words + punctuation with case kept and line breaks as a `<nl>` token (formatting is part of the fingerprint); vocabulary of the 30k most frequent training tokens; sequences truncated to 256 tokens (in `both` mode: 64 prompt tokens + `<sep>` + response).

## Running

Easiest: open `run.ipynb` in Google Colab with a GPU runtime and run all cells.

Locally:

```bash
pip install -r requirements.txt
python prepare.py                                # builds data/arena.csv
python baseline.py                               # majority + tf-idf baselines
python train.py --model cnn --input response     # rq1
python train.py --model lstm --input prompt      # rq2 (also: response, both)
python train.py --model cnn --input response --nomask   # self-identification ablation
```

Results land in `results/`: one folder per run with per-seed metrics (`seedN.json`), confusion matrices and training curves, plus `results/summary.csv` with mean ± std across seeds.

## Repository structure

```
prepare.py        dataset construction
baseline.py       majority + tf-idf/logistic regression baselines
train.py          cnn / lstm training and evaluation
src/data.py       tokenization, vocabulary, masking, dataset
src/models.py     TextCNN and BiLSTM
run.ipynb         colab runner
report/           project report (pdf)
```
