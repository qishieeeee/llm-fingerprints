# llm-fingerprints

Identifying which LLM (GPT, Claude, Llama, etc.) wrote a response using CNN and LSTM classifiers. CMPSC 448 project, Penn State.

**Name:** Qistina Zamzuri

**Report:** [`report/MidtermProjectReport_qjm5062.pdf`](report/MidtermProjectReport_qjm5062.pdf)

## Summary

I trained a TextCNN and a BiLSTM to tell which of five LLM families (Claude, Gemini, GPT, Llama, Mistral) wrote a response from Chatbot Arena.

- **RQ1:** from the response alone, both models reach about 0.61 macro-F1, three times chance (0.20). A TF-IDF + logistic regression baseline does as well, so the signal is mostly word choice and formatting.
- **RQ2:** the prompt alone scores at chance, and adding it to the response does not help. Arena assigns models to prompts at random, so the prompt carries no information about who answered.
- **RQ3 (extra):** a model that never saw a task in training still recognizes the families on it, losing about 0.06 F1 on coding and math and nothing on writing.
- Claude is the easiest family to recognize (F1 ≈ 0.86); Mistral the hardest (F1 ≈ 0.42), often confused with Llama and GPT.
- Masking self-identifying names ("OpenAI", "Claude", …) changes F1 by less than one standard deviation: the models learn style, not name tags.

## Results

Test set: 2,159 examples, 5 balanced classes. Self-identification masked. Mean ± std over 3 seeds; baselines are deterministic.

| Model | Prompt only | Response only | Prompt + response |
|---|---|---|---|
| Majority class | – | 0.065 | – |
| TF-IDF + LR | 0.207 | 0.617 | 0.602 |
| TextCNN | 0.189 ± 0.013 | 0.610 ± 0.003 | 0.603 ± 0.001 |
| BiLSTM | 0.185 ± 0.002 | 0.615 ± 0.007 | 0.595 ± 0.014 |

*Macro-F1. Accuracy and per-seed numbers are in `results/summary.csv` and `results/<model>-<input>/seedN.json`.*

**RQ3, cross-task generalization** (TextCNN, response input, macro-F1 on each task's test rows):

| Held-out task | Test rows | Trained with the task | Trained without it | Drop |
|---|---|---|---|---|
| Coding | 249 | 0.653 ± 0.030 | 0.588 ± 0.028 | 0.064 |
| Math | 169 | 0.597 ± 0.014 | 0.542 ± 0.026 | 0.055 |
| Writing | 265 | 0.578 ± 0.029 | 0.577 ± 0.017 | 0.000 |

Self-identification ablation (TextCNN, response input): masked 0.610 ± 0.003 vs. unmasked 0.615 ± 0.009.

## Data

Source: [`lmarena-ai/arena-human-preference-55k`](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k) (Apache-2.0; Chiang et al., 2024), about 57k Chatbot Arena battles in which two anonymous LLMs answer the same user prompt. The data is not stored in this repo; `prepare.py` downloads and rebuilds it.

`prepare.py` turns the battles into `(family, model, task, split, prompt, response)` rows:

1. each battle gives two examples (model A's and model B's response); only the first turn of each conversation is kept
2. model names are mapped to families by prefix (`gpt-3.5*`/`gpt-4*` → gpt, `claude*` → claude, `llama*` → llama, `mistral*`/`mixtral*` → mistral, `gemini*` → gemini); fine-tunes trained on other models' outputs (Vicuna, Koala, WizardLM, …) are dropped
3. empty responses and exact duplicates are removed
4. classes are balanced by downsampling every family to the size of the smallest one (capped at 3,000)
5. each prompt gets a rough task tag (coding / math / writing / other) by keyword rules
6. train / val / test = 70 / 15 / 15, split by prompt (`GroupShuffleSplit`), so the same prompt never appears in two splits

Model and company names (OpenAI, Claude, Llama, Gemini, …) are replaced with `<org>` before training so the classifier can't read off self-identification; `--nomask` turns this off.

## Models

- TextCNN (Kim, 2014): embedding (200) → Conv1d with widths 3/4/5 × 128 filters → ReLU → masked max-pool → dropout 0.5 → linear
- BiLSTM: embedding (200) → 2-layer bidirectional LSTM (128 per direction, packed sequences) → masked max-pool over time → dropout 0.5 → linear
- Baselines: majority class; TF-IDF (1–2 grams) + logistic regression

Training: Adam (lr 1e-3), cross-entropy, batch 64, up to 15 epochs, early stopping on validation macro-F1 (patience 3), gradient clipping 1.0, 3 seeds. Tokens: words and punctuation with case kept and line breaks as a `<nl>` token (formatting is part of the fingerprint); vocabulary of the 30k most frequent training tokens; sequences cut to 256 tokens (in `both` mode: 64 prompt tokens + `<sep>` + response).

## Running

Easiest: open `run.ipynb` in Google Colab with a GPU runtime (T4) and run all cells.

Locally or in Colab:

```bash
pip install -r requirements.txt
python prepare.py                                       # builds data/arena.csv
python baseline.py                                      # majority + tf-idf baselines
python train.py --model cnn --input response            # rq1 (also --model lstm)
python train.py --model lstm --input prompt             # rq2 (inputs: prompt, response, both)
python train.py --model cnn --input response --nomask   # self-identification ablation
python rq3.py                                           # rq3, cross-task generalization
```

`prepare.py` uses a fixed seed (42), so every run rebuilds the same dataset and split.

## Repository structure

```
prepare.py        dataset construction
baseline.py       majority + tf-idf/logistic regression baselines
train.py          cnn / lstm training and evaluation (rq1, rq2)
rq3.py            cross-task generalization experiment (rq3)
src/data.py       tokenization, vocabulary, self-id masking, dataset
src/models.py     TextCNN and BiLSTM
run.ipynb         colab runner
results/          summary.csv, tf-idf top features, one folder per run
                  (per-seed metrics, confusion matrices, training curves)
results/rq3/      rq3 per-seed results and summary
report/           project report (pdf)
```

## References

- Chiang et al. (2024). *Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference.* [arXiv:2403.04132](https://arxiv.org/abs/2403.04132)
- Kim (2014). *Convolutional Neural Networks for Sentence Classification.* [arXiv:1408.5882](https://arxiv.org/abs/1408.5882)
- Hochreiter & Schmidhuber (1997). *Long Short-Term Memory.* Neural Computation 9(8).
- Sun et al. (2025). *Idiosyncrasies in Large Language Models.* [arXiv:2502.12150](https://arxiv.org/abs/2502.12150)
