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
5. train / val / test = 70 / 15 / 15, split **by prompt** (`GroupShuffleSplit`) so the same prompt never appears in two splits

Model and company names (OpenAI, Claude, Llama, Gemini, …) are replaced with `<org>` before training so the classifier can't just read off self-identification; `--nomask` turns this off for an ablation.

## Setup

```bash
pip install -r requirements.txt
python prepare.py      # builds data/arena.csv
python baseline.py     # majority + tf-idf baselines
```

*(models and results sections coming as they're added)*
