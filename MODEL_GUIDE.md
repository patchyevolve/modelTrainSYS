# Model Capabilities Guide

## The 4 Training Types and What They Actually Do

---

### 1. Cybersecurity
**What it is:** A wide binary classifier trained to detect network attacks vs benign traffic.

**Trained on:** `cybersecurity_intrusion_data.csv`
- Features: packet size, protocol, login attempts, session duration, encryption, IP reputation, failed logins, browser type, unusual access time
- Label: `attack_detected` (0 = benign, 1 = attack)

**What it can do:**
- Given a row of network session data → outputs probability of attack (0.0–1.0)
- Detects: SQL injection, XSS, DDoS, malware, buffer overflow, privilege escalation, credential stuffing, zero-day patterns
- Uses adversarial training (FGSM) so it's harder to fool with evasion attempts

**What it CANNOT do:**
- Cannot chat or answer questions
- Cannot generate text
- Cannot process images
- Cannot handle data with different columns than it was trained on

**Output:** Single float (sigmoid probability). Threshold 0.5 → attack/benign label.

**Use it for:** Intrusion detection, network monitoring, security log classification.

---

### 2. Hierarchical Mamba
**What it is:** A 3-scale parallel feature extractor. Processes the same input at full width, half width, and quarter width simultaneously, then fuses all three.

**What it can do:**
- Binary or multi-class classification on tabular/CSV data
- Better at capturing both fine-grained and coarse patterns in the same features
- Works on any CSV where you have numeric + categorical columns

**What it CANNOT do:**
- Cannot chat or generate text
- Cannot process raw images (needs pre-extracted features)
- Cannot handle sequences longer than a single row

**Output:** Single logit → sigmoid → probability.

**Use it for:** Any structured tabular classification where features have multi-scale relationships (e.g. sensor data, financial features, log data).

---

### 3. Transformer Only
**What it is:** A residual MLP with skip connections — behaves like a shallow transformer applied to flat feature vectors.

**What it can do:**
- Binary classification on tabular data
- Faster training than Hierarchical Mamba, lower memory
- Good when you have clean, well-normalized features

**What it CANNOT do:**
- Cannot chat or generate text
- Cannot process sequences or images
- No multi-scale processing

**Output:** Single logit → sigmoid → probability.

**Use it for:** Quick baseline classification, smaller datasets, when you want fast training.

---

### 4. Mamba+Transformer
**What it is:** Same architecture as Hierarchical Mamba but with more layers in the fusion stage.

**What it can do:**
- Same as Hierarchical Mamba but higher capacity
- Better for larger datasets (10k+ rows)
- More parameters = needs more epochs to converge

**What it CANNOT do:** Same limitations as Hierarchical Mamba.

**Use it for:** Large tabular datasets where Hierarchical Mamba underfits.

---

## What Can Chat?

**Short answer: none of the trained models can chat.**

The `chat.py` file provides a CLI interface (`python start.py`) with commands like `status`, `train`, `run_inference`, `metrics` etc. The `chat` command in that interface just prints a placeholder message — it does not use any trained model to generate responses.

**What does have real chat/LLM capability:**
- `LLMReflector` in `reflector_trainer.py` — uses Groq `llama-3.3-70b-versatile` to validate and rewrite text outputs. But this is a post-processing corrector, not a chatbot.
- `AutoUpgradeSystem` in `auto_upgrade.py` — uses Groq to ask for architecture improvement suggestions.

**To add chat to a trained model you would need to:**
1. Train on a text dataset (question-answer pairs, dialogue)
2. Use a sequence-to-sequence architecture (the current models output a single number, not text)
3. Or use the Groq API directly (already wired in) as the chat backend

---

## Quick Reference Table

| Model Type        | Input              | Output         | Can Chat | Can Classify | Can Detect Attacks | Needs Training Data |
|-------------------|--------------------|----------------|----------|--------------|--------------------|---------------------|
| Cybersecurity     | Network session CSV| Attack prob    | No       | Yes (binary) | Yes (specialized)  | cybersecurity CSV   |
| Hierarchical Mamba| Any tabular CSV    | Class prob     | No       | Yes          | Yes (generic)      | Any labeled CSV     |
| Transformer Only  | Any tabular CSV    | Class prob     | No       | Yes          | Yes (generic)      | Any labeled CSV     |
| Mamba+Transformer | Any tabular CSV    | Class prob     | No       | Yes          | Yes (generic)      | Any labeled CSV     |
| LLMReflector      | Text string        | Corrected text | Partial  | No           | No                 | None (uses Groq API)|
| AutoUpgrade       | Model metrics      | Upgrade plan   | Partial  | No           | No                 | None (uses Groq API)|

---

## How to Know What a Saved Model Can Do

Check its `.json` metadata file in `trained_models/`:

```json
{
  "model_type": "Cybersecurity",     ← tells you the architecture
  "feature_dim": 16,                 ← how many input features it expects
  "num_classes": 2,                  ← binary (2) or multi-class
  "accuracy": "87.35%",              ← how well it performed
  "config": {
    "hidden_dim": 512,               ← model capacity
    "num_layers": 3
  }
}
```

Or from CLI:
```bash
python start.py --list-models        # see all models with accuracy + status
python start.py --load <name>        # load and print full config + data info
python inference.py --save           # run inference and get full metrics report
```

---

## What Would It Take to Add Real Chat

The system already has Groq wired in. To make a model that can answer questions:

```python
# This already works — uses Groq llama-3.3-70b-versatile
from reflector_trainer import _groq_chat

response = _groq_chat([
    {"role": "system", "content": "You are a cybersecurity assistant."},
    {"role": "user",   "content": "What is a SQL injection attack?"}
])
print(response)
```

The trained `.pt` models are classifiers — they output a number, not text. Chat requires a generative model, which Groq provides via the API already in the codebase.
