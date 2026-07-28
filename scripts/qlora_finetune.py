"""
QLoRA Fine-Tune — fine-tune billion-parameter models (LLaMA, DeepSeek-Coder) on code Q&A.
Runs on a single T4 (4-bit quantization + LoRA).

Usage:
    python scripts/qlora_finetune.py  # fine-tune LLaMA 3 8B on generated code QA
    python scripts/qlora_finetune.py --model deepseek-coder --epochs 5
    python scripts/qlora_finetune.py --infer-only  # just test the base model
"""

import argparse, json, sys, os, random, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

MODELS = {
    "llama3": "meta-llama/Meta-Llama-3-8B",
    "llama3.1": "meta-llama/Meta-Llama-3.1-8B",
    "deepseek-coder": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
    "codellama": "codellama/CodeLlama-7b-Instruct-hf",
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
}

CODE_TASKS = [
    # Python
    ("Write a Python function that reverses a linked list",
     "def reverse_linked_list(head):\n    prev = None\n    curr = head\n    while curr:\n        nxt = curr.next\n        curr.next = prev\n        prev = curr\n        curr = nxt\n    return prev"),
    ("Write a Python function that finds all prime numbers up to n",
     "def primes_up_to(n):\n    sieve = [True] * (n + 1)\n    sieve[0] = sieve[1] = False\n    for i in range(2, int(n**0.5) + 1):\n        if sieve[i]:\n            for j in range(i*i, n+1, i):\n                sieve[j] = False\n    return [i for i, p in enumerate(sieve) if p]"),
    ("Write a Python function that merges two sorted lists",
     "def merge_sorted(a, b):\n    i = j = 0; res = []\n    while i < len(a) and j < len(b):\n        if a[i] < b[j]: res.append(a[i]); i += 1\n        else: res.append(b[j]); j += 1\n    return res + a[i:] + b[j:]"),
    ("Write a Python function that implements binary search",
     "def binary_search(arr, target):\n    l, r = 0, len(arr) - 1\n    while l <= r:\n        m = (l + r) // 2\n        if arr[m] == target: return m\n        elif arr[m] < target: l = m + 1\n        else: r = m - 1\n    return -1"),
    ("Write a Python function that checks if two strings are anagrams",
     "def is_anagram(s1, s2): return sorted(s1) == sorted(s2)"),
    ("Write a Python function that finds the longest word in a sentence",
     "def longest_word(sentence): return max(sentence.split(), key=len)"),
    # C
    ("Write a C function that swaps two integers using pointers",
     "void swap(int *a, int *b) { int t = *a; *a = *b; *b = t; }"),
    ("Write a C function that returns the length of a string",
     "int strlen_custom(char *s) { int i = 0; while (s[i]) i++; return i; }"),
    # Bash
    ("Write a bash command to find all .py files modified in the last 7 days",
     "find . -name '*.py' -mtime -7"),
    ("Write a bash one-liner to count lines of code in a project",
     "find . -type f \\( -name '*.py' -o -name '*.c' -o -name '*.h' \\) -exec cat {} + | wc -l"),
    # Git
    ("Write a git command to undo the last commit keeping changes staged",
     "git reset --soft HEAD~1"),
    ("Write a git command to rebase onto main while preserving commits",
     "git rebase main"),
]

REASONING_TASKS = [
    "What is the capital of France?", "What is 15 + 27?",
    "Explain what a linked list is",
    "What is the difference between a list and a tuple in Python?",
    "Explain the difference between git merge and git rebase",
    "What is a race condition in concurrent programming?",
    "Explain the Factory design pattern",
    "What is an SQL injection and how to prevent it?",
    "What is the difference between TCP and UDP?",
    "Explain what a REST API is",
]


def generate_code_data(count: int = 500) -> list:
    rng = random.Random(0)
    data = []
    for i in range(count):
        idx = i % len(CODE_TASKS)
        q, a = CODE_TASKS[idx]
        if i >= len(CODE_TASKS):
            rng.shuffle(CODE_TASKS)
        data.append({"instruction": q, "output": a})
    return data


def format_chat(instruction: str, output: str = "") -> str:
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}"""


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tune on code")
    parser.add_argument("--model", default="llama3", choices=list(MODELS.keys()),
                        help="Base model to fine-tune")
    parser.add_argument("--epochs", type=int, default=3, help="Fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="LoRA learning rate")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--data-count", type=int, default=500, help="Training examples")
    parser.add_argument("--infer-only", action="store_true", help="Skip training, just test")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--save-dir", default="lora_adapters", help="Output dir for adapters")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else args.device
    print(f"Device: {device.upper()}")

    # Check for required libs
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
        import bitsandbytes as bnb
    except ImportError:
        print("Missing dependencies. Run: pip install transformers accelerate peft bitsandbytes")
        sys.exit(1)

    model_name = MODELS[args.model]
    save_dir = Path(args.save_dir) / args.model
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Load 4-bit quantized model ────────────────────────────────────────
    print(f"\nLoading {model_name} in 4-bit...")
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    print(f"  {sum(p.numel() for p in model.parameters()):,} total params")

    if args.infer_only:
        # Just test the base model
        test = random.choice(REASONING_TASKS)
        prompt = format_chat(test)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=150, temperature=0.3)
        print(f"\n  Base model test:\n  Q: {test}\n  A: {tokenizer.decode(out[0], skip_special_tokens=True)}")
        return

    # ── Add LoRA ──────────────────────────────────────────────────────────
    print(f"\nAdding LoRA (rank={args.rank})...")
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Prepare data ──────────────────────────────────────────────────────
    print(f"\nGenerating {args.data_count} code Q&A examples...")
    data = generate_code_data(args.data_count)
    texts = [format_chat(d["instruction"], d["output"]) for d in data]

    # Tokenize with labels
    enc = tokenizer(texts, truncation=True, padding="max_length",
                    max_length=512, return_tensors="pt")
    enc["labels"] = enc["input_ids"].clone()

    dataset = torch.utils.data.TensorDataset(
        enc["input_ids"], enc["attention_mask"], enc["labels"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\nTraining ({args.epochs} epochs)...")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for bx, mask, by in loader:
            bx, mask, by = bx.to(device), mask.to(device), by.to(device)
            opt.zero_grad()
            out = model(input_ids=bx, attention_mask=mask, labels=by)
            out.loss.backward()
            opt.step()
            total_loss += out.loss.item()
        avg = total_loss / max(len(loader), 1)
        print(f"  Epoch {epoch}/{args.epochs} loss={avg:.4f} ({time.time()-t0:.0f}s)")

    # ── Save adapters ─────────────────────────────────────────────────────
    adapter_path = save_dir / "adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\nAdapters saved to {adapter_path}")

    # ── Test ──────────────────────────────────────────────────────────────
    print(f"\nTesting fine-tuned model...")
    model.eval()
    test_q = random.choice(REASONING_TASKS)
    prompt = format_chat(test_q)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=150, temperature=0.3)
    answer = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"  Q: {test_q}\n  A: {answer.split('assistant')[-1].strip() if 'assistant' in answer else answer}")

    # Code test
    code_q, code_a = CODE_TASKS[0]
    prompt = format_chat(code_q)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, temperature=0.2)
    answer = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"\n  Q: {code_q}\n  A: {answer.split('assistant')[-1].strip() if 'assistant' in answer else answer}")

    total_time = time.time() - t0
    print(f"\nDone in {total_time:.0f}s. Adapters: {adapter_path}")
    print(f"Load with: PeftModel.from_pretrained(base_model, '{adapter_path}')")


if __name__ == "__main__":
    main()
