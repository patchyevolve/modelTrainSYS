"""Evaluation harness: perplexity, reasoning accuracy, generation quality."""

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.implementations import HMTLanguageModel, GenConfig
from data.advanced_tokenizer import AdvancedTokenizer


def compute_perplexity(
    model: HMTLanguageModel,
    tokenizer: AdvancedTokenizer,
    texts: List[str],
    batch_size: int = 4,
    max_length: Optional[int] = None,
    device: str = "cpu",
) -> Dict:
    model.eval()
    model.to(device)
    if max_length is None:
        max_length = model.max_seq

    total_loss = 0.0
    total_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_ids = []
        for t in batch_texts:
            ids = tokenizer.encode(t)
            if len(ids) < 2:
                continue
            ids = ids[:max_length + 1]
            batch_ids.append(ids)

        if not batch_ids:
            continue

        max_len = max(len(ids) for ids in batch_ids)
        padded = torch.zeros(len(batch_ids), max_len, dtype=torch.long)
        for j, ids in enumerate(batch_ids):
            padded[j, :len(ids)] = torch.tensor(ids)

        x = padded[:, :-1].to(device)
        y = padded[:, 1:].to(device)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device != "cpu")):
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1),
                ignore_index=0, reduction="sum")
            total_loss += loss.item()
            total_tokens += (y != 0).sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 100))

    return {"perplexity": ppl, "avg_loss": avg_loss, "tokens": total_tokens}


def compute_reasoning_accuracy(
    model: HMTLanguageModel,
    tokenizer: AdvancedTokenizer,
    questions: List[str],
    answers: List[str],
    gen_cfg: Optional[GenConfig] = None,
    max_new: int = 256,
    device: str = "cpu",
    extract_answer_fn: Optional[Callable] = None,
) -> Dict:
    if gen_cfg is None:
        gen_cfg = GenConfig(temperature=0.2, top_k=1, max_new_tokens=max_new)

    model.eval()
    model.to(device)

    def default_extract(text: str) -> str:
        text = text.strip()
        if "####" in text:
            text = text.split("####")[-1].strip()
        for pattern in [r"The answer is (\d+(?:\.\d+)?)", r"answer is (\d+(?:\.\d+)?)",
                        r"= (\d+(?:\.\d+)?)", r"(\d+(?:\.\d+))$"]:
            m = re.search(pattern, text)
            if m:
                return m.group(1)
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        return numbers[-1] if numbers else ""

    import re
    extract = extract_answer_fn or default_extract

    correct = 0
    incorrect = 0
    errors = 0
    results = []

    for q, a in zip(questions, answers):
        try:
            prompt_ids = tokenizer.encode(q)
            output_ids = model.generate(prompt_ids, cfg=gen_cfg, device=device)
            output_text = tokenizer.decode(output_ids, skip_special=True)
            pred = extract(output_text)
            expected = extract(a) if extract != default_extract else a.strip()
            is_correct = pred == expected or pred in expected
            if is_correct:
                correct += 1
            else:
                incorrect += 1
            results.append({
                "question": q[:100],
                "expected": expected,
                "predicted": pred,
                "correct": is_correct,
                "output": output_text[:200],
            })
        except Exception as e:
            errors += 1
            results.append({"question": q[:100], "error": str(e)})

    total = correct + incorrect
    return {
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "incorrect": incorrect,
        "errors": errors,
        "total": total,
        "results": results,
    }


def evaluate_model(
    model_path: str,
    eval_texts: Optional[List[str]] = None,
    eval_questions: Optional[List[str]] = None,
    eval_answers: Optional[List[str]] = None,
    batch_size: int = 4,
    device: str = "cpu",
) -> Dict:
    from core.text_model import load_lm
    model, tokenizer = load_lm(model_path, device=device)

    results = {"model": Path(model_path).name, "params": model.count_parameters()}

    if eval_texts:
        ppl_result = compute_perplexity(model, tokenizer, eval_texts, batch_size=batch_size, device=device)
        results["perplexity"] = ppl_result

    if eval_questions and eval_answers:
        acc_result = compute_reasoning_accuracy(model, tokenizer, eval_questions, eval_answers, device=device)
        results["reasoning"] = acc_result

    return results


def load_gsm8k(path: str, split: str = "test") -> Tuple[List[str], List[str]]:
    questions, answers = [], []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            questions.append(item["question"])
            answers.append(item["answer"])
    return questions[:100], answers[:100]


def save_eval_report(results: Dict, path: str):
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Eval report saved to {path}")
