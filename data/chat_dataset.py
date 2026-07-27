import json
import logging
import random
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable

log = logging.getLogger("ChatDataset")

def _torch():
    import torch
    return torch


CHAT_TEMPLATES = {
    "llama3": {
        "bos": "<|begin_of_text|>",
        "system": "<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
        "user": "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>",
    },
    "simple": {
        "bos": "",
        "user": "<|user|>\n{content}\n<|eot|>\n",
        "assistant": "<|assistant|>\n{content}\n<|eot|>\n",
        "system": "<|system|>\n{content}\n<|eot|>\n",
    },
    "raw": {
        "bos": "",
        "user": "User: {content}\n",
        "assistant": "Assistant: {content}\n",
        "system": "System: {content}\n",
    },
}


@dataclass
class ChatExample:
    messages: List[Dict[str, str]]
    loss_mask: List[bool] = field(default_factory=list)


class ChatDataset:
    """Torch Dataset for chat conversation training. Requires torch at runtime."""
    def __init__(self, examples: List[ChatExample], tokenizer, seq_len: int = 2048,
                 template_name: str = "llama3"):
        self.examples = examples
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.template = CHAT_TEMPLATES.get(template_name, CHAT_TEMPLATES["simple"])
        self._encoded = []
        self._encode_all()

    def _encode_all(self):
        torch = _torch()
        for ex in self.examples:
            tokens, mask = self._encode_conversation(ex)
            for i in range(0, len(tokens), self.seq_len // 2):
                chunk = tokens[i:i + self.seq_len]
                chunk_mask = mask[i:i + self.seq_len]
                if len(chunk) < 2:
                    continue
                x = torch.tensor(chunk[:self.seq_len], dtype=torch.long)
                y = torch.tensor(
                    [chunk[t + 1] if chunk_mask[t + 1] else -100 for t in range(len(chunk) - 1)] + [-100],
                    dtype=torch.long)[:self.seq_len]
                pad_len = self.seq_len - len(x)
                if pad_len > 0:
                    x = torch.cat([x, torch.zeros(pad_len, dtype=torch.long)])
                    y = torch.cat([y, torch.full((pad_len,), -100, dtype=torch.long)])
                self._encoded.append((x, y))

    def _encode_conversation(self, ex: ChatExample) -> Tuple[List[int], List[bool]]:
        tokens: List[int] = []
        mask: List[bool] = []
        bos = self.tokenizer.encode(self.template["bos"])
        tokens.extend(bos)
        mask.extend([False] * len(bos))

        for msg in ex.messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            template_key = role if role in self.template else "user"
            text = self.template[template_key].format(content=content)
            ids = self.tokenizer.encode(text)
            tokens.extend(ids)
            is_assistant = (role == "assistant")
            mask.extend([is_assistant] * len(ids))

        return tokens, mask

    def __len__(self):
        return len(self._encoded)

    def __getitem__(self, idx):
        return self._encoded[idx]


class ReasoningDataset:
    """Torch Dataset for GSM8K/MATH reasoning. Requires torch at runtime."""
    def __init__(self, examples: List[Dict], tokenizer, seq_len: int = 2048,
                 template_name: str = "llama3"):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.template = CHAT_TEMPLATES.get(template_name, CHAT_TEMPLATES["simple"])
        self._encoded: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._parse(examples)
        self._encode_all()

    def _parse(self, examples: List[Dict]):
        self.parsed: List[ChatExample] = []
        for ex in examples:
            question = ex.get("question", "")
            answer = ex.get("answer", "")
            if not question or not answer:
                continue

            answer_clean = re.sub(r'####\s*(\d+(?:\.\d+)?)', r'The answer is \1.', answer).strip()
            cot_text = f"Let me solve this step by step.\n\n{answer_clean}"

            self.parsed.append(ChatExample(messages=[
                {"role": "user", "content": question},
                {"role": "assistant", "content": cot_text},
            ]))

    def _encode_all(self):
        ds = ChatDataset(self.parsed, self.tokenizer, self.seq_len, self.template)
        self._encoded = ds._encoded

    def __len__(self):
        return len(self._encoded)

    def __getitem__(self, idx):
        return self._encoded[idx]


def parse_openai_messages(path: Path) -> List[ChatExample]:
    examples = []
    data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    if not isinstance(data, list):
        data = [data]
    for item in data:
        messages = item.get("messages", item.get("conversations", []))
        if not messages:
            continue
        parsed = []
        for m in messages:
            if isinstance(m, dict):
                role = m.get("role", m.get("from", "user"))
                content = m.get("content", m.get("value", ""))
                if role == "gpt":
                    role = "assistant"
                elif role == "human":
                    role = "user"
                parsed.append({"role": role, "content": content})
        if parsed:
            examples.append(ChatExample(messages=parsed))
    return examples


def parse_gsm8k(path: Path) -> List[Dict]:
    examples = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
            examples.append(item)
        except json.JSONDecodeError:
            pass
    return examples


def parse_math_dataset(path: Path) -> List[Dict]:
    examples = []
    data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    if isinstance(data, list):
        for item in data:
            problem = item.get("problem", item.get("question", ""))
            solution = item.get("solution", item.get("answer", ""))
            if problem and solution:
                examples.append({"question": problem, "answer": solution})
    return examples


def parse_sharegpt(path: Path) -> List[ChatExample]:
    examples = []
    data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    if not isinstance(data, list):
        data = [data]
    for item in data:
        convs = item.get("conversations", [])
        if not convs:
            continue
        parsed = []
        for c in convs:
            role = c.get("from", c.get("role", "user"))
            content = c.get("value", c.get("content", ""))
            if role == "gpt":
                role = "assistant"
            elif role in ("human", "user"):
                role = "user"
            parsed.append({"role": role, "content": content})
        if parsed:
            examples.append(ChatExample(messages=parsed))
    return examples


def build_chat_loaders(
    file_paths: List[str],
    tokenizer=None,
    seq_len: int = 2048,
    batch_size: int = 4,
    val_split: float = 0.05,
    template_name: str = "llama3",
    format_hint: Optional[str] = None,
):
    from data.advanced_tokenizer import AdvancedTokenizer
    from torch.utils.data import DataLoader, random_split
    torch = _torch()

    all_examples: List[ChatExample] = []
    total_reasoning = 0

    for fp in file_paths:
        p = Path(fp)
        if not p.exists():
            continue
        name = format_hint or p.name.lower()

        if "gsm8k" in name or "gsm" in name:
            raw = parse_gsm8k(p)
            ds = ReasoningDataset(raw, tokenizer, seq_len, template_name)
            total_reasoning += len(raw)
        elif "math" in name:
            raw = parse_math_dataset(p)
            ds = ReasoningDataset(raw, tokenizer, seq_len, template_name)
            total_reasoning += len(raw)
        elif "sharegpt" in name or "share_gpt" in name:
            all_examples.extend(parse_sharegpt(p))
        else:
            all_examples.extend(parse_openai_messages(p))

    if tokenizer is None:
        corpus_texts = []
        for ex in all_examples:
            for m in ex.messages:
                corpus_texts.append(m.get("content", ""))
        tokenizer = AdvancedTokenizer(vocab_size=8192)
        tokenizer.build(corpus_texts)

    dataset = ChatDataset(all_examples, tokenizer, seq_len, template_name)
    n_val = max(1, int(len(dataset) * val_split))
    n_train = max(1, len(dataset) - n_val)

    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    info = {
        "total_examples": len(all_examples),
        "reasoning_examples": total_reasoning,
        "train_windows": n_train,
        "val_windows": n_val,
        "seq_len": seq_len,
        "vocab_size": tokenizer.vocab_size_real,
        "template": template_name,
    }

    return train_loader, val_loader, tokenizer, info
