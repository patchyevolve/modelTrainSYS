"""
Advanced Tokenizer — BPE subword tokenizer with reasoning markers.
Wraps HuggingFace tokenizers when available; pure-Python fallback.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

log = logging.getLogger("AdvancedTokenizer")

HF_AVAILABLE = False
try:
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
    from tokenizers.normalizers import NFKC
    from tokenizers.processors import TemplateProcessing
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    HF_AVAILABLE = True
except ImportError:
    HFTokenizer = None


class AdvancedTokenizer:
    PAD = "<PAD>"
    UNK = "<UNK>"
    BOS = "<BOS>"
    EOS = "<EOS>"

    SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]

    def __init__(self, vocab_size: int = 8192):
        self.vocab_size = vocab_size
        self._hf: Optional['HFTokenizer'] = None
        self._vocab: Dict[str, int] = {}
        self._id2tok: Dict[int, str] = {}
        self._use_hf = HF_AVAILABLE

    def build(self, texts: List[str]) -> None:
        if self._use_hf and HFTokenizer is not None:
            self._build_hf(texts)
        else:
            self._build_fallback(texts)

    def _build_hf(self, texts: List[str]) -> None:
        tokenizer = HFTokenizer(BPE(unk_token=self.UNK))
        tokenizer.normalizer = NFKC()
        tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        tokenizer.post_processor = TemplateProcessing(
            single=f"{self.BOS} $A {self.EOS}",
            pair=f"{self.BOS} $A {self.EOS} $B:1 {self.EOS}:1",
            special_tokens=[
                (self.PAD, 0), (self.UNK, 1), (self.BOS, 2), (self.EOS, 3),
            ],
        )
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.SPECIAL_TOKENS,
            show_progress=True,
            initial_alphabet=ByteLevelPreTokenizer.alphabet(),
        )
        tokenizer.train_from_iterator(texts, trainer=trainer)
        self._hf = tokenizer
        self._vocab = tokenizer.get_vocab()
        self._id2tok = {v: k for k, v in self._vocab.items()}
        log.info(f"HF BPE tokenizer: vocab={tokenizer.get_vocab_size()}")

    def _build_fallback(self, texts: List[str]) -> None:
        from collections import Counter
        self._vocab = {t: i for i, t in enumerate(self.SPECIAL_TOKENS)}
        next_id = len(self.SPECIAL_TOKENS)

        word_freq = Counter()
        for text in texts:
            for word in text.strip().split():
                word_freq[word] += 1

        sorted_words = sorted(word_freq, key=lambda w: -word_freq[w])
        for w in sorted_words:
            if next_id >= self.vocab_size:
                break
            if w not in self._vocab:
                self._vocab[w] = next_id
                next_id += 1

        self._id2tok = {v: k for k, v in self._vocab.items()}
        actual_vs = len(self._vocab)
        log.info(f"Fallback word tokenizer: vocab={actual_vs}")

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        if self._hf is not None:
            encoded = self._hf.encode(text)
            ids = encoded.ids
            if not add_special:
                bos = self._vocab.get(self.BOS, 2)
                eos = self._vocab.get(self.EOS, 3)
                if ids and ids[0] == bos:
                    ids = ids[1:]
                if ids and ids[-1] == eos:
                    ids = ids[:-1]
            return ids
        tokens = []
        if add_special:
            tokens.append(self._vocab.get(self.BOS, 2))
        for word in text.strip().split():
            tid = self._vocab.get(word, self._vocab.get(self.UNK, 1))
            tokens.append(tid)
        if add_special:
            tokens.append(self._vocab.get(self.EOS, 3))
        return tokens

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        if self._hf is not None:
            return self._hf.decode(ids, skip_special_tokens=skip_special)
        special_ids = {self._vocab.get(t) for t in self.SPECIAL_TOKENS if t in self._vocab}
        tokens = []
        for i in ids:
            if skip_special and i in special_ids:
                continue
            tokens.append(self._id2tok.get(i, ""))
        return " ".join(tokens)

    @property
    def vocab_size_real(self) -> int:
        if self._hf is not None:
            return self._hf.get_vocab_size()
        return len(self._vocab)

    def save(self, path: str) -> None:
        data: Dict = {"vocab_size": self.vocab_size, "use_hf": self._hf is not None}
        if self._hf is not None:
            self._hf.save(str(Path(path).with_suffix(".json")))
            data["hf_path"] = str(Path(path).with_suffix(".json"))
        else:
            data["vocab"] = self._vocab
            data["id2tok"] = {str(k): v for k, v in self._id2tok.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "AdvancedTokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        t = cls(vocab_size=data.get("vocab_size", 8192))
        if data.get("use_hf") and HF_AVAILABLE:
            hf_path = data.get("hf_path", "")
            if hf_path and Path(hf_path).exists():
                t._hf = HFTokenizer.from_file(hf_path)
            else:
                local_hf = Path(path).with_suffix(".json")
                if local_hf.exists():
                    t._hf = HFTokenizer.from_file(str(local_hf))
            if t._hf is not None:
                t._vocab = t._hf.get_vocab()
                t._id2tok = {v: k for k, v in t._vocab.items()}
                t._use_hf = True
                return t
        t._vocab = data.get("vocab", {})
        t._id2tok = {int(k): v for k, v in data.get("id2tok", {}).items()}
        t._use_hf = False
        return t


def train_tokenizer(texts: List[str], vocab_size: int = 8192, save_path: Optional[str] = None) -> AdvancedTokenizer:
    tok = AdvancedTokenizer(vocab_size=vocab_size)
    tok.build(texts)
    if save_path:
        tok.save(save_path)
    return tok
