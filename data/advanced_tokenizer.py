"""
Advanced Tokenizer with Reasoning-Aware Tokenization
==================================================
Features:
- BPE-style subword tokenization for better quality
- Special tokens for reasoning markers
- Logical operator tokens
- Sentence boundary detection
"""

import json
import re
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter, defaultdict
import logging

log = logging.getLogger("AdvancedTokenizer")


class AdvancedTokenizer:
    """
    Hybrid tokenizer combining:
    - BPE subword tokenization for quality
    - Special reasoning tokens
    - Logical operator tokens
    """
    
    # Special tokens
    PAD   = "<PAD>"
    UNK   = "<UNK>"
    BOS   = "<BOS>"
    EOS   = "<EOS>"
    MASK  = "<MASK>"
    
    # Reasoning tokens
    REASON_START = "<REASON>"
    REASON_END   = "</REASON>"
    THOUGHT      = "<THOUGHT>"
    STEP         = "<STEP>"
    CONCLUSION   = "<CONCLUSION>"
    
    # Logical tokens
    THEREFORE  = "<THEREFORE>"
    BECAUSE    = "<BECAUSE>"
    IF         = "<IF>"
    THEN       = "<THEN>"
    AND        = "<AND>"
    OR         = "<OR>"
    NOT        = "<NOT>"
    SINCE      = "<SINCE>"
    HENCE      = "<HENCE>"
    
    # Question tokens
    QUESTION = "<QUESTION>"
    ANSWER  = "<ANSWER>"
    
    # Meta tokens
    CONTEXT  = "<CONTEXT>"
    RESPONSE = "<RESPONSE>"
    
    SPECIAL_TOKENS = [
        PAD, UNK, BOS, EOS, MASK,
        REASON_START, REASON_END, THOUGHT, STEP, CONCLUSION,
        THEREFORE, BECAUSE, IF, THEN, AND, OR, NOT, SINCE, HENCE,
        QUESTION, ANSWER, CONTEXT, RESPONSE
    ]
    
    def __init__(self, vocab_size: int = 8192, min_freq: int = 2):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.char2idx: Dict[str, int] = {}
        self.idx2char: Dict[int, str] = {}
        self.vocab: Set[str] = set()
        self.merges: List[Tuple[str, str]] = []
        self._special_token_ids: Dict[str, int] = {}
        
        # Reasoning patterns for loss weighting
        self.reasoning_patterns = [
            r"<REASON>|</REASON>",
            r"because|since|therefore|hence|thus",
            r"if.*then|when.*then|since.*then",
            r"step\s*\d+|\d+\.",
            r"conclusion:|in conclusion|finally",
            r"first|second|third|finally",
            r"however|although|but|yet",
            r"so|therefore|consequently",
            r"the reason is|this means|this implies",
            r"think|believe|consider|analyze",
        ]
        self.reasoning_regex = re.compile('|'.join(self.reasoning_patterns), re.IGNORECASE)
    
    def build(self, texts: List[str]) -> None:
        """Build vocabulary from texts."""
        log.info(f"Building vocabulary from {len(texts)} texts...")
        
        # Initialize with special tokens
        for i, token in enumerate(self.SPECIAL_TOKENS):
            self._special_token_ids[token] = i
        
        # Count character frequencies
        char_freq = Counter()
        for text in texts:
            for char in text:
                char_freq[char] += 1
        
        # Initialize vocab with characters
        self.vocab = {c for c, f in char_freq.items() if f >= self.min_freq}
        
        # Initialize mappings
        next_id = len(self.SPECIAL_TOKENS)
        self.char2idx = {t: i for i, t in enumerate(self.SPECIAL_TOKENS)}
        for c in self.vocab:
            if c not in self.char2idx:
                self.char2idx[c] = next_id
                next_id += 1
        
        # BPE merges
        self._build_bpe(texts)
        
        # Finalize vocab
        self._finalize_vocab()
        
        log.info(f"Vocabulary built: {len(self.char2idx)} tokens")
    
    def _build_bpe(self, texts: List[str], num_merges: int = 1000):
        """Build BPE merges from texts."""
        # Tokenize into words
        words = []
        for text in texts:
            # Split on whitespace and punctuation
            tokens = re.findall(r'\w+|[^\w\s]', text)
            words.extend([tuple(t) for t in tokens if t])
        
        # Count bigram frequencies
        while len(self.merges) < num_merges and len(self.vocab) < self.vocab_size:
            # Count all bigrams
            bigram_freq = Counter()
            for word in words:
                for i in range(len(word) - 1):
                    bigram = (word[i], word[i+1])
                    bigram_freq[bigram] += 1
            
            if not bigram_freq:
                break
            
            # Find most common bigram
            best_bigram = bigram_freq.most_common(1)[0][0]
            
            # Merge in all words
            new_words = []
            for word in words:
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i+1]) == best_bigram:
                        new_word.append(best_bigram[0] + best_bigram[1])
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_words.append(tuple(new_word))
            
            words = new_words
            self.merges.append(best_bigram)
            self.vocab.add(best_bigram[0] + best_bigram[1])
            
            if len(self.merges) % 100 == 0:
                log.info(f"BPE merge {len(self.merges)}: {best_bigram[0]}+{best_bigram[1]} (freq={bigram_freq[best_bigram]})")
    
    def _finalize_vocab(self):
        """Finalize vocabulary mappings."""
        self.idx2char = {i: t for t, i in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs with reasoning markers."""
        tokens = []
        
        if add_special_tokens:
            tokens.append(self.char2idx.get(self.BOS, 0))
        
        # Pre-process for reasoning markers
        processed = text
        
        # Add reasoning markers for logical patterns
        processed = self._add_reasoning_markers(processed)
        
        # Tokenize with BPE
        words = re.findall(r'\w+|[^\w\s]', processed)
        
        for word in words:
            if word in self.char2idx:
                tokens.append(self.char2idx[word])
            elif len(word) == 1 and word in self.vocab:
                tokens.append(self.char2idx.get(word, self.char2idx[self.UNK]))
            else:
                # BPE-style tokenization
                subword_tokens = self._bpe_tokenize(word)
                tokens.extend(subword_tokens)
        
        if add_special_tokens:
            tokens.append(self.char2idx.get(self.EOS, 0))
        
        return tokens
    
    def _add_reasoning_markers(self, text: str) -> str:
        """Add reasoning markers to text."""
        # This is for pre-processing, mark logical patterns
        patterns = [
            (r'\bbecause\b', ' BECAUSE '),
            (r'\btherefore\b', ' THEREFORE '),
            (r'\bhence\b', ' HENCE '),
            (r'\bthus\b', ' HENCE '),
            (r'\bsince\b', ' SINCE '),
            (r'\bif\b', ' IF '),
            (r'\bthen\b', ' THEN '),
            (r'\bhowever\b', ' HOWEVER '),
            (r'\balthough\b', ' ALTHOUGH '),
            (r'\bfirst\b', ' FIRST '),
            (r'\bsecond\b', ' SECOND '),
            (r'\bthird\b', ' THIRD '),
            (r'\bfinally\b', ' FINALLY '),
            (r'\bStep\s*(\d+)', r' STEP \1 '),
            (r'\b(\d+)\.', r' \1 DOT '),
        ]
        
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _bpe_tokenize(self, word: str) -> List[int]:
        """BPE-style tokenization."""
        if not word:
            return []
        
        # Start with characters
        tokens = list(word)
        
        # Apply merges
        for merge in self.merges:
            merged = merge[0] + merge[1]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] + tokens[i+1] == merged:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        # Convert to IDs
        result = []
        for t in tokens:
            if t in self.char2idx:
                result.append(self.char2idx[t])
            elif t.lower() in self.char2idx:
                result.append(self.char2idx[t.lower()])
            else:
                result.append(self.char2idx.get(self.UNK, 1))
        
        return result
    
    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs to text."""
        chars = []
        special_ids = {self.char2idx[t] for t in self.SPECIAL_TOKENS if t in self.char2idx}
        
        for i in ids:
            token = self.idx2char.get(i, "")
            if skip_special and i in special_ids:
                continue
            chars.append(token)
        
        return "".join(chars)
    
    def get_reasoning_mask(self, ids: List[int]) -> List[float]:
        """Get loss weight mask for reasoning tokens."""
        weights = []
        special_ids = {self.char2idx.get(t, -1) for t in self.SPECIAL_TOKENS}
        logic_ids = {
            self.char2idx.get(t, -1) for t in [
                self.THEREFORE, self.BECAUSE, self.IF, self.THEN,
                self.SINCE, self.HENCE, self.AND, self.OR, self.NOT,
                self.REASON_START, self.REASON_END, self.STEP, self.THOUGHT
            ]
        }
        
        for i in ids:
            if i in special_ids:
                weights.append(0.0)  # Don't train on special tokens
            elif i in logic_ids:
                weights.append(3.0)  # High weight for logical tokens
            else:
                weights.append(1.0)  # Normal weight
        
        return weights
    
    def save(self, path: str) -> None:
        """Save tokenizer to file."""
        data = {
            "char2idx": self.char2idx,
            "idx2char": {str(k): v for k, v in self.idx2char.items()},
            "vocab_size": self.vocab_size,
            "merges": [list(m) for m in self.merges],
            "special_token_ids": {t: i for t, i in self._special_token_ids.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> "AdvancedTokenizer":
        """Load tokenizer from file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        
        t = cls()
        t.char2idx = data["char2idx"]
        t.idx2char = {int(k): v for k, v in data["idx2char"].items()}
        t.vocab_size = data["vocab_size"]
        t.merges = [tuple(m) for m in data.get("merges", [])]
        t._special_token_ids = data.get("special_token_ids", {})
        
        return t


class ReasoningTokenizer(AdvancedTokenizer):
    """Extended tokenizer with focus on reasoning patterns."""
    
    # Additional reasoning tokens
    ANALYSIS   = "<ANALYSIS>"
    HYPOTHESIS = "<HYPOTHESIS>"
    EVIDENCE   = "<EVIDENCE>"
    CONCLUSION = "<CONCLUSION>"
    REFLECTION = "<REFLECTION>"
    
    def __init__(self, vocab_size: int = 16384, min_freq: int = 2):
        super().__init__(vocab_size, min_freq)
        self.reasoning_patterns = [
            r"<REASON>|<REASON>",
            r"because|since|therefore|hence|thus",
            r"if.*then|when.*then",
            r"step\s*\d+",
            r"first|second|third|finally",
            r"however|although|but|yet",
            r"think|believe|consider|analyze",
            r"evidence shows|research suggests",
            r"in conclusion|to summarize",
            r"this means|this implies|this suggests",
        ]
        self.reasoning_regex = re.compile('|'.join(self.reasoning_patterns), re.IGNORECASE)
