"""
Reasoning-Aware Training System
==============================
Features:
- Multi-task learning (language modeling + reasoning)
- Reasoning-aware loss weighting
- Curriculum learning
- Multi-format data support (text, CSV, JSON, Q&A, reasoning chains)
- Dynamic data mixing
"""

import math
import random
from typing import List, Dict, Tuple, Optional, Callable, Any
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from data.advanced_tokenizer import AdvancedTokenizer, ReasoningTokenizer
from data.text_dataset import CharTokenizer

log = logging.getLogger("ReasoningTrainer")


class TaskType(Enum):
    LANGUAGE_MODEL = "language_model"
    REASONING = "reasoning"
    QA = "question_answering"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"


@dataclass
class TrainingConfig:
    """Training configuration with reasoning support."""
    vocab_size: int = 8192
    seq_len: int = 256
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    
    # Reasoning settings
    reasoning_weight: float = 3.0
    logic_weight: float = 2.5
    coherence_weight: float = 1.5
    base_weight: float = 1.0
    
    # Loss weights
    use_weighted_loss: bool = True
    use_curriculum: bool = True
    
    # Training
    epochs: int = 10
    warmup_steps: int = 500
    gradient_clip: float = 1.0
    log_interval: int = 10
    
    # Data mixing
    mix_ratio: Dict[str, float] = field(default_factory=lambda: {
        "general": 0.4,
        "reasoning": 0.3,
        "qa": 0.2,
        "logic": 0.1,
    })


class ReasoningDataset(Dataset):
    """
    Multi-task dataset with reasoning support.
    Handles text, Q&A, reasoning chains, and logic problems.
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: AdvancedTokenizer,
        seq_len: int = 256,
        task_type: TaskType = TaskType.LANGUAGE_MODEL,
        reasoning_weight: float = 3.0,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.task_type = task_type
        self.reasoning_weight = reasoning_weight
        
        # Encode all texts
        self.encodings = []
        self.masks = []
        
        for text in texts:
            ids = tokenizer.encode(text)
            self.encodings.append(ids)
            
            # Get reasoning mask
            mask = tokenizer.get_reasoning_mask(ids)
            self.masks.append(mask)
        
        # Pre-compute valid indices
        self.indices = self._build_indices()
    
    def _build_indices(self) -> List[int]:
        """Build valid starting indices for each sequence."""
        indices = []
        for i, enc in enumerate(self.encodings):
            if len(enc) > 2:
                indices.append(i)
        return indices
    
    def __len__(self) -> int:
        return len(self.indices) * 100  # Many epochs
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a training example."""
        # Sample from indices (with repetition for multiple epochs)
        seq_idx = self.indices[idx % len(self.indices)]
        enc = self.encodings[seq_idx]
        mask = self.masks[seq_idx]
        
        # Random starting position
        max_start = max(0, len(enc) - self.seq_len - 1)
        start = random.randint(0, max_start) if max_start > 0 else 0
        
        # Extract sequence
        seq = enc[start:start + self.seq_len + 1]
        msk = mask[start:start + self.seq_len + 1]
        
        # Pad if needed
        if len(seq) < self.seq_len + 1:
            seq = seq + [self.tokenizer.char2idx.get("<PAD>", 0)] * (self.seq_len + 1 - len(seq))
            msk = msk + [0.0] * (self.seq_len + 1 - len(msk))
        
        # Create input and target
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        weights = torch.tensor(msk[:-1], dtype=torch.float)
        
        return x, y, weights


class ReasoningAwareLoss(nn.Module):
    """
    Multi-component loss with reasoning awareness.
    
    Components:
    1. Standard cross-entropy loss
    2. Reasoning token boost
    3. Logical coherence regularization
    """
    
    def __init__(
        self,
        reasoning_weight: float = 3.0,
        logic_weight: float = 2.5,
        use_focal: bool = False,
    ):
        super().__init__()
        self.reasoning_weight = reasoning_weight
        self.logic_weight = logic_weight
        self.use_focal = use_focal
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
        if use_focal:
            self.focal_gamma = 2.0
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute reasoning-aware loss.
        
        Args:
            logits: (B, T, V)
            targets: (B, T)
            weights: (B, T) token-level weights
            
        Returns:
            loss: scalar loss
            metrics: dict of loss components
        """
        B, T, V = logits.shape
        
        # Flatten for loss computation
        logits_flat = logits.view(-1, V)
        targets_flat = targets.view(-1)
        
        # Standard cross-entropy
        ce_loss = self.ce_loss(logits_flat, targets_flat)  # (B*T,)
        
        # Apply focal loss if enabled
        if self.use_focal:
            pt = torch.exp(-ce_loss)
            focal_weight = (1 - pt) ** self.focal_gamma
            ce_loss = focal_weight * ce_loss
        
        # Apply token weights
        if weights is not None:
            weights_flat = weights.view(-1)
            # Optional global boost for highlighted reasoning tokens.
            if self.reasoning_weight > 1.0:
                weights_flat = torch.where(
                    weights_flat > 1.0,
                    weights_flat * self.reasoning_weight,
                    weights_flat,
                )
            # Normalize weights
            weight_sum = weights_flat.sum() + 1e-8
            loss = (ce_loss * weights_flat).sum() / weight_sum
        else:
            loss = ce_loss.mean()
        
        # Compute metrics
        with torch.no_grad():
            high_weight_ratio = 0.0
            if weights is not None:
                weights_flat = weights.view(-1)
                high_weight_ratio = (weights_flat > 2.0).sum().item() / len(weights_flat)
            
            metrics = {
                "total_loss": loss.item(),
                "avg_prob": torch.exp(-ce_loss).mean().item(),
                "high_weight_tokens": high_weight_ratio,
            }
        
        return loss, metrics


class CurriculumScheduler:
    """
    Curriculum learning scheduler.
    
    Stages:
    1. Simple sentences (short, common words)
    2. Compound sentences (longer, more vocabulary)
    3. Complex reasoning (logical connectives, chains)
    4. Mixed difficulty (all types)
    """
    
    def __init__(
        self,
        stages: List[Dict[str, Any]] = None,
        total_steps: int = 10000,
    ):
        if stages is None:
            stages = [
                {"name": "simple", "difficulty": 0.2, "reasoning_weight": 1.0, "duration": 0.15},
                {"name": "medium", "difficulty": 0.5, "reasoning_weight": 2.0, "duration": 0.25},
                {"name": "hard", "difficulty": 0.8, "reasoning_weight": 3.0, "duration": 0.30},
                {"name": "expert", "difficulty": 1.0, "reasoning_weight": 4.0, "duration": 0.30},
            ]
        
        self.stages = stages
        self.total_steps = total_steps
        self.cumulative = [0.0]
        for s in stages:
            self.cumulative.append(self.cumulative[-1] + s["duration"])
        
        self.current_stage = 0
        self.stage_progress = 0.0
    
    def step(self, global_step: int) -> Dict[str, float]:
        """Get curriculum parameters for current step."""
        # Determine stage
        progress = min(global_step / self.total_steps, 1.0)
        
        for i, cum in enumerate(self.cumulative[1:], 1):
            if progress < cum:
                self.current_stage = i - 1
                prev_cum = self.cumulative[i - 1]
                stage_len = self.cumulative[i] - prev_cum
                self.stage_progress = (progress - prev_cum) / stage_len
                break
        
        stage = self.stages[self.current_stage]
        
        # Interpolate within stage
        difficulty = stage["difficulty"]
        reasoning_weight = stage["reasoning_weight"]
        
        # Smooth transition
        if self.stage_progress < 0.2:
            # Warm-up within stage
            factor = self.stage_progress / 0.2
            difficulty *= factor
            reasoning_weight = 1.0 + (reasoning_weight - 1.0) * factor
        elif self.stage_progress > 0.8:
            # Cool-down: prepare for next stage
            factor = (self.stage_progress - 0.8) / 0.2
            next_stage = self.stages[min(self.current_stage + 1, len(self.stages) - 1)]
            difficulty = difficulty + (next_stage["difficulty"] - difficulty) * factor
            reasoning_weight = reasoning_weight + (next_stage["reasoning_weight"] - reasoning_weight) * factor
        
        return {
            "difficulty": difficulty,
            "reasoning_weight": reasoning_weight,
            "stage_name": stage["name"],
            "stage_progress": self.stage_progress,
        }


class MultiFormatDataLoader:
    """
    Load and process data from multiple formats:
    - Plain text files (.txt)
    - CSV files (text columns)
    - JSON/JSONL files
    - Q&A pairs
    - Reasoning chains
    """
    
    def __init__(self, tokenizer: AdvancedTokenizer, seq_len: int = 256):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
    
    def load_file(self, path: str) -> List[str]:
        """Load data from file, auto-detecting format."""
        path = Path(path)
        ext = path.suffix.lower()
        
        if ext == ".txt":
            return self._load_txt(path)
        elif ext == ".csv":
            return self._load_csv(path)
        elif ext == ".json":
            return self._load_json(path)
        elif ext == ".jsonl":
            return self._load_jsonl(path)
        else:
            log.warning(f"Unknown format: {ext}, treating as text")
            return self._load_txt(path)
    
    def _load_txt(self, path: Path) -> List[str]:
        """Load plain text file."""
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        # Split into lines and filter
        lines = [l.strip() for l in content.split("\n") if l.strip()]
        return lines
    
    def _load_csv(self, path: Path) -> List[str]:
        """Load CSV file, extracting text columns."""
        import pandas as pd
        df = pd.read_csv(path)
        
        texts = []
        text_cols = [c for c in df.columns if any(
            k in c.lower() for k in ["text", "content", "message", "body", "sentence", "review"]
        )]
        
        if text_cols:
            for col in text_cols:
                texts.extend(df[col].dropna().astype(str).tolist())
        else:
            # Use all object columns
            for col in df.select_dtypes(include="object").columns:
                texts.extend(df[col].dropna().astype(str).tolist())
        
        return texts
    
    def _load_json(self, path: Path) -> List[str]:
        """Load JSON file."""
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        texts = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict):
                    for key in ["text", "content", "message", "body"]:
                        if key in item:
                            texts.append(str(item[key]))
                            break
        elif isinstance(data, str):
            texts.append(data)
        
        return texts
    
    def _load_jsonl(self, path: Path) -> List[str]:
        """Load JSONL file."""
        import json
        texts = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if isinstance(obj, str):
                        texts.append(obj)
                    elif isinstance(obj, dict):
                        for key in ["text", "content", "message", "input", "output"]:
                            if key in obj:
                                texts.append(str(obj[key]))
                                break
                except json.JSONDecodeError:
                    continue
        return texts
    
    def load_qa_pairs(self, path: str) -> List[str]:
        """Load Q&A pairs as training data."""
        path = Path(path)
        if path.suffix == ".csv":
            texts = self._load_csv(path)
        elif path.suffix == ".json":
            texts = self._load_json(path)
        else:
            return []
        
        # Format as Q: ... A: ...
        formatted = []
        for i in range(0, len(texts) - 1, 2):
            if i + 1 < len(texts):
                formatted.append(f"Question: {texts[i]} Answer: {texts[i+1]}")
        
        return formatted
    
    def load_reasoning_chains(self, path: str) -> List[str]:
        """Load reasoning chains (premise → step → step → conclusion)."""
        path = Path(path)
        texts = []
        
        if path.suffix == ".json":
            import json
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for item in data:
                if isinstance(item, dict) and "chain" in item:
                    chain = item["chain"]
                    formatted = " ".join(f"Step {i}: {s}" for i, s in enumerate(chain, 1))
                    texts.append(formatted)
                elif isinstance(item, list):
                    formatted = " ".join(f"Step {i}: {s}" for i, s in enumerate(item, 1))
                    texts.append(formatted)
        
        return texts


def create_reasoning_trainer(
    model,
    config: TrainingConfig,
    device: torch.device,
) -> "ReasoningTrainer":
    """Create a reasoning-aware trainer."""
    return ReasoningTrainer(model, config, device)


class ReasoningTrainer:
    """
    Complete training system with reasoning awareness.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
    ):
        self.model = model
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            total_steps=config.epochs * 1000,
            pct_start=0.1,
        )
        
        # Loss function
        self.loss_fn = ReasoningAwareLoss(
            reasoning_weight=config.reasoning_weight,
            logic_weight=config.logic_weight,
        )
        
        # Curriculum
        self.curriculum = CurriculumScheduler(
            total_steps=config.epochs * 1000,
        )
        
        # Metrics
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "reasoning_loss": [],
            "learning_rate": [],
        }
    
    def train_step(
        self,
        xb: torch.Tensor,
        yb: torch.Tensor,
        weights: torch.Tensor,
    ) -> Tuple[float, Dict[str, float]]:
        """Single training step with reasoning awareness."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        logits = self.model(xb)
        
        # Compute loss
        loss, metrics = self.loss_fn(logits, yb, weights)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.gradient_clip
        )
        
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item(), metrics
    
    def evaluate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        correct = 0
        
        with torch.no_grad():
            for xb, yb, weights in val_loader:
                xb, yb, weights = xb.to(self.device), yb.to(self.device), weights.to(self.device)
                
                logits = self.model(xb)
                
                # Perplexity
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
                total_loss += loss.item() * xb.size(0)
                total_tokens += xb.size(0)
                
                # Accuracy
                preds = logits.argmax(dim=-1)
                correct += (preds[:, :-1] == yb[:, 1:]).sum().item()
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        accuracy = correct / (total_tokens * (xb.size(1) - 1))
        
        return {
            "val_loss": avg_loss,
            "val_perplexity": perplexity,
            "val_accuracy": accuracy,
        }
    
    def train(
        self,
        train_data: List[str],
        val_data: Optional[List[str]] = None,
        tokenizer: Optional[AdvancedTokenizer] = None,
        callbacks: List[Callable] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the model with reasoning awareness.
        
        Args:
            train_data: List of training texts
            val_data: Optional validation texts
            tokenizer: Tokenizer (will create default if None)
            callbacks: List of callbacks for logging, etc.
        """
        if callbacks is None:
            callbacks = []
        
        # Create tokenizer if not provided
        if tokenizer is None:
            tokenizer = ReasoningTokenizer(vocab_size=self.config.vocab_size)
            tokenizer.build(train_data)
        
        # Create datasets
        train_dataset = ReasoningDataset(
            train_data,
            tokenizer,
            seq_len=self.config.seq_len,
            reasoning_weight=self.config.reasoning_weight,
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        
        val_loader = None
        if val_data:
            val_dataset = ReasoningDataset(
                val_data,
                tokenizer,
                seq_len=self.config.seq_len,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
            )
        
        # Training loop
        global_step = 0
        
        for epoch in range(self.config.epochs):
            # Update curriculum
            curriculum_params = self.curriculum.step(global_step)
            
            # Update loss weights
            self.loss_fn.reasoning_weight = curriculum_params["reasoning_weight"]
            
            epoch_loss = 0
            epoch_steps = 0
            
            for batch in train_loader:
                xb, yb, weights = batch
                xb, yb, weights = xb.to(self.device), yb.to(self.device), weights.to(self.device)
                
                loss, metrics = self.train_step(xb, yb, weights)
                epoch_loss += loss
                epoch_steps += 1
                global_step += 1
                
                # Log
                if global_step % self.config.log_interval == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    log.info(
                        f"Step {global_step} | "
                        f"Loss: {loss:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Stage: {curriculum_params['stage_name']} | "
                        f"Reasoning W: {curriculum_params['reasoning_weight']:.1f}"
                    )
                    
                    # Callbacks
                    for cb in callbacks:
                        cb(global_step, loss, metrics, curriculum_params)
            
            # Validation
            if val_loader:
                val_metrics = self.evaluate(val_loader)
                log.info(
                    f"Epoch {epoch+1}/{self.config.epochs} | "
                    f"Train Loss: {epoch_loss/epoch_steps:.4f} | "
                    f"Val Loss: {val_metrics['val_loss']:.4f} | "
                    f"Val PPL: {val_metrics['val_perplexity']:.2f}"
                )
                self.metrics["val_loss"].append(val_metrics["val_loss"])
            else:
                log.info(f"Epoch {epoch+1}/{self.config.epochs} | Train Loss: {epoch_loss/epoch_steps:.4f}")
            
            self.metrics["train_loss"].append(epoch_loss / epoch_steps)
            self.metrics["learning_rate"].append(self.optimizer.param_groups[0]["lr"])
        
        return self.metrics
