"""
Unified Training Engine
=====================
Single training interface for all model types:
- Text Generation (LM)
- Image Classification
- Tabular Classification
- Regression
"""

import time
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

from core.implementations import HMTLanguageModel, HMTClassifier, HMTImageClassifier
from core.text_model import lm_train_step, lm_val_loss, save_lm
from core.device_manager import get_best_device, move_batch
from data.text_dataset import build_text_loaders
from training.reasoning_trainer import ReasoningAwareLoss, CurriculumScheduler
from data.image_dataset import build_image_loaders
from data.data_loader import build_loaders

try:
    from data.hf_dataset_loader import build_hf_loaders, DATASETS_AVAILABLE
except:
    DATASETS_AVAILABLE = False
    build_hf_loaders = None


class ModelType(Enum):
    LANGUAGE_MODEL = "language_model"
    CLASSIFIER = "classifier"
    IMAGE_CLASSIFIER = "image_classifier"
    REGRESSION = "regression"


@dataclass
class TrainConfig:
    """Training configuration."""
    model_type: ModelType = ModelType.CLASSIFIER
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-4
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    seq_len: int = 128
    optimizer: str = "Adam"
    scheduler: str = "None"
    use_reflector: bool = False
    reasoning_weight: float = 1.0
    logic_weight: float = 2.5
    curriculum: bool = False
    focal_loss: bool = False
    dataset_name: str = ""
    reasoning_only: bool = False


@dataclass
class TrainResult:
    """Training result."""
    model: nn.Module
    best_state: Dict
    metrics: Dict[str, List[float]]
    info: Dict
    tokenizer: Any = None


class TrainingRuntimeError(RuntimeError):
    """Structured runtime error with stable error code."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


class UnifiedTrainer:
    """
    Unified trainer for all model types.
    Clean separation: UI calls this, this handles all logic.
    """
    
    def __init__(
        self,
        config: TrainConfig,
        files: List[str],
        progress_callback: Optional[Callable] = None,
        log_callback: Optional[Callable] = None,
    ):
        self.config = config
        self.files = files
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        
        self.device = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.tokenizer = None
        self.train_loader = None
        self.val_loader = None
        self.data_info = None
        
        self.stop_flag = None
        self.best_state = None
        self.best_loss = float("inf")
        self.start_time = time.time()
        
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "accuracy": [],
            "lr": [],
        }
    
    def set_stop_flag(self, flag):
        self.stop_flag = flag
    
    def log(self, msg: str, level: str = "info"):
        if self.log_callback:
            self.log_callback(msg, level)
    
    def progress(self, **kwargs):
        if self.progress_callback:
            self.progress_callback(**kwargs)
    
    def run(self) -> TrainResult:
        """Main training entry point."""
        self.log(f"Starting {self.config.model_type.value} training...")
        if self.config.use_reflector:
            self.log(
                "Reflector is enabled in config. UnifiedTrainer currently applies reasoning-aware "
                "training only; dedicated reflector-integrated training is in training/reflector_trainer.py.",
                "warn",
            )
        try:
            if self.config.model_type == ModelType.LANGUAGE_MODEL:
                return self._train_lm()
            elif self.config.model_type == ModelType.IMAGE_CLASSIFIER:
                return self._train_image()
            elif self.config.model_type == ModelType.CLASSIFIER:
                return self._train_tabular()
            else:
                return self._train_tabular()
        except TrainingRuntimeError:
            raise
        except Exception as e:
            raise TrainingRuntimeError("TRN-RUN-001", str(e)) from e
    
    def _setup_device(self, param_count: int, batch_size: int):
        """Setup compute device."""
        # Production-safe policy:
        # - Prefer CUDA when present.
        # - Otherwise force CPU. DirectML has shown unstable behavior in LM/tabular training.
        try:
            force = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            force = "cpu"
        self.device, name = get_best_device(param_count, batch_size, force=force)
        self.log(f"Device: {name}")
    
    def _create_optimizer(self):
        """Create optimizer from config."""
        opt_name = self.config.optimizer
        if opt_name == "AdamW":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.config.lr, weight_decay=0.01)
        elif opt_name == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.config.lr, momentum=0.9)
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.config.lr)
        
        sched = self.config.scheduler
        if sched == "CosineAnnealing":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.epochs)
        elif sched == "StepLR":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=max(1, self.config.epochs // 3), gamma=0.5)

    def _build_reasoning_weights(self, yb: torch.Tensor, boost: float) -> torch.Tensor:
        """
        Build token-level weights for reasoning-heavy spans.
        Uses lightweight keyword matching on decoded char-level sequences.
        """
        weights = torch.ones_like(yb, dtype=torch.float, device=yb.device)
        if boost <= 1.0 or self.tokenizer is None:
            return weights
        if not hasattr(self.tokenizer, "decode"):
            return weights

        keywords = (
            "because", "therefore", "reason", "thought", "hence",
            "since", "thus", "step", "logic", "if", "then",
        )
        try:
            y_cpu = yb.detach().cpu().tolist()
            for i, row in enumerate(y_cpu):
                text = self.tokenizer.decode(row, skip_special=False).lower()
                if not text:
                    continue
                for kw in keywords:
                    start = 0
                    while True:
                        pos = text.find(kw, start)
                        if pos < 0:
                            break
                        # Char-level tokenizer: token index ~= char index.
                        lo = max(0, pos)
                        hi = min(len(row), pos + len(kw) + 8)  # include local context after keyword
                        if hi > lo:
                            weights[i, lo:hi] = boost
                        start = pos + 1
        except Exception:
            return torch.ones_like(yb, dtype=torch.float, device=yb.device)

        return weights
    
    def _train_lm(self) -> TrainResult:
        """Train language model."""
        self.log("Loading text data...")
        
        # Load data
        if self.config.dataset_name and DATASETS_AVAILABLE:
            try:
                self.train_loader, self.val_loader, self.tokenizer, self.data_info = build_hf_loaders(
                    self.config.dataset_name,
                    seq_len=self.config.seq_len,
                    batch_size=self.config.batch_size,
                )
                self.log(f"Loaded dataset: {self.config.dataset_name}")
            except Exception as e:
                self.log(f"Dataset load failed: {e}", "err")
                return None
        else:
            if not self.files:
                self.log("No files provided", "err")
                return None
            
            self.train_loader, self.val_loader, self.tokenizer, self.data_info = build_text_loaders(
                self.files,
                seq_len=self.config.seq_len,
                batch_size=self.config.batch_size,
                reasoning_only=self.config.reasoning_only,
            )
            self.log(f"Corpus: {self.data_info['corpus_chars']:,} chars, vocab={self.data_info['vocab_size']}")
            if self.config.reasoning_only:
                self.log("Reasoning-only dataset filter is enabled.")
        
        # Build model
        hidden = self.config.hidden_dim
        heads = max(1, min(8, hidden // 64))
        hidden = (hidden // heads) * heads
        
        self.model = HMTLanguageModel(
            vocab_size=self.data_info["vocab_size"],
            dim=hidden,
            num_layers=self.config.num_layers,
            num_heads=heads,
            num_scales=3,
            max_seq=self.config.seq_len,
            dropout=0.1,
        )
        
        param_count = sum(p.numel() for p in self.model.parameters())
        self._setup_device(param_count, self.config.batch_size)
        self.model = self.model.to(self.device)
        self.log(f"Model: {param_count:,} params, dim={hidden}, layers={self.config.num_layers}")
        
        self._create_optimizer()
        
        # Setup loss
        loss_fn = None
        curriculum = None
        if self.config.reasoning_weight > 1.0:
            try:
                loss_fn = ReasoningAwareLoss(
                    reasoning_weight=self.config.reasoning_weight,
                    logic_weight=self.config.logic_weight,
                    use_focal=self.config.focal_loss,
                )
                if self.config.curriculum:
                    curriculum = CurriculumScheduler(total_steps=10000)
                self.log(f"Reasoning loss: weight={self.config.reasoning_weight}")
            except ImportError:
                self.log("Reasoning trainer not available", "warn")
        
        # Training loop
        n_batches = self.data_info.get("train_batches", 100)
        max_steps = min(n_batches, 2000)
        total_steps = self.config.epochs * max_steps
        self.log(
            f"LM setup: train_batches={n_batches}, max_steps={max_steps}, "
            f"batch={self.config.batch_size}, seq_len={self.config.seq_len}"
        )
        if max_steps <= 0:
            self.log("No training batches available for current settings.", "err")
            return None
        
        for epoch in range(1, self.config.epochs + 1):
            if self.stop_flag and self.stop_flag.is_set():
                self.log("Training stopped", "warn")
                break
            
            # Update curriculum
            if curriculum:
                params = curriculum.step((epoch - 1) * max_steps)
                if loss_fn:
                    loss_fn.reasoning_weight = params["reasoning_weight"]
            
            epoch_loss = 0
            step = 0
            self.model.train()
            
            for xb, yb in self.train_loader:
                if self.stop_flag and self.stop_flag.is_set():
                    break
                if step >= max_steps:
                    break
                
                xb, yb = move_batch((xb, yb), self.device)
                
                if loss_fn:
                    logits = self.model(xb)
                    weights = self._build_reasoning_weights(
                        yb, boost=float(loss_fn.reasoning_weight)
                    )
                    loss, loss_meta = loss_fn(logits, yb, weights)
                    
                    loss_val = loss.item()
                    if math.isnan(loss_val) or math.isinf(loss_val):
                        continue
                    if step == 0:
                        pct = 100.0 * float(loss_meta.get("high_weight_tokens", 0.0))
                        self.log(f"Reasoning token emphasis active: {pct:.1f}% tokens boosted.")
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                else:
                    loss_val = lm_train_step(
                        self.model, self.optimizer, xb, yb,
                        reasoning_weight=self.config.reasoning_weight,
                        tokenizer=self.tokenizer
                    )
                
                epoch_loss += loss_val
                step += 1

                if step == 1:
                    self.log("First training step completed; progress will update continuously.")
                
                # Progress update
                now = time.time()
                if step == 1 or step % 5 == 0 or step == max_steps:
                    avg_loss = epoch_loss / max(step, 1)
                    elapsed = now - self.start_time
                    done = (epoch - 1) * max_steps + step
                    eta = int((elapsed / done) * (total_steps - done))
                    
                    stage = ""
                    if curriculum:
                        stage = f" [{curriculum.stages[curriculum.current_stage]['name']}]"
                    
                    self.progress(
                        epoch=epoch,
                        epochs=self.config.epochs,
                        loss=avg_loss,
                        lr=self.optimizer.param_groups[0]["lr"],
                        eta=f"{eta//60}m {eta%60}s",
                        stage=stage,
                        pct=(done / total_steps) * 100,
                    )
            
            # Validation
            avg_loss = epoch_loss / max(step, 1)
            self.metrics["train_loss"].append(avg_loss)
            
            if self.val_loader:
                val_loss = lm_val_loss(self.model, self.val_loader, device=self.device, max_batches=50)
                self.metrics["val_loss"].append(val_loss)
                
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                
                self.log(f"Epoch {epoch}/{self.config.epochs} | loss={avg_loss:.4f} | val={val_loss:.4f}")
            else:
                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    self.best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                self.log(f"Epoch {epoch}/{self.config.epochs} | loss={avg_loss:.4f}")
            
            if self.scheduler:
                self.scheduler.step()
        
        self.log("Training complete!")
        
        return TrainResult(
            model=self.model,
            best_state=self.best_state,
            metrics=self.metrics,
            info=self.data_info,
            tokenizer=self.tokenizer,
        )
    
    def _train_image(self) -> TrainResult:
        """Train image classifier."""
        self.log("Loading image data...")
        
        if not self.files:
            self.log("No files provided", "err")
            return None
        
        self.train_loader, self.val_loader, class_names, self.data_info = build_image_loaders(
            self.files, batch_size=self.config.batch_size)
        
        self.log(f"Classes: {len(class_names)}, samples: {len(self.train_loader.dataset)}")
        
        # Build model
        self.model = HMTImageClassifier(
            num_classes=len(class_names),
            dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            dropout=0.1,
        )
        
        param_count = sum(p.numel() for p in self.model.parameters())
        self._setup_device(param_count, self.config.batch_size)
        self.model = self.model.to(self.device)
        
        self._create_optimizer()
        
        # Training loop
        max_steps = min(len(self.train_loader), 2000)
        total_steps = self.config.epochs * max_steps
        
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(1, self.config.epochs + 1):
            if self.stop_flag and self.stop_flag.is_set():
                break
            
            epoch_loss = 0
            correct = 0
            total = 0
            step = 0
            self.model.train()
            
            for xb, yb in self.train_loader:
                if self.stop_flag and self.stop_flag.is_set():
                    break
                if step >= max_steps:
                    break
                
                xb, yb = xb.to(self.device), yb.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                total += yb.size(0)
                correct += predicted.eq(yb).sum().item()
                step += 1
                
                if step % 20 == 0:
                    elapsed = time.time() - self.start_time
                    done = (epoch - 1) * max_steps + step
                    eta = int((elapsed / done) * (total_steps - done))
                    acc = 100. * correct / total
                    self.progress(epoch=epoch, epochs=self.config.epochs,
                                loss=epoch_loss/step, accuracy=acc,
                                eta=f"{eta//60}m {eta%60}s", pct=(done/total_steps)*100)
            
            avg_loss = epoch_loss / max(step, 1)
            acc = 100. * correct / max(total, 1)
            self.metrics["train_loss"].append(avg_loss)
            self.metrics["accuracy"].append(acc)
            
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            
            self.log(f"Epoch {epoch}/{self.config.epochs} | loss={avg_loss:.4f} | acc={acc:.1f}%")
        
        self.log("Training complete!")
        return TrainResult(model=self.model, best_state=self.best_state,
                          metrics=self.metrics, info=self.data_info)
    
    def _train_tabular(self) -> TrainResult:
        """Train tabular classifier/regressor."""
        self.log("Loading tabular data...")
        
        if not self.files:
            self.log("No files provided", "err")
            return None
        
        self.train_loader, self.val_loader, self.data_info = build_loaders(
            self.files, batch_size=self.config.batch_size)
        
        num_classes = self.data_info.get("num_classes", 2)
        # data_loader reports `feature_dim`; keep `num_features` for compatibility.
        num_features = self.data_info.get(
            "feature_dim",
            self.data_info.get("num_features", self.config.hidden_dim),
        )
        
        self.log(f"Features: {num_features}, Classes: {num_classes}")
        
        self.model = HMTClassifier(
            input_dim=num_features,
            num_classes=num_classes,
            dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=0.1,
        )
        
        param_count = sum(p.numel() for p in self.model.parameters())
        self._setup_device(param_count, self.config.batch_size)
        self.model = self.model.to(self.device)
        
        self._create_optimizer()
        
        max_steps = min(len(self.train_loader), 2000)
        total_steps = self.config.epochs * max_steps
        
        if num_classes > 1:
            # Classification
            criterion = nn.CrossEntropyLoss()
        else:
            # Regression
            criterion = nn.MSELoss()
        
        for epoch in range(1, self.config.epochs + 1):
            if self.stop_flag and self.stop_flag.is_set():
                break
            
            epoch_loss = 0
            correct = 0
            total = 0
            step = 0
            self.model.train()
            
            for xb, yb in self.train_loader:
                if self.stop_flag and self.stop_flag.is_set():
                    break
                if step >= max_steps:
                    break
                
                xb, yb = xb.to(self.device), yb.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(xb)
                
                if num_classes > 1:
                    # CrossEntropy expects (B,) long labels
                    loss = criterion(outputs, yb.squeeze(1).long())
                else:
                    # MSE expects same shape (B, 1) float labels
                    loss = criterion(outputs, yb.float())

                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                if num_classes > 1:
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(yb.squeeze(1)).sum().item()
                total += yb.size(0)
                step += 1
                
                if step % 20 == 0:
                    elapsed = time.time() - self.start_time
                    done = (epoch - 1) * max_steps + step
                    eta = int((elapsed / done) * (total_steps - done))
                    acc = 100. * correct / max(total, 1) if total > 0 else 0
                    self.progress(epoch=epoch, epochs=self.config.epochs,
                                loss=epoch_loss/step, accuracy=acc,
                                eta=f"{eta//60}m {eta%60}s", pct=(done/total_steps)*100)
            
            avg_loss = epoch_loss / max(step, 1)
            acc = 100. * correct / max(total, 1) if total > 0 else 0
            self.metrics["train_loss"].append(avg_loss)
            self.metrics["accuracy"].append(acc)
            
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            
            self.log(f"Epoch {epoch}/{self.config.epochs} | loss={avg_loss:.4f} | acc={acc:.1f}%")
        
        self.log("Training complete!")
        return TrainResult(model=self.model, best_state=self.best_state,
                          metrics=self.metrics, info=self.data_info)


def train_model(config: TrainConfig, files: List[str],
                progress_cb: Optional[Callable] = None,
                log_cb: Optional[Callable] = None,
                stop_flag=None) -> TrainResult:
    """Simple entry point for training."""
    trainer = UnifiedTrainer(config, files, progress_cb, log_cb)
    if stop_flag:
        trainer.set_stop_flag(stop_flag)
    return trainer.run()
