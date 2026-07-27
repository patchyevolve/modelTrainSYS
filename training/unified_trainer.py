import time
import math
import copy
import gc
import os
import threading
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn

from core.implementations import HMTLanguageModel
from core.text_model import lm_val_loss, save_lm
from core.device_manager import get_best_device, move_batch
from core.memory_monitor import MemoryMonitor, get_available_mb, get_process_mb
from data.text_dataset import build_text_loaders
from data.advanced_tokenizer import AdvancedTokenizer


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: Optional[int] = None
    seq_len: int = 128
    optimizer: str = "AdamW"
    scheduler: str = "Cosine"
    use_amp: bool = True
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    use_gradient_checkpointing: bool = False
    ema_decay: float = 0.0
    max_ram_mb: int = 0
    num_workers: int = 0
    pin_memory: bool = False
    cpu_threads: int = 0
    use_subprocess: bool = False
    prompt_ratio: float = 0.3
    template_name: str = "step_by_step"
    template_fields: Optional[Dict[str, str]] = None
    resume_path: Optional[str] = None
    resume_freeze_layers: int = 0
    resume_freeze_embeddings: bool = False


@dataclass
class TrainResult:
    model: nn.Module
    ema_model: Optional[nn.Module]
    best_state: Dict
    metrics: Dict[str, List[float]]
    info: Dict
    tokenizer: Any = None
    optimizer_state: Optional[Dict] = None


class TrainingRuntimeError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


class UnifiedTrainer:
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
        self.ema_model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.tokenizer = None
        self.train_loader = None
        self.val_loader = None
        self.data_info = None

        self.stop_flag = None
        self.best_state = None
        self.best_loss = float("inf")
        self.start_time = time.time()
        self._criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self._mem_monitor: Optional[MemoryMonitor] = None
        self._resume_count = 0
        self._saved_files: List[str] = []

        self.metrics = {
            "train_loss": [],
            "val_loss": [],
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

    def _setup_threads(self):
        n = self.config.cpu_threads
        if n > 0:
            torch.set_num_threads(n)
            if hasattr(torch, 'set_num_interop_threads'):
                torch.set_num_interop_threads(max(2, n // 2))
            self.log(f"Torch threads set to {n}")

    def _setup_memory_monitor(self, initial_batch: int):
        if self.config.max_ram_mb > 0:
            self._mem_monitor = MemoryMonitor(max_ram_mb=self.config.max_ram_mb)
            self._mem_monitor.start(initial_batch=initial_batch)
            self.log(f"Memory monitor active (limit={self.config.max_ram_mb} MB)")

    def _setup_device(self):
        force = "cuda" if torch.cuda.is_available() else "cpu"
        self.device, name = get_best_device(0, self.config.batch_size, force=force)
        self.log(f"Device: {name}")
        if self.config.use_amp and "cuda" in str(self.device):
            self.scaler = torch.cuda.amp.GradScaler()
            self.log("AMP enabled")
        else:
            self.config.use_amp = False

    def _create_optimizer_and_scheduler(self, steps_per_epoch: int):
        total_steps = self.config.epochs * steps_per_epoch
        warmup = min(self.config.warmup_steps, total_steps // 10)

        opt_name = self.config.optimizer
        if opt_name == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.config.lr, momentum=0.9,
                weight_decay=self.config.weight_decay)
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.config.lr,
                weight_decay=self.config.weight_decay)

        sched = self.config.scheduler
        if sched == "Cosine":
            def lr_lambda(step):
                if step < warmup:
                    return step / max(1, warmup)
                progress = (step - warmup) / max(1, total_steps - warmup)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        elif sched == "StepLR":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=max(1, self.config.epochs // 3), gamma=0.5)

    def _init_ema(self):
        if self.config.ema_decay > 0.0:
            self.ema_model = copy.deepcopy(self.model)
            for p in self.ema_model.parameters():
                p.requires_grad_(False)
            self.log(f"EMA enabled (decay={self.config.ema_decay})")

    def _update_ema(self):
        if self.ema_model is None:
            return
        decay = self.config.ema_decay
        with torch.no_grad():
            for ema_p, p in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_p.lerp_(p, 1.0 - decay)

    def run(self) -> TrainResult:
        if self.config.use_subprocess:
            return self._run_subprocess()
        self.log("Starting LM training...")
        return self._train_lm()

    def _build_model(self, vocab_size: int):
        hidden = self.config.hidden_dim
        heads = max(1, self.config.num_heads)
        hidden = (hidden // heads) * heads

        self.model = HMTLanguageModel(
            vocab_size=vocab_size,
            dim=hidden,
            num_layers=self.config.num_layers,
            num_heads=heads,
            n_kv_heads=self.config.num_kv_heads,
            max_seq=self.config.seq_len,
            dropout=0.1,
            use_gradient_checkpointing=self.config.use_gradient_checkpointing,
        )

    def _load_checkpoint(self, path: str):
        """Load state dict and tokenizer from a checkpoint for resume training."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        saved_cfg = ckpt.get("config", ckpt.get("model_config", {}))

        # Validate architecture compatibility
        for key, label in [("hidden_dim", "dim"), ("num_layers", "layers"),
                           ("num_heads", "heads"), ("vocab_size", "vocab")]:
            saved_val = saved_cfg.get(key, saved_cfg.get(label, 0))
            current_val = getattr(self.config, key, 0) or self.config.__dict__.get(key, 0)
            if saved_val and current_val and saved_val != current_val:
                raise TrainingRuntimeError(
                    "ARCH_MISMATCH",
                    f"Checkpoint has {key}={saved_val}, "
                    f"config has {key}={current_val}. Must match for resume."
                )

        self.model.load_state_dict(ckpt["model_state_dict"])

        # Load tokenizer from checkpoint directory
        base = Path(path)
        tok_path = base.with_suffix(".tokenizer.json")
        if tok_path.exists():
            self.tokenizer = AdvancedTokenizer.load(str(tok_path))
            self.log(f"Loaded tokenizer from {tok_path.name} ({self.tokenizer.vocab_size_real} vocab)")
        else:
            self.log("No tokenizer found alongside checkpoint — will train new one", "warn")

        # Freeze layers if configured
        if self.config.resume_freeze_layers > 0:
            frozen = 0
            for layer in self.model.decoder.layers[:self.config.resume_freeze_layers]:
                for p in layer.parameters():
                    p.requires_grad_(False)
                    frozen += 1
            self.log(f"Frozen {frozen} params in first {self.config.resume_freeze_layers} layers")

        if self.config.resume_freeze_embeddings:
            for p in self.model.embed.parameters():
                p.requires_grad_(False)
            self.log("Embedding layer frozen")

        # Record resume
        self._resume_count = saved_cfg.get("_resume_count", 0) + 1
        self._saved_files = saved_cfg.get("_files_trained_on", [])
        self.log(f"Resumed from {Path(path).name} (resume #{self._resume_count})")

    def _load_data(self):
        return build_text_loaders(
            self.files, seq_len=self.config.seq_len, batch_size=self.config.batch_size,
            prompt_ratio=self.config.prompt_ratio,
            template_name=self.config.template_name,
            template_fields=self.config.template_fields)

    def _ram_report(self) -> str:
        snap = self._mem_monitor.snapshot if self._mem_monitor else None
        if snap:
            return f"ram={snap.process_mb:.0f}/{snap.available_mb:.0f}MB"
        return f"ram={get_process_mb():.0f}MB avail={get_available_mb():.0f}MB"

    def _train_lm(self) -> TrainResult:
        self.log("Loading data...")
        if not self.files:
            self.log("No files provided", "err")
            return None

        self._setup_threads()
        self.train_loader, self.val_loader, self.tokenizer, self.data_info = self._load_data()

        self.log(f"Data: vocab={self.data_info['vocab_size']}, windows={self.data_info.get('train_windows', 0)}")

        self._build_model(self.data_info["vocab_size"])

        # Resume checkpoint if requested
        if self.config.resume_path:
            self._load_checkpoint(self.config.resume_path)

        param_count = sum(p.numel() for p in self.model.parameters())
        self._setup_device()
        self.model = self.model.to(self.device)
        self.log(f"Model: {param_count:,} params, dim={self.config.hidden_dim}, layers={self.config.num_layers}")

        self._setup_memory_monitor(self.config.batch_size)

        n_batches = len(self.train_loader)
        steps_per_epoch = max(1, n_batches // self.config.grad_accum_steps)
        total_steps = self.config.epochs * steps_per_epoch
        self._create_optimizer_and_scheduler(steps_per_epoch)
        self._init_ema()

        self.log(f"Training: {self.config.epochs} epochs, {total_steps} steps, "
                 f"batch={self.config.batch_size}, accum={self.config.grad_accum_steps}")

        global_step = 0
        self.optimizer.zero_grad()
        effective_batch = self.config.batch_size

        for epoch in range(1, self.config.epochs + 1):
            if self.stop_flag and self.stop_flag.is_set():
                self.log("Training stopped", "warn")
                break

            epoch_loss = 0
            step = 0
            self.model.train()

            if self._mem_monitor:
                recommended = self._mem_monitor.recommended_batch
                if recommended != effective_batch:
                    effective_batch = recommended
                    self.log(f"Adaptive batch: {effective_batch} ({self._ram_report()})")

            for batch in self.train_loader:
                if self.stop_flag and self.stop_flag.is_set():
                    break
                if step >= steps_per_epoch:
                    break

                xb, yb = batch
                xb, yb = move_batch((xb, yb), self.device)

                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        logits = self.model(xb)
                        loss = self._criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
                    self.scaler.scale(loss).backward()
                else:
                    logits = self.model(xb)
                    loss = self._criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
                    loss.backward()

                epoch_loss += loss.item()

                if (step + 1) % self.config.grad_accum_steps == 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                    self._update_ema()
                    if self.scheduler:
                        self.scheduler.step()
                    global_step += 1

                step += 1

                if step % max(1, steps_per_epoch // 10) == 0 or step == steps_per_epoch:
                    avg_loss = epoch_loss / max(step, 1)
                    elapsed = time.time() - self.start_time
                    eta = int((elapsed / max(global_step, 1)) * max(total_steps - global_step, 1))
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    self.progress(
                        epoch=epoch, epochs=self.config.epochs,
                        loss=avg_loss, lr=current_lr,
                        eta=f"{eta//60}m {eta%60}s",
                        pct=(global_step / total_steps) * 100 if total_steps > 0 else 0,
                        ram_info=self._ram_report(),
                    )

            if step % self.config.grad_accum_steps != 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self._update_ema()
                if self.scheduler:
                    self.scheduler.step()
                global_step += 1

            avg_loss = epoch_loss / max(step, 1)
            self.metrics["train_loss"].append(avg_loss)
            self.metrics["lr"].append(self.optimizer.param_groups[0]["lr"])

            if self.val_loader:
                val_loss = lm_val_loss(self.model, self.val_loader, device=self.device, max_batches=50)
                self.metrics["val_loss"].append(val_loss)
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                self.log(f"Epoch {epoch}/{self.config.epochs} | loss={avg_loss:.4f} | val={val_loss:.4f} | {self._ram_report()}")
            else:
                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    self.best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                self.log(f"Epoch {epoch}/{self.config.epochs} | loss={avg_loss:.4f} | {self._ram_report()}")

            gc.collect()

        if self._mem_monitor:
            self._mem_monitor.stop()

        self.log("Training complete!")
        return TrainResult(
            model=self.model,
            ema_model=self.ema_model,
            best_state=self.best_state,
            metrics=self.metrics,
            info=self.data_info,
            tokenizer=self.tokenizer,
            optimizer_state=self.optimizer.state_dict() if self.optimizer else None,
        )

    def _run_subprocess(self) -> TrainResult:
        self.log("Starting training in subprocess...")
        queue: "mp.Queue" = mp.Queue()
        barrier = mp.Event()
        result_holder: Dict = {}

        def _watch_progress():
            while not barrier.is_set() or not queue.empty():
                try:
                    msg = queue.get(timeout=0.2)
                except Exception:
                    continue
                if msg is None:
                    break
                typ = msg.get("type")
                if typ == "progress":
                    self.progress(**msg.get("data", {}))
                elif typ == "log":
                    self.log(msg.get("text", ""), msg.get("level", "info"))
                elif typ == "result":
                    result_holder["path"] = msg.get("path")

        proc = mp.Process(
            target=_train_subprocess_main,
            args=(self.config, self.files, queue),
            daemon=True,
        )
        proc.start()

        _watch_progress()
        proc.join(timeout=60)

        result_path = result_holder.get("path", "")
        if result_path and Path(result_path).exists():
            from core.text_model import load_lm
            model, tokenizer = load_lm(result_path, device="cpu")
            ckpt = torch.load(result_path, map_location="cpu", weights_only=False)
            metrics = ckpt.get("_train_metrics", self.metrics)
            self.log(f"Subprocess training complete: {result_path}")
            return TrainResult(
                model=model,
                ema_model=None,
                best_state=ckpt.get("model_state_dict", {}),
                metrics=metrics,
                info=ckpt.get("data_info", {}),
                tokenizer=tokenizer,
            )

        self.log("Subprocess training failed or no result found", "err")
        return None


def _train_subprocess_main(config: TrainConfig, files: List[str], queue: "mp.Queue"):
    try:
        if config.cpu_threads > 0:
            torch.set_num_threads(config.cpu_threads)
        trainer = UnifiedTrainer(config, files,
                                 progress_callback=lambda **kw: queue.put({"type": "progress", "data": kw}),
                                 log_callback=lambda msg, lvl="info": queue.put({"type": "log", "text": msg, "level": lvl}))
        result = trainer.run()
        if result:
            checkpoint_path = f"trained_models/subprocess_{int(time.time())}.pt"
            save_lm(result.model, result.tokenizer,
                    config={"config": config.__dict__, **result.info},
                    path=checkpoint_path)
            queue.put({"type": "result", "path": checkpoint_path})
    except Exception as e:
        queue.put({"type": "log", "text": f"Subprocess error: {e}", "level": "err"})
    finally:
        queue.put(None)


def train_model(config: TrainConfig, files: List[str],
                progress_cb: Optional[Callable] = None,
                log_cb: Optional[Callable] = None,
                stop_flag=None) -> TrainResult:
    trainer = UnifiedTrainer(config, files, progress_cb, log_cb)
    if stop_flag:
        trainer.set_stop_flag(stop_flag)
    return trainer.run()
