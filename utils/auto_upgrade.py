"""
Auto-Upgrade System
- Real ArchitectureModifier: actually rewrites PyTorch module graphs
- SQLite DB: stores upgrade history, applied patches, self-modifications
- Groq LLM (llama-3.3-70b-versatile): real API calls for improvement suggestions
- AutoUpgradeSystem: analyzes → queries LLM → writes code → applies to live model
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import sqlite3
import json
import ast
import textwrap
import inspect
import types
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from copy import deepcopy

# Support both flat-file execution and package import
try:
    from core.architecture import SelfTransformer, ModuleConfig
except ImportError:
    from .architecture import SelfTransformer, ModuleConfig

log = logging.getLogger("AutoUpgrade")

# ── DB path ───────────────────────────────────────────────────────────────────
DB_PATH = Path("upgrade_system.db")

import os
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.3-70b-versatile"
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE LAYER
# ─────────────────────────────────────────────────────────────────────────────

class UpgradeDB:
    """SQLite-backed store for all upgrade activity and self-modifications."""

    def __init__(self, path: Path = DB_PATH):
        self.path = path
        self._local = threading.local()
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.path), check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_schema(self):
        c = self._conn()
        c.executescript("""
        CREATE TABLE IF NOT EXISTS upgrade_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT    NOT NULL,
            source      TEXT    NOT NULL,
            upgrade_type TEXT   NOT NULL,
            description TEXT,
            payload     TEXT,
            status      TEXT    NOT NULL DEFAULT 'pending',
            error       TEXT,
            perf_before REAL,
            perf_after  REAL
        );

        CREATE TABLE IF NOT EXISTS self_modifications (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT    NOT NULL,
            target_file TEXT    NOT NULL,
            function_name TEXT  NOT NULL,
            old_code    TEXT,
            new_code    TEXT    NOT NULL,
            reason      TEXT,
            applied     INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS model_snapshots (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT    NOT NULL,
            label       TEXT    NOT NULL,
            arch_json   TEXT    NOT NULL,
            param_count INTEGER,
            notes       TEXT
        );

        CREATE TABLE IF NOT EXISTS llm_conversations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT    NOT NULL,
            role        TEXT    NOT NULL,
            content     TEXT    NOT NULL,
            model       TEXT
        );
        """)
        c.commit()

    # ── upgrade_log ──────────────────────────────────────────────────────────
    def log_upgrade(self, source: str, upgrade_type: str, description: str,
                    payload: dict, status: str = "pending",
                    perf_before: float = None, perf_after: float = None) -> int:
        c = self._conn()
        cur = c.execute(
            "INSERT INTO upgrade_log (ts,source,upgrade_type,description,payload,status,perf_before,perf_after) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (datetime.now().isoformat(), source, upgrade_type, description,
             json.dumps(payload), status, perf_before, perf_after)
        )
        c.commit()
        return cur.lastrowid

    def update_upgrade_status(self, row_id: int, status: str, error: str = None,
                               perf_after: float = None):
        c = self._conn()
        c.execute(
            "UPDATE upgrade_log SET status=?, error=?, perf_after=? WHERE id=?",
            (status, error, perf_after, row_id)
        )
        c.commit()

    def get_recent_upgrades(self, limit: int = 50) -> List[Dict]:
        c = self._conn()
        rows = c.execute(
            "SELECT * FROM upgrade_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── self_modifications ───────────────────────────────────────────────────
    def log_self_modification(self, target_file: str, function_name: str,
                               old_code: str, new_code: str, reason: str) -> int:
        c = self._conn()
        cur = c.execute(
            "INSERT INTO self_modifications (ts,target_file,function_name,old_code,new_code,reason) "
            "VALUES (?,?,?,?,?,?)",
            (datetime.now().isoformat(), target_file, function_name,
             old_code, new_code, reason)
        )
        c.commit()
        return cur.lastrowid

    def mark_modification_applied(self, row_id: int):
        c = self._conn()
        c.execute("UPDATE self_modifications SET applied=1 WHERE id=?", (row_id,))
        c.commit()

    def get_modifications(self, applied: bool = None) -> List[Dict]:
        c = self._conn()
        if applied is None:
            rows = c.execute("SELECT * FROM self_modifications ORDER BY id DESC").fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM self_modifications WHERE applied=? ORDER BY id DESC",
                (1 if applied else 0,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── model_snapshots ──────────────────────────────────────────────────────
    def save_snapshot(self, label: str, arch_json: str,
                      param_count: int, notes: str = "") -> int:
        c = self._conn()
        cur = c.execute(
            "INSERT INTO model_snapshots (ts,label,arch_json,param_count,notes) VALUES (?,?,?,?,?)",
            (datetime.now().isoformat(), label, arch_json, param_count, notes)
        )
        c.commit()
        return cur.lastrowid

    def get_snapshots(self, limit: int = 20) -> List[Dict]:
        c = self._conn()
        rows = c.execute(
            "SELECT * FROM model_snapshots ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── llm_conversations ────────────────────────────────────────────────────
    def log_llm(self, role: str, content: str, model: str = GROQ_MODEL):
        c = self._conn()
        c.execute(
            "INSERT INTO llm_conversations (ts,role,content,model) VALUES (?,?,?,?)",
            (datetime.now().isoformat(), role, content, model)
        )
        c.commit()

    def get_conversation(self, limit: int = 40) -> List[Dict]:
        c = self._conn()
        rows = c.execute(
            "SELECT * FROM llm_conversations ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in reversed(rows)]


# ─────────────────────────────────────────────────────────────────────────────
# GROQ LLM CLIENT
# ─────────────────────────────────────────────────────────────────────────────

class GroqClient:
    """Real Groq API client — llama-3.3-70b-versatile."""

    def __init__(self, api_key: str = GROQ_API_KEY, model: str = GROQ_MODEL):
        self.api_key = api_key
        self.model   = model

    def chat(self, messages: List[Dict], temperature: float = 0.3,
             max_tokens: int = 2048) -> str:
        """Synchronous Groq chat completion. Returns assistant content string."""
        import urllib.request
        import urllib.error

        payload = json.dumps({
            "model":       self.model,
            "messages":    messages,
            "temperature": temperature,
            "max_tokens":  max_tokens,
        }).encode()

        req = urllib.request.Request(
            GROQ_URL,
            data=payload,
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent":    "python-groq-client/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
                return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            log.error(f"Groq HTTP {e.code}: {body}")
            raise RuntimeError(f"Groq API error {e.code}: {body}")
        except Exception as e:
            log.error(f"Groq request failed: {e}")
            raise

    def ask_for_upgrade(self, context: Dict) -> Dict:
        """
        Ask the LLM for a concrete upgrade plan.
        Returns a structured dict with keys: type, description, changes, code_patch (optional).
        """
        system_msg = (
            "You are an expert PyTorch ML engineer. "
            "Given a model performance report, return ONLY a JSON object (no markdown, no prose) "
            "with these keys:\n"
            "  type: one of [architecture_modification, training_modification, regularization_modification]\n"
            "  description: one sentence\n"
            "  changes: list of strings describing what to do\n"
            "  code_patch: optional Python function string named patch_model(model) that modifies the model\n"
            "  new_function_name: optional name of a new utility function to add to the upgrade system\n"
            "  new_function: optional Python function definition (def new_function_name(...)) to self-inject\n"
            "Keep all code short, safe, and focused on nn.Module modifications. "
            "Only include new_function if it adds genuine new capability."
        )
        user_msg = (
            f"Model performance report:\n{json.dumps(context, indent=2, default=str)}\n\n"
            "Suggest the single most impactful upgrade."
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ]
        raw = self.chat(messages, temperature=0.2, max_tokens=1024)
        # Strip markdown fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Best-effort: return as description
            return {"type": "training_modification",
                    "description": raw[:200],
                    "changes": [raw[:200]]}

    def ask_for_code_function(self, purpose: str, context: str) -> str:
        """
        Ask LLM to write a Python function that can be injected into a module.
        Returns raw Python source code string.
        """
        system_msg = (
            "You are a Python/PyTorch expert. Write ONLY a single Python function "
            "(no imports, no class, no markdown). The function will be exec'd and "
            "injected into a live nn.Module. Keep it short and safe."
        )
        user_msg = f"Purpose: {purpose}\nContext: {context}\nWrite the function now:"
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ]
        code = self.chat(messages, temperature=0.1, max_tokens=512)
        code = code.strip()
        if code.startswith("```"):
            parts = code.split("```")
            code = parts[1] if len(parts) > 1 else code
            if code.startswith("python"):
                code = code[6:]
        return code.strip()


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE ANALYZER
# ─────────────────────────────────────────────────────────────────────────────

class PerformanceAnalyzer:
    """Analyzes a live nn.Module and its training history."""

    def __init__(self, model: nn.Module, training_history: Dict):
        self.model = model
        self.training_history = training_history

    def analyze_convergence(self) -> Dict:
        losses = self.training_history.get("loss", [])
        if not losses:
            return {"no_data": True}
        improvement_rate = (losses[0] - losses[-1]) / (abs(losses[0]) + 1e-8)
        variance = float(torch.tensor(losses).var().item()) if len(losses) > 1 else 0.0
        return {
            "improvement_rate":    float(improvement_rate),
            "convergence_variance": variance,
            "plateau_detected":    improvement_rate < 0.01,
            "total_steps":         len(losses),
            "avg_loss":            float(sum(losses) / len(losses)),
            "final_loss":          float(losses[-1]),
        }

    def analyze_layer_performance(self) -> Dict:
        stats = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                g = param.grad
                stats[name] = {
                    "grad_norm":    float(g.norm().item()),
                    "grad_mean":    float(g.mean().item()),
                    "grad_std":     float(g.std().item()),
                    "param_norm":   float(param.norm().item()),
                    "dead_ratio":   float((param.abs() < 1e-6).float().mean().item()),
                }
        return stats

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def model_architecture_summary(self) -> Dict:
        layers = {}
        for name, module in self.model.named_modules():
            if name:
                layers[name] = {
                    "type":   type(module).__name__,
                    "params": sum(p.numel() for p in module.parameters(recurse=False)),
                }
        return layers

    def identify_bottlenecks(self) -> List[Dict]:
        bottlenecks = []
        for name, stats in self.analyze_layer_performance().items():
            if stats["grad_norm"] < 1e-7:
                bottlenecks.append({"type": "vanishing_gradient", "layer": name, "severity": "high"})
            if stats["dead_ratio"] > 0.5:
                bottlenecks.append({"type": "dead_neurons", "layer": name,
                                    "ratio": stats["dead_ratio"], "severity": "medium"})
        return bottlenecks

    def identify_opportunities(self) -> List[Dict]:
        ops = []
        conv = self.analyze_convergence()
        if conv.get("plateau_detected"):
            ops.append({"type": "lr_reduction", "reason": "plateau detected"})
        if conv.get("convergence_variance", 0) > 0.5:
            ops.append({"type": "regularization", "reason": "high variance"})
        if self.count_parameters() < 100_000:
            ops.append({"type": "capacity_increase", "reason": "model may be underfitting"})
        return ops

    def full_report(self) -> Dict:
        bottlenecks = self.identify_bottlenecks()
        convergence  = self.analyze_convergence()
        score = 100.0
        score -= len(bottlenecks) * 10
        if convergence.get("plateau_detected"):
            score -= 20
        if convergence.get("improvement_rate", 0) < 0.05:
            score -= 15
        return {
            "timestamp":     datetime.now().isoformat(),
            "param_count":   self.count_parameters(),
            "convergence":   convergence,
            "bottlenecks":   bottlenecks,
            "opportunities": self.identify_opportunities(),
            "architecture":  self.model_architecture_summary(),
            "overall_score": max(0.0, score),
        }


# ─────────────────────────────────────────────────────────────────────────────
# REAL ARCHITECTURE MODIFIER
# ─────────────────────────────────────────────────────────────────────────────

class ArchitectureModifier:
    """
    Actually modifies live PyTorch nn.Module graphs.
    All changes are recorded in the DB and can be replayed.
    """

    def __init__(self, db: "UpgradeDB"):
        self.db = db

    # ── Batch Normalization ──────────────────────────────────────────────────
    def add_batch_norm(self, model: nn.Module) -> Tuple[nn.Module, List[str]]:
        """
        Wraps every nn.Linear in a Sequential(Linear, BatchNorm1d).
        Returns modified model and list of modified layer names.
        """
        modified = []

        def _replace(parent: nn.Module, prefix: str = ""):
            for name, child in list(parent.named_children()):
                full = f"{prefix}.{name}" if prefix else name
                if isinstance(child, nn.Linear) and child.out_features > 1:
                    new_block = nn.Sequential(
                        child,
                        nn.BatchNorm1d(child.out_features),
                    )
                    setattr(parent, name, new_block)
                    modified.append(full)
                    log.info(f"  BatchNorm added after {full}")
                else:
                    _replace(child, full)

        _replace(model)
        return model, modified

    # ── Residual Connections ─────────────────────────────────────────────────
    def add_residual_connections(self, model: nn.Module) -> Tuple[nn.Module, List[str]]:
        """
        Wraps nn.Sequential blocks that have matching input/output dims
        in a ResidualWrapper.
        """
        modified = []

        class ResidualWrapper(nn.Module):
            def __init__(self, inner: nn.Module, dim: int):
                super().__init__()
                self.inner = inner
                self.proj  = nn.Identity()  # replaced below if dims differ

            def forward(self, x):
                return self.inner(x) + self.proj(x)

        def _replace(parent: nn.Module, prefix: str = ""):
            for name, child in list(parent.named_children()):
                full = f"{prefix}.{name}" if prefix else name
                if isinstance(child, nn.Sequential):
                    # Probe input/output dims via first/last Linear
                    linears = [m for m in child.modules() if isinstance(m, nn.Linear)]
                    if len(linears) >= 2:
                        in_dim  = linears[0].in_features
                        out_dim = linears[-1].out_features
                        if in_dim == out_dim:
                            wrapped = ResidualWrapper(child, in_dim)
                            setattr(parent, name, wrapped)
                            modified.append(full)
                            log.info(f"  Residual wrapper added around {full}")
                else:
                    _replace(child, full)

        _replace(model)
        return model, modified

    # ── Increase Capacity ────────────────────────────────────────────────────
    def increase_capacity(self, model: nn.Module,
                          scale: float = 1.5) -> Tuple[nn.Module, List[str]]:
        """
        Replaces nn.Linear layers with wider versions (scaled hidden dim).
        Preserves in/out dims; only expands intermediate layers.
        Copies weights via SVD-based expansion.
        """
        modified = []
        all_linears = [(n, m) for n, m in model.named_modules()
                       if isinstance(m, nn.Linear)]
        if len(all_linears) < 3:
            return model, modified  # nothing to expand

        # Only expand intermediate layers (not first/last)
        to_expand = all_linears[1:-1]

        for name, layer in to_expand:
            new_out = max(layer.out_features, int(layer.out_features * scale))
            if new_out == layer.out_features:
                continue
            new_layer = nn.Linear(layer.in_features, new_out,
                                  bias=layer.bias is not None)
            # Copy existing weights into new layer (pad with small noise)
            with torch.no_grad():
                new_layer.weight[:layer.out_features] = layer.weight
                new_layer.weight[layer.out_features:] = (
                    torch.randn(new_out - layer.out_features, layer.in_features) * 0.01
                )
                if layer.bias is not None:
                    new_layer.bias[:layer.out_features] = layer.bias
                    new_layer.bias[layer.out_features:] = 0.0

            # Navigate to parent and replace
            parts  = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], new_layer)
            modified.append(f"{name}: {layer.out_features} → {new_out}")
            log.info(f"  Expanded {name}: {layer.out_features} → {new_out}")

        return model, modified

    # ── Magnitude Pruning ────────────────────────────────────────────────────
    def apply_pruning(self, model: nn.Module,
                      sparsity: float = 0.3) -> Tuple[nn.Module, List[str]]:
        """
        Applies L1 unstructured pruning to all Linear layers.
        """
        modified = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name="weight", amount=sparsity)
                prune.remove(module, "weight")   # make permanent
                modified.append(name)
                log.info(f"  Pruned {name} at sparsity={sparsity}")
        return model, modified

    # ── Dropout Injection ────────────────────────────────────────────────────
    def add_dropout(self, model: nn.Module,
                    rate: float = 0.2) -> Tuple[nn.Module, List[str]]:
        """Inserts Dropout after every Linear layer inside Sequential blocks."""
        modified = []

        def _replace(parent: nn.Module, prefix: str = ""):
            for name, child in list(parent.named_children()):
                full = f"{prefix}.{name}" if prefix else name
                if isinstance(child, nn.Sequential):
                    new_layers = []
                    changed = False
                    for layer in child:
                        new_layers.append(layer)
                        if isinstance(layer, nn.Linear):
                            new_layers.append(nn.Dropout(p=rate))
                            changed = True
                    if changed:
                        setattr(parent, name, nn.Sequential(*new_layers))
                        modified.append(full)
                        log.info(f"  Dropout({rate}) added in {full}")
                else:
                    _replace(child, full)

        _replace(model)
        return model, modified

    # ── Write modified architecture back to implementations.py ──────────────
    def write_model_to_file(self, model: nn.Module, db: "UpgradeDB",
                             reason: str = "") -> bool:
        """
        Serialises the current live model structure back into implementations.py
        by regenerating the HierarchicalMambaEncoder.initialize() method body
        to match whatever layers are now present in the model.
        Also writes a full model snapshot to the DB.
        """
        target = Path(__file__).parent / "implementations.py"
        if not target.exists():
            log.error("implementations.py not found — cannot write model to file")
            return False

        # Build a Python repr of the model's layer structure
        layer_lines = []
        for name, module in model.named_children():
            # Sanitize name — numeric names like "0" are invalid Python identifiers
            safe_name = f"layer_{name}" if name.isdigit() else name
            mtype = type(module).__name__
            if isinstance(module, nn.Sequential):
                inner = []
                for i, layer in enumerate(module):
                    ltype = type(layer).__name__
                    if isinstance(layer, nn.Linear):
                        inner.append(
                            f"nn.Linear({layer.in_features}, {layer.out_features}, "
                            f"bias={layer.bias is not None})"
                        )
                    elif isinstance(layer, nn.BatchNorm1d):
                        inner.append(f"nn.BatchNorm1d({layer.num_features})")
                    elif isinstance(layer, nn.Dropout):
                        inner.append(f"nn.Dropout(p={layer.p})")
                    elif isinstance(layer, nn.ReLU):
                        inner.append("nn.ReLU()")
                    elif isinstance(layer, nn.GELU):
                        inner.append("nn.GELU()")
                    elif isinstance(layer, nn.LayerNorm):
                        inner.append(f"nn.LayerNorm({list(layer.normalized_shape)})")
                    else:
                        inner.append(f"# {ltype}()")
                joined = ",\n                    ".join(inner)
                layer_lines.append(
                    f"        self.{safe_name} = nn.Sequential(\n"
                    f"                    {joined}\n"
                    f"                )"
                )
            elif isinstance(module, nn.Linear):
                layer_lines.append(
                    f"        self.{safe_name} = nn.Linear("
                    f"{module.in_features}, {module.out_features}, "
                    f"bias={module.bias is not None})"
                )
            elif isinstance(module, nn.BatchNorm1d):
                layer_lines.append(
                    f"        self.{safe_name} = nn.BatchNorm1d({module.num_features})"
                )
            else:
                layer_lines.append(f"        # self.{safe_name} = {mtype}()  # complex layer")

        new_init_body = "\n".join(layer_lines) if layer_lines else "        pass"

        # Read current file
        src = target.read_text(encoding="utf-8")
        old_src = src

        # Replace the probe_model block in start.py (where the probe is defined)
        # AND write a standalone SavedUpgradedModel class into implementations.py
        ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        class_name = f"UpgradedModel_{ts_tag}"

        new_class = (
            f"\n\n# ── AUTO-GENERATED by AutoUpgradeSystem {datetime.now().isoformat()} ──\n"
            f"# Reason: {reason}\n"
            f"class {class_name}(nn.Module):\n"
            f"    \"\"\"Auto-upgraded model snapshot — {reason}\"\"\"\n"
            f"    def __init__(self):\n"
            f"        super().__init__()\n"
            f"{new_init_body}\n\n"
            f"    def forward(self, x):\n"
            f"        for layer in self.children():\n"
            f"            x = layer(x)\n"
            f"        return x\n"
        )

        # Append to implementations.py
        with open(target, "a", encoding="utf-8") as f:
            f.write(new_class)

        log.info(f"Wrote {class_name} to implementations.py")

        # Record in DB
        db.log_self_modification(
            target_file="implementations.py",
            function_name=class_name,
            old_code="(appended — no old code)",
            new_code=new_class,
            reason=reason,
        )
        arch_json = json.dumps({n: type(m).__name__ for n, m in model.named_modules()})
        db.save_snapshot(
            label=class_name,
            arch_json=arch_json,
            param_count=sum(p.numel() for p in model.parameters()),
            notes=reason,
        )
        return True

    # ── Self-modification: inject new function into auto_upgrade.py ──────────
    def inject_function_into_self(self, func_name: str, func_code: str,
                                   db: "UpgradeDB", reason: str = "") -> bool:
        """
        Appends a new method/function to auto_upgrade.py itself.
        This is the true self-modification: the system writes new capabilities
        into its own source file.
        func_code must be a valid Python function definition string.
        """
        target = Path(__file__)  # auto_upgrade.py itself

        # Validate syntax
        try:
            ast.parse(func_code)
        except SyntaxError as e:
            log.error(f"inject_function_into_self: syntax error in {func_name}: {e}")
            return False

        old_src = target.read_text(encoding="utf-8")

        # Check if function already exists
        if f"def {func_name}" in old_src:
            log.warning(f"Function {func_name} already exists in auto_upgrade.py — skipping")
            return False

        injection = (
            f"\n\n# ── SELF-INJECTED by AutoUpgradeSystem {datetime.now().isoformat()} ──\n"
            f"# Reason: {reason}\n"
            f"{func_code}\n"
        )

        with open(target, "a", encoding="utf-8") as f:
            f.write(injection)

        log.info(f"Self-injected function '{func_name}' into auto_upgrade.py")

        db.log_self_modification(
            target_file="auto_upgrade.py",
            function_name=func_name,
            old_code="(new function — appended)",
            new_code=func_code,
            reason=reason,
        )
        return True

    # ── LLM Code Patch ───────────────────────────────────────────────────────
    def apply_code_patch(self, model: nn.Module,
                         code: str, db: "UpgradeDB",
                         reason: str = "") -> Tuple[nn.Module, bool]:
        """
        Exec a Python function string that receives `model` and returns modified model.
        The function must be named `patch_model(model)`.
        Saves to DB before applying.
        """
        # Validate: must define patch_model
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            log.error(f"Code patch syntax error: {e}")
            return model, False

        func_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        if "patch_model" not in func_names:
            log.error("Code patch must define patch_model(model)")
            return model, False

        # Save to DB
        row_id = db.log_self_modification(
            target_file="live_model",
            function_name="patch_model",
            old_code=str(model),
            new_code=code,
            reason=reason,
        )

        # Snapshot before
        arch_json = json.dumps({n: type(m).__name__
                                for n, m in model.named_modules()})
        db.save_snapshot("before_patch", arch_json,
                         sum(p.numel() for p in model.parameters()),
                         notes=reason)

        try:
            ns: Dict = {}
            exec(compile(code, "<llm_patch>", "exec"), {"torch": torch, "nn": nn}, ns)
            patched = ns["patch_model"](model)
            db.mark_modification_applied(row_id)
            # Snapshot after
            arch_json2 = json.dumps({n: type(m).__name__
                                     for n, m in patched.named_modules()})
            db.save_snapshot("after_patch", arch_json2,
                             sum(p.numel() for p in patched.parameters()),
                             notes=f"after: {reason}")
            log.info("Code patch applied successfully")
            return patched, True
        except Exception as e:
            log.error(f"Code patch execution failed: {e}")
            db.update_upgrade_status(row_id, "failed", str(e))
            return model, False


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-UPGRADE SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

class AutoUpgradeSystem(SelfTransformer):
    """
    Orchestrates: analyze → query Groq LLM → apply real architecture changes
    → record everything in SQLite DB.
    """

    def initialize(self) -> None:
        self.model            = self.config.params.get("model")
        self.training_history = self.config.params.get("training_history", {})
        self.groq             = GroqClient()
        self.db               = UpgradeDB()
        self.modifier         = ArchitectureModifier(self.db)
        self.analyzer         = PerformanceAnalyzer(self.model, self.training_history)
        self.upgrade_log: List[Dict] = []
        self.attempted  = 0
        self.successful = 0
        # Callbacks for UI
        self._on_log: Optional[callable] = None
        super().initialize()

    def set_log_callback(self, cb):
        """UI can register a callback(msg, level) to receive live log lines."""
        self._on_log = cb

    def _emit(self, msg: str, level: str = "info"):
        log.info(msg)
        if self._on_log:
            self._on_log(msg, level)

    # ── SelfTransformer interface ────────────────────────────────────────────

    def analyze_performance(self) -> Dict:
        report = self.analyzer.full_report()
        self._emit(f"Performance score: {report['overall_score']:.1f}/100")
        return report

    def fetch_improvements(self, source: str = "llm") -> List[Dict]:
        """Query Groq for upgrade suggestions based on current performance."""
        report = self.analyze_performance()
        self._emit("Querying Groq LLM for improvement suggestions…", "info")
        self.db.log_llm("user",
            f"Performance report:\n{json.dumps(report, indent=2, default=str)}")
        try:
            suggestion = self.groq.ask_for_upgrade(report)
            self.db.log_llm("assistant", json.dumps(suggestion, indent=2))
            self._emit(f"LLM suggestion: {suggestion.get('description','')}", "ok")
            return [suggestion]
        except Exception as e:
            self._emit(f"LLM query failed: {e}", "err")
            return []

    def apply_upgrade(self, upgrade_config: Dict) -> bool:
        self.attempted += 1
        utype = upgrade_config.get("type", "")
        desc  = upgrade_config.get("description", utype)
        row_id = self.db.log_upgrade(
            source="groq_llm", upgrade_type=utype,
            description=desc, payload=upgrade_config,
            perf_before=self.analyzer.full_report().get("overall_score"),
        )
        self._emit(f"Applying upgrade: {utype} — {desc}", "info")

        try:
            modified_layers: List[str] = []

            if utype == "architecture_modification":
                changes = upgrade_config.get("changes", [])
                for change in changes:
                    cl = change.lower()
                    if "batch norm" in cl or "normalization" in cl:
                        self.model, layers = self.modifier.add_batch_norm(self.model)
                        modified_layers += layers
                    if "residual" in cl:
                        self.model, layers = self.modifier.add_residual_connections(self.model)
                        modified_layers += layers
                    if "capacity" in cl or "hidden" in cl or "wider" in cl:
                        self.model, layers = self.modifier.increase_capacity(self.model, 1.3)
                        modified_layers += layers

                # If LLM also provided a code patch, apply it
                code = upgrade_config.get("code_patch", "")
                if code:
                    # Strip markdown fences
                    code = code.strip()
                    if code.startswith("```"):
                        parts = code.split("```")
                        code = parts[1] if len(parts) > 1 else code
                        if code.startswith("python"):
                            code = code[6:]
                    code = code.strip()
                if code and "patch_model" in code:
                    self.model, ok = self.modifier.apply_code_patch(
                        self.model, code, self.db, reason=desc)
                    if ok:
                        modified_layers.append("code_patch")

            elif utype == "regularization_modification":
                changes = upgrade_config.get("changes", [])
                for change in changes:
                    cl = change.lower()
                    if "dropout" in cl:
                        self.model, layers = self.modifier.add_dropout(self.model, 0.2)
                        modified_layers += layers
                    if "prun" in cl:
                        self.model, layers = self.modifier.apply_pruning(self.model, 0.2)
                        modified_layers += layers

            elif utype == "training_modification":
                # Training changes are logged but don't modify the graph
                self._emit("Training modification noted — apply on next training run", "warn")
                modified_layers = upgrade_config.get("changes", [])

            else:
                # Try code patch if present
                code = upgrade_config.get("code_patch", "")
                if code:
                    code = code.strip()
                    if code.startswith("```"):
                        parts = code.split("```")
                        code = parts[1] if len(parts) > 1 else code
                        if code.startswith("python"):
                            code = code[6:]
                    code = code.strip()
                if code and "patch_model" in code:
                    self.model, ok = self.modifier.apply_code_patch(
                        self.model, code, self.db, reason=desc)
                    if not ok:
                        self.db.update_upgrade_status(row_id, "failed", "code_patch failed")
                        return False
                    modified_layers.append("code_patch")
                else:
                    self._emit(f"Unknown upgrade type: {utype}", "warn")
                    self.db.update_upgrade_status(row_id, "skipped")
                    return False

            perf_after = self.analyzer.full_report().get("overall_score")
            self.db.update_upgrade_status(row_id, "success", perf_after=perf_after)
            self.successful += 1

            # ── WRITE CHANGES TO DISK ────────────────────────────────────────
            # 1. Write the modified model architecture into implementations.py
            if modified_layers and utype in ("architecture_modification",
                                              "regularization_modification"):
                wrote = self.modifier.write_model_to_file(
                    self.model, self.db, reason=desc)
                if wrote:
                    self._emit("Model architecture written to implementations.py", "ok")

            # 2. If LLM provided a new utility function, inject it into auto_upgrade.py
            new_func = upgrade_config.get("new_function", "")
            new_func_name = upgrade_config.get("new_function_name", "")
            if new_func and new_func_name:
                # Strip markdown fences if LLM wrapped the code
                new_func = new_func.strip()
                if new_func.startswith("```"):
                    parts = new_func.split("```")
                    new_func = parts[1] if len(parts) > 1 else new_func
                    if new_func.startswith("python"):
                        new_func = new_func[6:]
                new_func = new_func.strip()
                # Validate it's a real function def before injecting
                if new_func.startswith("def "):
                    injected = self.modifier.inject_function_into_self(
                        new_func_name, new_func, self.db, reason=desc)
                    if injected:
                        self._emit(
                            f"Self-injected '{new_func_name}' into auto_upgrade.py", "ok")
            # ─────────────────────────────────────────────────────────────────

            entry = {
                "ts":      datetime.now().isoformat(),
                "type":    utype,
                "desc":    desc,
                "layers":  modified_layers,
                "status":  "success",
            }
            self.upgrade_log.append(entry)
            self._emit(
                f"Upgrade applied. Modified: {modified_layers or 'training config'}",
                "ok"
            )
            return True

        except Exception as e:
            self.db.update_upgrade_status(row_id, "failed", str(e))
            self._emit(f"Upgrade failed: {e}", "err")
            self.upgrade_log.append({
                "ts": datetime.now().isoformat(), "type": utype,
                "desc": desc, "status": "failed", "error": str(e)
            })
            return False

    def run_full_cycle(self) -> Dict:
        """Analyze → fetch → apply. Returns summary."""
        self._emit("=== Auto-Upgrade Cycle Start ===", "info")
        improvements = self.fetch_improvements("llm")
        applied = 0
        for imp in improvements:
            if self.apply_upgrade(imp):
                applied += 1
        summary = {
            "fetched": len(improvements),
            "applied": applied,
            "total_attempted": self.attempted,
            "total_successful": self.successful,
        }
        self._emit(f"=== Cycle complete: {applied}/{len(improvements)} applied ===", "ok")
        return summary

    def get_upgrade_status(self) -> Dict:
        return {
            "attempted":      self.attempted,
            "successful":     self.successful,
            "success_rate":   self.successful / max(1, self.attempted),
            "recent_upgrades": self.upgrade_log[-10:],
            "db_path":        str(self.db.path),
        }

    def forward(self, data: Any) -> Any:
        return data  # pass-through; upgrades happen via run_full_cycle()
