"""
Smart Upgrade System - Full Project Analysis
- Analyzes ALL project files
- Single LLM call with complete context
- Verifies changes before applying
- Updates context when files change
- Minimizes API calls via caching
"""

import ast
import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import traceback

from project_context import ProjectFileDB, ProjectAnalyzer, GroqClientCached

log = logging.getLogger("SmartUpgrade")

PROJECT_ROOT = Path(__file__).parent


class CodeVerifier:
    """Verifies code correctness before applying."""

    @staticmethod
    def check_syntax(code: str) -> Tuple[bool, str]:
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"

    @staticmethod
    def check_imports(code: str) -> Tuple[bool, List[str]]:
        missing = []
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    try:
                        __import__(alias.name)
                    except ImportError:
                        missing.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    try:
                        __import__(node.module)
                    except ImportError:
                        missing.append(node.module)
        return len(missing) == 0, missing

    @staticmethod
    def check_structure(code: str, target_file: str) -> Tuple[bool, str]:
        target_name = Path(target_file).stem
        tree = ast.parse(code)
        has_changes = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if node.name == target_name or f"{target_name}_upgraded" in node.name:
                    has_changes = True
            elif isinstance(node, ast.FunctionDef):
                if not node.name.startswith('_'):
                    has_changes = True
        return has_changes, "Structure looks valid" if has_changes else "No significant changes found"


class SmartUpgradeSystem:
    """
    Complete project upgrade system with full context awareness.
    Single LLM call for multiple file improvements.
    """

    def __init__(self, project_root: Path = PROJECT_ROOT):
        self.project_root = project_root
        self.db = ProjectFileDB(PROJECT_ROOT / "project_context.db")
        self.analyzer = ProjectAnalyzer(project_root)
        self.groq = GroqClientCached()
        self.verifier = CodeVerifier()
        self._on_log: Optional[callable] = None
        self._suggestions: List[Dict] = []
        self._applied: List[Dict] = []

    def set_log_callback(self, cb):
        self._on_log = cb

    def _emit(self, msg: str, level: str = "info"):
        log.info(f"[{level}] {msg}")
        if self._on_log:
            self._on_log(msg, level)

    def analyze_project(self, force: bool = False) -> Dict:
        """Analyze all project files and store context."""
        self._emit("Analyzing project files...")

        files = self.analyzer.get_python_files()
        self._emit(f"Found {len(files)} Python files")

        current_mtimes = {str(f.relative_to(self.project_root)): f.stat().st_mtime 
                        for f in files}

        outdated = self.db.get_outdated_files(current_mtimes) if not force else []

        if outdated:
            self._emit(f"Re-analyzing {len(outdated)} outdated files")
        else:
            self._emit("All files up to date")

        contexts = []
        analyzed = 0

        for file_path in files:
            rel_path = str(file_path.relative_to(self.project_root))

            if not force and rel_path not in outdated:
                existing = self.db.get_file_context(rel_path)
                if existing:
                    contexts.append(existing)
                    continue

            info = self.analyzer.analyze_file(file_path)
            if info:
                self.db.save_file_context(
                    file_path=info['file_path'],
                    file_hash=info['file_hash'],
                    ast_summary=info['ast_summary'],
                    imports=info['imports'],
                    exports=info['exports'],
                    docstring=info['docstring'],
                )

                for dep in info.get('dependencies', []):
                    if dep['type'] in ('import', 'import_from'):
                        self.db.save_relationship(
                            source=info['file_path'],
                            target=dep['target'],
                            import_type=dep['type'],
                            line=dep['line']
                        )

                contexts.append(info)
                analyzed += 1

        self._emit(f"Analyzed {analyzed} files, {len(contexts)} total in context")
        return {
            'files_analyzed': analyzed,
            'total_files': len(contexts),
            'outdated': len(outdated),
        }

    def get_full_context(self) -> str:
        """Get complete project context for LLM."""
        contexts = self.db.get_all_contexts()
        return self.analyzer.generate_context_for_llm(contexts)

    def query_for_upgrades(self, focus_files: List[str] = None,
                          max_upgrades: int = 5) -> List[Dict]:
        """
        Single LLM call to get upgrade suggestions for multiple files.
        Uses caching to avoid duplicate API calls.
        """
        cache_key = f"upgrades_{focus_files}_{max_upgrades}" if focus_files else f"upgrades_all_{max_upgrades}"

        cached = self.db.get_llm_cache(cache_key)
        if cached:
            self._emit("Using cached upgrade suggestions")
            try:
                suggestions = json.loads(cached)
                return self._filter_duplicate_suggestions(suggestions)
            except json.JSONDecodeError:
                pass

        self._emit("Querying LLM for upgrades...")

        contexts = self.db.get_all_contexts()

        if focus_files:
            contexts = [c for c in contexts if c['file_path'] in focus_files]
        else:
            key_files = {
                'architecture.py', 'implementations.py', 'trainer.py', 
                'data_loader.py', 'inference.py', 'text_model.py',
                'text_dataset.py', 'image_dataset.py', 'chat.py', 'model_chat.py'
            }
            contexts = [c for c in contexts if c['file_path'] in key_files]
            if not contexts:
                contexts = self.db.get_all_contexts()[:10]

        project_context = self.analyzer.generate_context_for_llm(contexts, max_chars=6000)

        history = self.db.get_upgrade_history(limit=10)
        history_text = ""
        if history:
            history_text = "\n\nPrevious upgrades (avoid repeating failed ones):\n"
            for h in history[-5:]:
                history_text += f"- {h['file_path']}: {h['status']} - {h.get('llm_reasoning', '')[:100]}\n"

        system_msg = """You are an expert Python/ML engineer. Analyze the project context and suggest upgrades.

IMPORTANT RULES:
1. Only suggest changes where the new_code does NOT already exist in the project
2. new_code MUST be a COMPLETE, VALID Python code block with proper indentation
3. current_code MUST be an EXACT substring from the target file
4. Both current_code and new_code must parse as valid Python syntax

Return ONLY valid JSON array (no markdown, no prose):
[
  {
    "file": "filename.py",
    "function": "function_name or null",
    "issue": "brief description",
    "reasoning": "why this helps",
    "current_code": "EXACT lines to replace - must exist verbatim in file, include ALL indentation",
    "new_code": "COMPLETE replacement code - must be valid Python with proper indentation, no truncation",
    "verify": "what to check"
  }
]

GOOD current_code example:
    def train(self):
        loss = self.compute_loss()
        return loss

BAD current_code (incomplete):
    def train(self):

GOOD new_code example:
    def train(self):
        loss = self.compute_loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

BAD new_code (truncated/partial):
    def train(self):
        loss = self.compute_loss()

Focus on:
1. Bug fixes and error handling
2. Performance improvements
3. Code quality
4. Missing imports
5. Consistency across files

CRITICAL: Only suggest truly NEW improvements. If similar code exists, skip it."""

        user_msg = f"""Project context:
{project_context}
{history_text}

Suggest {max_upgrades} concrete, verifiable upgrades as JSON array."""

        try:
            response = self.groq.chat([
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ], use_cache=False, temperature=0.1, max_tokens=4096)

            raw = response.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            suggestions = json.loads(raw)

            self.db.save_llm_cache(cache_key, 
                hashlib.sha256(user_msg.encode()).hexdigest()[:16],
                json.dumps(suggestions), ttl_seconds=1800)

            filtered = self._filter_duplicate_suggestions(suggestions)
            self._suggestions = filtered
            self._emit(f"Got {len(filtered)} unique upgrade suggestions (filtered {len(suggestions) - len(filtered)} duplicates)")
            return filtered

        except json.JSONDecodeError as e:
            self._emit(f"Failed to parse LLM response: {e}", "error")
            return []
        except Exception as e:
            self._emit(f"LLM query failed: {e}", "error")
            return []

    def _filter_duplicate_suggestions(self, suggestions: List[Dict]) -> List[Dict]:
        """Filter out suggestions where new_code already exists or is invalid."""
        filtered = []
        for i, s in enumerate(suggestions):
            new_code = s.get('new_code', '')
            current_code = s.get('current_code', '')
            file_path = s.get('file', 'unknown')

            if not new_code:
                self._emit(f"[{i+1}] {file_path}: Skipping - no new_code", "warn")
                continue

            if not current_code:
                self._emit(f"[{i+1}] {file_path}: Skipping - no current_code", "warn")
                continue

            if len(new_code) < 30:
                self._emit(f"[{i+1}] {file_path}: Skipping - code too short ({len(new_code)} chars)", "warn")
                continue

            if self._code_exists_in_project(new_code):
                self._emit(f"[{i+1}] {file_path}: Skipping - already exists", "warn")
                continue

            syntax_ok, err = self.verifier.check_syntax(new_code)
            if not syntax_ok:
                self._emit(f"[{i+1}] {file_path}: Skipping - bad syntax: {err[:60]}", "warn")
                continue

            filtered.append(s)

        if len(filtered) < len(suggestions):
            self._emit(f"Filtered {len(suggestions) - len(filtered)} invalid/duplicate suggestions")

        return filtered

    def _code_exists_in_project(self, code: str) -> bool:
        """Check if code already exists in any project file."""
        code_stripped = code.strip()
        if len(code_stripped) < 20:
            return False

        for py_file in self.project_root.rglob("*.py"):
            if any(ignored in py_file.parts for ignored in self.analyzer.ignored_dirs):
                continue
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                if code_stripped in content:
                    return True
            except Exception:
                continue
        return False

    def apply_upgrade(self, suggestion: Dict) -> Tuple[bool, str]:
        """Apply a single upgrade - exact match only."""
        file_path = suggestion.get('file')
        if not file_path:
            return False, "No file specified"

        target = self.project_root / file_path
        if not target.exists():
            return False, f"File not found: {file_path}"

        current_code = suggestion.get('current_code', '')
        new_code = suggestion.get('new_code', '')

        if not new_code:
            return False, "No replacement code provided"

        syntax_ok, err = self.verifier.check_syntax(new_code)
        if not syntax_ok:
            return False, f"Syntax error: {err}"

        try:
            original = target.read_text(encoding='utf-8')

            if current_code:
                if current_code not in original:
                    return False, "Current code not found - exact match required"
                modified = original.replace(current_code, new_code, 1)
            else:
                modified = original + "\n" + new_code

            structure_ok, msg = self.verifier.check_structure(modified, file_path)
            if not structure_ok:
                return False, f"Structure check failed: {msg}"

            row_id = self.db.log_upgrade(
                file_path=file_path,
                function_name=suggestion.get('function') or 'N/A',
                old_code=current_code[:500] if current_code else "",
                new_code=new_code,
                llm_reasoning=suggestion.get('reasoning', ''),
                status='applied'
            )

            target.write_text(modified, encoding='utf-8')

            self.db.save_file_context(
                file_path=file_path,
                file_hash=hashlib.md5(modified.encode()).hexdigest()[:16],
                ast_summary={},
                imports=[],
                exports=[],
                docstring="",
                llm_summary=f"Upgraded: {suggestion.get('issue', '')}"
            )

            self._applied.append({
                'file': file_path,
                'function': suggestion.get('function'),
                'row_id': row_id
            })

            self._emit(f"Applied upgrade to {file_path}")
            return True, "Applied successfully"

        except Exception as e:
            error_msg = f"Failed: {e}"
            self._emit(error_msg, "error")
            self.db.log_upgrade(
                file_path=file_path,
                function_name=suggestion.get('function') or 'N/A',
                old_code=current_code[:500] if current_code else "",
                new_code=new_code,
                llm_reasoning=suggestion.get('reasoning', ''),
                status='failed'
            )
            return False, error_msg

    def verify_upgrade(self, file_path: str) -> Tuple[bool, str]:
        """Verify an upgrade was applied correctly."""
        target = self.project_root / file_path
        if not target.exists():
            return False, "File not found"

        try:
            source = target.read_text(encoding='utf-8')
            ast.parse(source)
            return True, "Syntax valid"
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        except Exception as e:
            return False, f"Verification failed: {e}"

    def run_full_upgrade_cycle(self, max_upgrades: int = 5,
                               auto_apply: bool = False) -> Dict:
        """Complete upgrade cycle: analyze → suggest → apply."""
        self._emit("=== Smart Upgrade Cycle ===")

        analysis = self.analyze_project()
        self._emit(f"Analysis: {analysis['files_analyzed']} files, {analysis['total_files']} total")

        suggestions = self.query_for_upgrades(max_upgrades=max_upgrades)

        if not suggestions:
            self._emit("No suggestions from LLM", "warn")
            return {'analysis': analysis, 'suggestions': [], 'applied': []}

        results = []
        for i, suggestion in enumerate(suggestions):
            file_path = suggestion.get('file', 'unknown')
            self._emit(f"[{i+1}/{len(suggestions)}] {file_path}: {suggestion.get('issue', '...')[:50]}")

            if auto_apply:
                ok, msg = self.apply_upgrade(suggestion)
                results.append({
                    'file': file_path,
                    'success': ok,
                    'message': msg,
                    'suggestion': suggestion
                })
                if ok:
                    verify_ok, verify_msg = self.verify_upgrade(file_path)
                    if not verify_ok:
                        self._emit(f"  WARNING: {verify_msg}", "warn")

        if self._applied:
            self._emit("Clearing cache and re-analyzing changed files...")
            self.db.clear_llm_cache()
            self.db.save_llm_cache("upgrades_all_5", "forced", "[]", ttl_seconds=0)
            self.analyze_project(force=True)

        self._emit(f"=== Cycle complete: {len(self._applied)}/{len(suggestions)} applied ===")

        return {
            'analysis': analysis,
            'suggestions': suggestions,
            'results': results,
            'applied_count': len(self._applied),
            'groq_stats': self.groq.get_stats(),
        }

    def get_status(self) -> Dict:
        """Get current system status."""
        contexts = self.db.get_all_contexts()
        history = self.db.get_upgrade_history(limit=20)

        return {
            'files_tracked': len(contexts),
            'total_upgrades': len(history),
            'successful': sum(1 for h in history if h['status'] == 'success'),
            'failed': sum(1 for h in history if h['status'] == 'failed'),
            'pending': sum(1 for h in history if h['status'] == 'pending'),
            'groq_calls': self.groq.call_count,
            'cache_hits': self.groq.cache_hits,
            'recent': history[:5],
        }

    def revert_upgrade(self, history_id: int) -> Tuple[bool, str]:
        """Revert a previously applied upgrade."""
        history = self.db.get_upgrade_history()
        for h in history:
            if h['id'] == history_id:
                file_path = h['file_path']
                target = self.project_root / file_path

                if target.exists() and h['old_code']:
                    try:
                        current = target.read_text(encoding='utf-8')
                        modified = current.replace(h['new_code'][:200], h['old_code'][:200], 1)
                        target.write_text(modified, encoding='utf-8')
                        self.db.update_upgrade_status(history_id, 'reverted')
                        self._emit(f"Reverted upgrade to {file_path}")
                        return True, "Reverted successfully"
                    except Exception as e:
                        return False, f"Revert failed: {e}"

                return False, "No old code to revert to"

        return False, "Upgrade not found"


def quick_upgrade(project_root: Path = None) -> Dict:
    """Convenience function for one-shot upgrades."""
    if project_root is None:
        project_root = PROJECT_ROOT
    system = SmartUpgradeSystem(project_root)
    return system.run_full_upgrade_cycle(max_upgrades=5, auto_apply=True)
