"""
Project Context System for Auto-Upgrade
- Per-file AST analysis stored in SQLite
- Full project context for LLM
- Cached LLM responses to minimize API calls
- Incremental updates when files change
"""

import ast
import hashlib
import json
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import logging

log = logging.getLogger("ProjectContext")

DB_PATH = Path("project_context.db")


class ProjectFileDB:
    """SQLite-backed context store for each project file."""

    def __init__(self, path: Path = DB_PATH):
        self.path = path
        self._local = threading.local()
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.path), check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        self._local.conn.execute("PRAGMA journal_mode=WAL")
        return self._local.conn

    def _init_schema(self):
        c = self._conn()
        c.executescript("""
        CREATE TABLE IF NOT EXISTS file_context (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path       TEXT    UNIQUE NOT NULL,
            file_hash       TEXT,
            last_modified   REAL,
            ast_summary     TEXT,
            imports         TEXT,
            exports         TEXT,
            docstring       TEXT,
            llm_summary     TEXT,
            last_analyzed   TEXT,
            analysis_count  INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS file_relationships (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file    TEXT    NOT NULL,
            target_file    TEXT    NOT NULL,
            import_type    TEXT,
            line_number    INTEGER,
            UNIQUE(source_file, target_file)
        );

        CREATE TABLE IF NOT EXISTS llm_cache (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            cache_key      TEXT    UNIQUE NOT NULL,
            request_hash   TEXT    NOT NULL,
            response       TEXT,
            created_at     TEXT,
            expires_at     TEXT,
            used_count     INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS upgrade_history (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path       TEXT    NOT NULL,
            function_name   TEXT,
            old_code        TEXT,
            new_code        TEXT,
            llm_reasoning  TEXT,
            status          TEXT    DEFAULT 'pending',
            perf_delta      REAL,
            error           TEXT,
            applied_at      TEXT,
            reverted_at     TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_file_path ON file_context(file_path);
        CREATE INDEX IF NOT EXISTS idx_cache_key ON llm_cache(cache_key);
        CREATE INDEX IF NOT EXISTS idx_source_file ON file_relationships(source_file);
        """)

        c.commit()

    def save_file_context(self, file_path: str, file_hash: str, 
                          ast_summary: Dict, imports: List, exports: List,
                          docstring: str, llm_summary: str = None) -> None:
        c = self._conn()
        c.execute("""
            INSERT INTO file_context (file_path, file_hash, last_modified, ast_summary, 
                                      imports, exports, docstring, llm_summary, last_analyzed, analysis_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT(file_path) DO UPDATE SET
                file_hash = excluded.file_hash,
                last_modified = excluded.last_modified,
                ast_summary = excluded.ast_summary,
                imports = excluded.imports,
                exports = excluded.exports,
                docstring = excluded.docstring,
                llm_summary = COALESCE(excluded.llm_summary, file_context.llm_summary),
                last_analyzed = excluded.last_analyzed,
                analysis_count = file_context.analysis_count + 1
        """, (file_path, file_hash, time.time(),
              json.dumps(ast_summary), json.dumps(imports),
              json.dumps(exports), docstring, llm_summary, datetime.now().isoformat()))
        c.commit()

    def get_file_context(self, file_path: str) -> Optional[Dict]:
        c = self._conn()
        row = c.execute("SELECT * FROM file_context WHERE file_path = ?", (file_path,)).fetchone()
        if row:
            return dict(row)
        return None

    def get_all_contexts(self) -> List[Dict]:
        c = self._conn()
        rows = c.execute("SELECT * FROM file_context ORDER BY file_path").fetchall()
        return [dict(r) for r in rows]

    def get_outdated_files(self, current_mtimes: Dict[str, float]) -> List[str]:
        c = self._conn()
        rows = c.execute("SELECT file_path, last_modified FROM file_context").fetchall()
        outdated = []
        for row in rows:
            fp = row["file_path"]
            if fp in current_mtimes:
                if row["last_modified"] and row["last_modified"] < current_mtimes[fp]:
                    outdated.append(fp)
        return outdated

    def save_relationship(self, source: str, target: str, 
                         import_type: str = None, line: int = None) -> None:
        c = self._conn()
        c.execute("""
            INSERT OR IGNORE INTO file_relationships 
            (source_file, target_file, import_type, line_number)
            VALUES (?, ?, ?, ?)
        """, (source, target, import_type, line))
        c.commit()

    def get_relationships(self, file_path: str = None, 
                          as_source: bool = True) -> List[Dict]:
        c = self._conn()
        if file_path:
            if as_source:
                rows = c.execute(
                    "SELECT * FROM file_relationships WHERE source_file = ?", 
                    (file_path,)).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM file_relationships WHERE target_file = ?", 
                    (file_path,)).fetchall()
        else:
            rows = c.execute("SELECT * FROM file_relationships").fetchall()
        return [dict(r) for r in rows]

    def save_llm_cache(self, cache_key: str, request_hash: str, 
                       response: str, ttl_seconds: int = 3600) -> None:
        c = self._conn()
        expires = datetime.fromtimestamp(time.time() + ttl_seconds).isoformat()
        c.execute("""
            INSERT INTO llm_cache (cache_key, request_hash, response, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                response = excluded.response,
                expires_at = excluded.expires_at,
                used_count = llm_cache.used_count + 1
        """, (cache_key, request_hash, response, datetime.now().isoformat(), expires))
        c.commit()

    def get_llm_cache(self, cache_key: str) -> Optional[str]:
        c = self._conn()
        row = c.execute("""
            SELECT response FROM llm_cache 
            WHERE cache_key = ? AND expires_at > ? AND used_count < 10
        """, (cache_key, datetime.now().isoformat())).fetchone()
        return row["response"] if row else None

    def clear_llm_cache(self) -> None:
        c = self._conn()
        c.execute("DELETE FROM llm_cache")
        c.commit()

    def invalidate_cache_for_key(self, cache_key: str) -> None:
        c = self._conn()
        c.execute("DELETE FROM llm_cache WHERE cache_key LIKE ?", (f"{cache_key}%",))
        c.commit()

    def log_upgrade(self, file_path: str, function_name: str,
                    old_code: str, new_code: str, 
                    llm_reasoning: str, status: str = "pending") -> int:
        c = self._conn()
        cur = c.execute("""
            INSERT INTO upgrade_history 
            (file_path, function_name, old_code, new_code, llm_reasoning, status, applied_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (file_path, function_name, old_code, new_code, 
              llm_reasoning, status, datetime.now().isoformat()))
        c.commit()
        return cur.lastrowid

    def update_upgrade_status(self, row_id: int, status: str, 
                             perf_delta: float = None, error: str = None) -> None:
        c = self._conn()
        if status == "reverted":
            c.execute("UPDATE upgrade_history SET status=?, reverted_at=? WHERE id=?",
                     (status, datetime.now().isoformat(), row_id))
        elif status == "success":
            c.execute("UPDATE upgrade_history SET status=?, perf_delta=? WHERE id=?",
                     (status, perf_delta, row_id))
        else:
            c.execute("UPDATE upgrade_history SET status=?, error=? WHERE id=?",
                     (status, error, row_id))
        c.commit()

    def get_upgrade_history(self, file_path: str = None, 
                            limit: int = 50) -> List[Dict]:
        c = self._conn()
        if file_path:
            rows = c.execute(
                "SELECT * FROM upgrade_history WHERE file_path = ? ORDER BY id DESC LIMIT ?",
                (file_path, limit)).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM upgrade_history ORDER BY id DESC LIMIT ?",
                (limit,)).fetchall()
        return [dict(r) for r in rows]


class ProjectAnalyzer:
    """Analyzes Python files using AST."""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.ignored_dirs = {
            '__pycache__', '.git', '.venv', 'venv', 'env',
            'randomDATA', 'trained_models', 'inference_results',
            '.git', 'backups', 'savedconfig'
        }
        self.ignored_files = {
            'setup.py', 'run.bat', '__init__.py'
        }

    def get_python_files(self) -> List[Path]:
        files = []
        for py_file in self.project_root.rglob("*.py"):
            if any(ignored in py_file.parts for ignored in self.ignored_dirs):
                continue
            if py_file.name in self.ignored_files:
                continue
            if not py_file.is_file():
                continue
            files.append(py_file)
        return sorted(files)

    def compute_file_hash(self, path: Path) -> str:
        return hashlib.md5(path.read_bytes()).hexdigest()[:16]

    def parse_file(self, path: Path) -> Optional[Dict]:
        try:
            source = path.read_text(encoding='utf-8', errors='ignore')
            tree = ast.parse(source, filename=str(path))
            return {
                'source': source,
                'tree': tree,
                'path': str(path.relative_to(self.project_root))
            }
        except Exception as e:
            log.warning(f"Failed to parse {path}: {e}")
            return None

    def extract_file_info(self, parsed: Dict) -> Dict:
        tree = parsed['tree']
        source = parsed['source']
        path = parsed['path']

        functions = []
        classes = []
        imports = []
        exports = []
        docstring = ast.get_docstring(tree) or ""

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append({
                    'name': node.name,
                    'line': node.lineno,
                    'args': [a.arg for a in node.args.args],
                    'decorators': [d.attr if hasattr(d, 'attr') else str(d) 
                                  for d in node.decorator_list],
                    'docstring': ast.get_docstring(node) or "",
                    'returns': ast.unparse(node.returns) if node.returns else None,
                })
                exports.append(node.name)

            elif isinstance(node, ast.ClassDef):
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                classes.append({
                    'name': node.name,
                    'line': node.lineno,
                    'bases': [ast.unparse(b) for b in node.bases],
                    'methods': methods,
                    'docstring': ast.get_docstring(node) or "",
                })
                exports.append(node.name)

            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                else:
                    if node.module:
                        imports.append(node.module)

        ast_summary = {
            'functions': functions,
            'classes': classes,
            'total_lines': len(source.splitlines()),
            'import_count': len(imports),
        }

        return {
            'ast_summary': ast_summary,
            'imports': imports,
            'exports': exports,
            'docstring': docstring[:500] if docstring else "",
        }

    def get_dependencies(self, parsed: Dict) -> List[Dict]:
        deps = []
        tree = parsed['tree']
        path = parsed['path']

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and not node.module.startswith('_'):
                    for alias in node.names:
                        target = f"{node.module}.{alias.name}" if alias.name != '*' else node.module
                        deps.append({
                            'target': target,
                            'type': 'import_from',
                            'line': node.lineno,
                            'name': alias.name
                        })
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    deps.append({
                        'target': alias.name,
                        'type': 'import',
                        'line': node.lineno,
                        'name': alias.name
                    })

        return deps

    def analyze_file(self, path: Path, llm_summary: str = None) -> Optional[Dict]:
        parsed = self.parse_file(path)
        if not parsed:
            return None

        info = self.extract_file_info(parsed)
        deps = self.get_dependencies(parsed)

        return {
            'file_path': str(path.relative_to(self.project_root)),
            'file_hash': self.compute_file_hash(path),
            'last_modified': path.stat().st_mtime,
            'ast_summary': info['ast_summary'],
            'imports': info['imports'],
            'exports': info['exports'],
            'docstring': info['docstring'],
            'llm_summary': llm_summary,
            'dependencies': deps,
        }

    def generate_context_for_llm(self, contexts: List[Dict], max_chars: int = 8000) -> str:
        sections = []
        sections.append("# PROJECT CONTEXT (summary)\n")

        total_chars = 0
        for ctx in contexts:
            if total_chars > max_chars:
                sections.append(f"\n... and {len(contexts) - len(sections)} more files")
                break

            sections.append(f"\n## {ctx['file_path']}")
            ast_summary = ctx.get('ast_summary', '{}')
            if isinstance(ast_summary, str):
                try:
                    ast_summary = json.loads(ast_summary)
                except json.JSONDecodeError:
                    ast_summary = {}

            if ctx.get('docstring'):
                doc = ctx['docstring'][:100]
                sections.append(f"Desc: {doc}")

            classes = ast_summary.get('classes', [])
            functions = ast_summary.get('functions', [])[:5]

            if classes:
                names = ', '.join(c['name'] for c in classes[:5])
                sections.append(f"Classes: {names}")

            if functions:
                names = ', '.join(f['name'] for f in functions[:8])
                sections.append(f"Functions: {names}")

            total_chars += sum(len(s) for s in sections[-10:])

        return '\n'.join(sections)


class GroqClientCached:
    """Groq client with response caching."""

    def __init__(self, api_key: str = None):
        import os
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.model = "llama-3.3-70b-versatile"
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        self.cache: Dict[str, str] = {}
        self.call_count = 0
        self.cache_hits = 0

    def _make_request_hash(self, messages: List[Dict]) -> str:
        content = json.dumps(messages, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def chat(self, messages: List[Dict], use_cache: bool = True,
             temperature: float = 0.2, max_tokens: int = 2048) -> str:
        request_hash = self._make_request_hash(messages)

        if use_cache:
            cached = self.cache.get(request_hash)
            if cached:
                self.cache_hits += 1
                log.info(f"Cache hit ({self.cache_hits} total)")
                return cached

        import urllib.request
        import urllib.error

        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode()

        req = urllib.request.Request(
            self.url, data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "python-groq-client/1.0",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
                content = data["choices"][0]["message"]["content"]
                self.call_count += 1
                log.info(f"Groq API call #{self.call_count}")

                if use_cache:
                    self.cache[request_hash] = content

                return content
        except Exception as e:
            log.error(f"Groq API error: {e}")
            raise

    def get_stats(self) -> Dict:
        return {
            'total_calls': self.call_count,
            'cache_hits': self.cache_hits,
            'cache_size': len(self.cache),
        }
