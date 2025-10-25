"""Structured logging setup with MDC and artifact management."""
import logging
import os
import json
import uuid
import time
import hashlib
import pathlib
import shutil
import re
from logging.handlers import RotatingFileHandler
from contextvars import ContextVar

# MDC Context Variables
CTX = {
    "run_id": ContextVar("run_id", default="-"),
    "lot_id": ContextVar("lot_id", default="-"),
    "issue_key": ContextVar("issue_key", default="-"),
    "node": ContextVar("node", default="-"),
    "file": ContextVar("file", default="-"),
    "func": ContextVar("func", default="-"),
}


class CtxFilter(logging.Filter):
    """Inject MDC context into log records."""
    
    def filter(self, record):
        record.run_id = CTX["run_id"].get()
        record.lot_id = CTX["lot_id"].get()
        record.issue_key = CTX["issue_key"].get()
        record.node = CTX["node"].get()
        record.file = CTX["file"].get()
        record.func = CTX["func"].get()
        return True


class RedactFilter(logging.Filter):
    """Redact secrets from log messages."""
    
    def __init__(self):
        self.patterns = [
            (re.compile(r'(squ_)[a-zA-Z0-9_-]{20,}'), r'\1***'),
            (re.compile(r'(ghp_)[a-zA-Z0-9_-]{20,}'), r'\1***'),
            (re.compile(r'(sk-)[a-zA-Z0-9_-]{20,}'), r'\1***'),
            (re.compile(r'([A-Za-z0-9_-]{40,})'), r'***'),
        ]
    
    def filter(self, record):
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            for pattern, replacement in self.patterns:
                record.msg = pattern.sub(replacement, record.msg)
        return True


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        base = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "run_id": getattr(record, "run_id", "-"),
            "lot_id": getattr(record, "lot_id", "-"),
            "issue_key": getattr(record, "issue_key", "-"),
            "node": getattr(record, "node", "-"),
            "file": getattr(record, "file", "-"),
            "func": getattr(record, "func", "-"),
        }
        
        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)
        
        # Add extras
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            base.update(record.extra)
        
        return json.dumps(base, ensure_ascii=False)


class Span:
    """Context manager for timing operations."""
    
    def __init__(self, logger, name, **kw):
        self.logger = logger
        self.name = name
        self.kw = kw
    
    def __enter__(self):
        self.t0 = time.time()
        self.logger.debug(f"span.enter:{self.name}", extra={"extra": self.kw})
        return self
    
    def __exit__(self, exc_type, exc, tb):
        dt = int((time.time() - self.t0) * 1000)
        exit_extra = {"duration_ms": dt} | self.kw
        self.logger.debug(f"span.exit:{self.name}", extra={"extra": exit_extra})


def setup_logging():
    """Setup structured logging with console and file handlers."""
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_json = os.getenv("LOG_JSON", "1") == "1"
    log_redact = os.getenv("LOG_REDACT", "1") == "1"
    
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    
    ctx_filter = CtxFilter()
    redact_filter = RedactFilter() if log_redact else None
    
    # Console handler (human readable)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(
        "%(levelname)s %(name)s [run=%(run_id)s lot=%(lot_id)s node=%(node)s]: %(message)s"
    ))
    ch.addFilter(ctx_filter)
    if redact_filter:
        ch.addFilter(redact_filter)
    
    # File handler (JSON structured)
    if log_json:
        log_dir = os.getenv("LOG_DIR", "logs")
        os.makedirs(log_dir, exist_ok=True)
        fh = RotatingFileHandler(
            os.path.join(log_dir, "pipeline.jsonl"),
            maxBytes=10_000_000,
            backupCount=5
        )
        fh.setLevel(level)
        fh.setFormatter(JsonFormatter())
        fh.addFilter(ctx_filter)
        if redact_filter:
            fh.addFilter(redact_filter)
        root.addHandler(fh)
    
    root.addHandler(ch)
    
    # Quiet noisy loggers
    logging.getLogger("urllib3").setLevel("WARNING")
    logging.getLogger("httpx").setLevel("WARNING")
    logging.getLogger("openai").setLevel("WARNING")
    
    return root


def set_ctx(**pairs):
    """Set MDC context variables."""
    for k, v in pairs.items():
        if k in CTX:
            CTX[k].set(str(v))


def sha256_text(s: str) -> str:
    """Generate SHA256 hash of text."""
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()[:16]


def save_artifact(run_dir: str, name: str, content: str) -> str:
    """Save artifact to disk and return path."""
    p = pathlib.Path(run_dir)
    p.mkdir(parents=True, exist_ok=True)
    f = p / name
    f.write_text(content, encoding="utf-8", errors="ignore")
    return str(f)


def log_event(event: str, **kw):
    """Log structured event with extra fields."""
    logger = logging.getLogger("app")
    logger.info(event, extra={"extra": kw})


def build_debug_bundle(run_id: str) -> str:
    """Create debug bundle zip file."""
    src = pathlib.Path("artifacts") / run_id
    if not src.exists():
        return ""
    
    out = pathlib.Path("artifacts") / f"{run_id}_bundle.zip"
    shutil.make_archive(out.with_suffix(""), "zip", src)
    log_event("bundle.created", path=str(out), size_mb=round(out.stat().st_size / 1024 / 1024, 2))
    return str(out)


def get_run_artifacts_dir() -> str:
    """Get artifacts directory for current run."""
    run_id = CTX["run_id"].get()
    if run_id == "-":
        return "artifacts/unknown"
    return f"artifacts/{run_id}"


def should_sample_llm() -> bool:
    """Check if LLM call should be sampled for full logging."""
    sample_rate = int(os.getenv("LOG_SAMPLE_LLM", "1"))
    if sample_rate <= 1:
        return True
    return hash(CTX["run_id"].get()) % sample_rate == 0