"""Thread-based asynchronous job queue for heavy/long-running tasks.

The queue is purpose-built to off-load blocking operations – such as LLM API
calls and sandboxed code execution – from the Streamlit main thread.  It uses a
`queue.Queue` plus a pool of worker threads to process jobs concurrently while
keeping a simple *polling* API appropriate for Streamlit’s event-loop model.

Public API
~~~~~~~~~~
>>> from utils.job_queue import submit_llm_call, get_job_status, get_job_result
>>> job_id = submit_llm_call(messages)
>>> # Render spinner while waiting
>>> status = get_job_status(job_id)

The module intentionally avoids asyncio to remain compatible with Streamlit’s
single-threaded execution model and to simplify integration.
"""
from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from queue import Queue, Empty
from enum import Enum, auto
from typing import Any, Callable, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Job & status definitions
# ---------------------------------------------------------------------------

class JobStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    DONE = auto()
    ERROR = auto()
    CANCELLED = auto()

@dataclass
class Job:
    job_id: str
    func: Callable
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    # Internal: event to support blocking `wait`
    _done_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def set_result(self, result: Any):
        self.result = result
        self.status = JobStatus.DONE
        self._done_event.set()

    def set_error(self, exc: Exception):
        self.error = str(exc)
        self.status = JobStatus.ERROR
        self._done_event.set()

    def wait(self, timeout: Optional[float] = None) -> Any:
        """Block until job finishes or *timeout* expires.  Returns job.result"""
        self._done_event.wait(timeout)
        return self.result

# ---------------------------------------------------------------------------
# Worker implementation
# ---------------------------------------------------------------------------

class _Worker(threading.Thread):
    daemon = True

    def __init__(self, queue: Queue[Job]):
        super().__init__(name="JobWorker")
        self._queue = queue
        self._running = True

    def run(self):
        while self._running:
            try:
                job: Job = self._queue.get(timeout=0.5)
            except Empty:
                continue

            if job.status == JobStatus.CANCELLED:
                self._queue.task_done()
                continue

            logger.debug(f"🔧 Worker picked job {job.job_id}")
            job.status = JobStatus.RUNNING
            try:
                result = job.func(*job.args, **job.kwargs)
                job.set_result(result)
                logger.debug(f"✅ Job {job.job_id} completed")
            except Exception as exc:
                logger.exception(f"❌ Job {job.job_id} failed: {exc}")
                job.set_error(exc)
            finally:
                self._queue.task_done()

    def stop(self):
        self._running = False

# ---------------------------------------------------------------------------
# Singleton queue manager
# ---------------------------------------------------------------------------

class _JobQueue:
    """Simple job queue singleton."""

    def __init__(self, max_workers: int = 4):
        self._queue: Queue[Job] = Queue()
        self._jobs: Dict[str, Job] = {}
        self._workers = [_Worker(self._queue) for _ in range(max_workers)]
        for w in self._workers:
            w.start()
        logger.info(f"🚀 Job queue initialised with {max_workers} worker threads")

    # ---- public API ---------------------------------------------------------

    def submit(self, func: Callable, *args, **kwargs) -> str:
        job_id = str(uuid.uuid4())
        job = Job(job_id, func, args, kwargs)
        self._jobs[job_id] = job
        self._queue.put(job)
        logger.debug(f"📥 Job {job_id} submitted: {func.__name__}")
        return job_id

    def get_status(self, job_id: str) -> Optional[JobStatus]:
        job = self._jobs.get(job_id)
        return job.status if job else None

    def get_result(self, job_id: str) -> Any:
        job = self._jobs.get(job_id)
        if not job:
            raise KeyError(f"Unknown job_id {job_id}")
        if job.status == JobStatus.ERROR:
            raise RuntimeError(job.error)
        return job.result

    def wait(self, job_id: str, timeout: Optional[float] = None) -> Any:
        job = self._jobs.get(job_id)
        if not job:
            raise KeyError(f"Unknown job_id {job_id}")
        return job.wait(timeout)

# Create global singleton
_job_queue = _JobQueue(max_workers=4)

# ---------------------------------------------------------------------------
# Convenience wrappers for common tasks
# ---------------------------------------------------------------------------

# We import lazily to avoid circular deps

def _llm_call_wrapper(messages, model, temperature, max_tokens, stream):
    from app_core.api import make_llm_call
    return make_llm_call(messages, model=model, temperature=temperature, max_tokens=max_tokens, stream=stream)


def submit_llm_call(messages, model="gpt-4.1", temperature=0.2, max_tokens=4000, stream=False) -> str:
    """Submit an LLM call to be processed in the background.  Returns *job_id*."""
    return _job_queue.submit(
        _llm_call_wrapper, messages, model, temperature, max_tokens, stream
    )


def _code_execution_wrapper(code: str, df, should_plot: bool):
    from agents.execution import ExecutionAgent  # local import to avoid cycles
    return ExecutionAgent(code, df, should_plot)


def submit_code_execution(code: str, df, should_plot: bool) -> str:
    """Submit a code-execution job to the queue.  Returns *job_id*."""
    return _job_queue.submit(_code_execution_wrapper, code, df, should_plot)

# ---------------------------------------------------------------------------
# Higher-level composite job: CodeGeneration + Execution (no reasoning)
# ---------------------------------------------------------------------------


def _analysis_pipeline(query: str, df, chat_history):
    """Run CodeGenerationAgent + ExecutionAgent synchronously (worker thread)."""
    from agents import CodeGenerationAgent, ExecutionAgent  # local import

    code, should_plot_flag, code_thinking = CodeGenerationAgent(query, df, chat_history)
    result_obj = ExecutionAgent(code, df, should_plot_flag)

    return {
        "code": code,
        "should_plot": should_plot_flag,
        "code_thinking": code_thinking,
        "result": result_obj,
    }


def submit_analysis_job(query: str, df, chat_history):
    """Submit full analysis job (code generation + execution) and return job_id."""
    return _job_queue.submit(_analysis_pipeline, query, df, chat_history)

# Re-export simple accessors
get_job_status = _job_queue.get_status
get_job_result = _job_queue.get_result
wait_for_job = _job_queue.wait 