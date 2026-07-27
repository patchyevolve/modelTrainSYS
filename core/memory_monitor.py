import gc
import threading
import time
from dataclasses import dataclass
from typing import Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class MemSnapshot:
    available_mb: float
    used_mb: float
    percent: float
    process_mb: float


def get_available_mb() -> float:
    if PSUTIL_AVAILABLE:
        return psutil.virtual_memory().available / (1024 * 1024)
    return 1024.0


def get_process_mb() -> float:
    if PSUTIL_AVAILABLE:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    return 0.0


def get_snapshot() -> MemSnapshot:
    if PSUTIL_AVAILABLE:
        v = psutil.virtual_memory()
        return MemSnapshot(
            available_mb=v.available / (1024 * 1024),
            used_mb=v.used / (1024 * 1024),
            percent=v.percent,
            process_mb=psutil.Process().memory_info().rss / (1024 * 1024),
        )
    return MemSnapshot(available_mb=1024, used_mb=0, percent=0, process_mb=0)


def adaptive_batch_size(
    current: int,
    min_batch: int = 1,
    max_batch: int = 512,
    ramp_up_threshold_mb: float = 2048.0,
    ramp_down_threshold_mb: float = 512.0,
    scale_step: int = 2,
) -> int:
    avail = get_available_mb()
    if avail < ramp_down_threshold_mb:
        return max(min_batch, current // scale_step)
    if avail > ramp_up_threshold_mb:
        return min(max_batch, current * scale_step)
    return current


class MemoryMonitor:
    def __init__(
        self,
        max_ram_mb: int = 0,
        check_interval: float = 3.0,
        ramp_up_mb: float = 2048.0,
        ramp_down_mb: float = 512.0,
    ):
        total = psutil.virtual_memory().total / (1024 * 1024) if PSUTIL_AVAILABLE else 16384
        self.max_ram_mb = max_ram_mb if max_ram_mb > 0 else int(total * 0.85)
        self.check_interval = check_interval
        self.ramp_up_mb = ramp_up_mb
        self.ramp_down_mb = ramp_down_mb
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._current_batch: int = 32
        self._min_batch: int = 1
        self._max_batch: int = 512
        self._last_snapshot: Optional[MemSnapshot] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def recommended_batch(self) -> int:
        with self._lock:
            return self._current_batch

    @recommended_batch.setter
    def recommended_batch(self, value: int):
        with self._lock:
            self._current_batch = max(self._min_batch, min(self._max_batch, value))

    @property
    def snapshot(self) -> Optional[MemSnapshot]:
        return self._last_snapshot

    def start(self, initial_batch: int = 32, min_batch: int = 1, max_batch: int = 512):
        self._current_batch = initial_batch
        self._min_batch = min_batch
        self._max_batch = max_batch
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.wait(self.check_interval):
            self._last_snapshot = get_snapshot()
            avail = self._last_snapshot.available_mb
            limit = self.max_ram_mb
            with self._lock:
                if avail < self.ramp_down_mb or (self._last_snapshot.process_mb > limit * 0.9):
                    self._current_batch = max(self._min_batch, self._current_batch // 2)
                    gc.collect()
                elif avail > self.ramp_up_mb and self._last_snapshot.process_mb < limit * 0.7:
                    self._current_batch = min(self._max_batch, self._current_batch * 2)
