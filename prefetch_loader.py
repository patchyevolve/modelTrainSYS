"""
Thread-safe prefetch wrapper for DataLoader.
Works on Windows without num_workers (which requires fork/spawn guards).

Loads the next batch on a background thread while the current batch trains.
Effectively hides data loading latency behind compute — free speedup.
"""

import threading
import queue
from typing import Iterator, Any


class PrefetchLoader:
    """
    Wraps any DataLoader and prefetches the next batch in a background thread.

    Usage:
        loader = PrefetchLoader(DataLoader(...), buffer_size=2)
        for xb, yb in loader:
            train_step(xb, yb)

    buffer_size=2 means up to 2 batches are pre-loaded ahead.
    Higher values use more RAM but smooth out I/O spikes.
    """

    def __init__(self, loader, buffer_size: int = 2):
        self.loader      = loader
        self.buffer_size = buffer_size

    def __len__(self):
        return len(self.loader)

    def __iter__(self) -> Iterator[Any]:
        q        = queue.Queue(maxsize=self.buffer_size)
        sentinel = object()   # unique end-of-stream marker

        def _producer():
            try:
                for batch in self.loader:
                    q.put(batch)
            finally:
                q.put(sentinel)

        t = threading.Thread(target=_producer, daemon=True)
        t.start()

        while True:
            item = q.get()
            if item is sentinel:
                break
            yield item
