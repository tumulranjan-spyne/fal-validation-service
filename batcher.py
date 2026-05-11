"""
Dynamic request batcher for GPU inference.

Collects individual inference requests and runs them as one batch when either:
  - max_batch_size requests are accumulated, or
  - max_delay_ms milliseconds have passed since the first request.
"""
import time
import queue
import threading
from typing import List, Callable, Any
import numpy as np


class DynamicBatcher:
    def __init__(
        self,
        max_batch_size: int,
        max_delay_ms: float,
        model_fn: Callable[[List[np.ndarray]], List[Any]],
    ):
        self._max_batch = max_batch_size
        self._max_delay = max_delay_ms / 1000.0
        self._model_fn = model_fn
        self._queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def infer(self, input_array: np.ndarray) -> Any:
        """Submit a preprocessed array and block until result is ready."""
        result_queue: queue.Queue = queue.Queue()
        self._queue.put((input_array, result_queue))
        return result_queue.get()

    def _loop(self) -> None:
        batch: List[np.ndarray] = []
        result_queues: List[queue.Queue] = []
        last_arrival = time.time()

        def flush() -> None:
            nonlocal batch, result_queues
            if not batch:
                return
            results = self._model_fn(batch)
            for rq, out in zip(result_queues, results):
                rq.put(out)
            batch.clear()
            result_queues.clear()

        while not self._stop.is_set():
            try:
                inp, rq = self._queue.get(timeout=self._max_delay)
                batch.append(inp)
                result_queues.append(rq)
                last_arrival = time.time()
                if len(batch) >= self._max_batch:
                    flush()
            except queue.Empty:
                if batch and (time.time() - last_arrival) >= self._max_delay:
                    flush()

    def shutdown(self) -> None:
        self._stop.set()