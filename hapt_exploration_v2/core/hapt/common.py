import csv
import hashlib
import json
import logging
import platform
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default, allow_nan=False), encoding='utf-8')


def digest_file(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def digest_value(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=json_default).encode()).hexdigest()


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f'Empty result table: {path}')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def make_logger(directory):
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('hapt')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    for handler in [logging.StreamHandler(), logging.FileHandler(directory / 'run.log', encoding='utf-8')]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


class ResourceMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stop_event = threading.Event()
        self.started = time.perf_counter()
        self.utc_start = datetime.now(timezone.utc).isoformat()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()

    def sample(self):
        while not self.stop_event.wait(0.1):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def finish(self):
        self.stop_event.set()
        self.thread.join()
        return {'started_utc': self.utc_start, 'finished_utc': datetime.now(timezone.utc).isoformat(), 'wall_seconds': time.perf_counter() - self.started, 'sampled_peak_rss_bytes': self.peak, 'platform': platform.platform(), 'logical_cpu_count': psutil.cpu_count(), 'total_memory_bytes': psutil.virtual_memory().total, 'available_memory_bytes_end': psutil.virtual_memory().available}
