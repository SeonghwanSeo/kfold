# Copyright 2026 Korea Advanced Institute of Science and Technology (KAIST)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run one prediction stage in isolated GPU processes."""

import logging
import multiprocessing
from multiprocessing.connection import wait


def distribute(costs: list[int], num_workers: int, method: str) -> list[list[int]]:
    """Assign job indices round-robin or balance their estimated costs."""
    if method == "round-robin":
        return [list(range(rank, len(costs), num_workers)) for rank in range(num_workers)]
    groups: list[list[int]] = [[] for _ in range(num_workers)]
    loads = [0] * num_workers
    for index in sorted(range(len(costs)), key=lambda index: costs[index], reverse=True):
        rank = min(range(num_workers), key=lambda rank: loads[rank])
        groups[rank].append(index)
        loads[rank] += costs[index]
    # Preserve the stage's original processing order within each worker.
    return [sorted(group) for group in groups]


def launch(worker, args, worker_jobs, *, stage: str):
    """Launch one GPU process for each preassigned group of jobs."""
    import torch

    # Validate GPU selection and launch the preassigned jobs in isolated workers.
    num_visible = torch.cuda.device_count()
    if any(gpu_id >= num_visible for gpu_id in args.gpu_ids):
        raise ValueError(
            f"Requested GPU IDs {args.gpu_ids}; visible GPU count: {num_visible}."
        )
    workers = len(worker_jobs)
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_worker_entry,
            args=(worker, args.gpu_ids[rank], args, jobs, workers),
        )
        for rank, jobs in enumerate(worker_jobs)
    ]
    # Start workers and stop the stage as soon as one reports a failure.
    try:
        for process in processes:
            process.start()
        pending = {
            process.sentinel: (process, gpu_id)
            for process, gpu_id in zip(processes, args.gpu_ids, strict=False)
        }
        while pending:
            for sentinel in wait(pending):
                process, gpu_id = pending.pop(sentinel)
                process.join()
                if process.exitcode:
                    raise SystemExit(
                        f"{stage} stopped: GPU {gpu_id} worker failed "
                        f"(exit {process.exitcode})."
                    )
    finally:
        # Reap remaining workers when the stage fails or is interrupted.
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()


def _worker_entry(worker, gpu_id, args, jobs, num_workers):
    """Configure a GPU process and show inference logs and library warnings."""
    import torch
    from huggingface_hub.utils import disable_progress_bars

    # Spawned workers do not inherit the parent's progress-bar settings.
    disable_progress_bars()
    # Configure process-local logging and CUDA before entering the stage worker.
    gpu_label = f" | gpu-{gpu_id}" if num_workers > 1 else ""
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s | %(name)s{gpu_label} | %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
    )
    logging.getLogger("cli").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    torch.cuda.set_device(gpu_id)
    torch.set_float32_matmul_precision("highest")
    worker(gpu_id, args, jobs)
