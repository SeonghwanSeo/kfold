"""Run one prediction stage in isolated GPU processes."""

import logging
import multiprocessing
from multiprocessing.connection import wait


def launch(worker, args, jobs, *, stage: str):
    import torch

    # Validate GPU selection and distribute jobs across isolated workers.
    num_visible = torch.cuda.device_count()
    if any(gpu_id >= num_visible for gpu_id in args.gpu_ids):
        raise ValueError(
            f"Requested GPU IDs {args.gpu_ids}; visible GPU count: {num_visible}."
        )
    workers = min(len(args.gpu_ids), len(jobs))
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_worker_entry,
            args=(worker, args.gpu_ids[rank], args, jobs[rank::workers], workers),
        )
        for rank in range(workers)
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

    # Configure process-local logging and CUDA before entering the stage worker.
    gpu_label = f" [GPU {gpu_id}]" if num_workers > 1 else ""
    logging.basicConfig(
        level=logging.WARNING,
        format=f"%(levelname)s: [%(name)s]{gpu_label} %(message)s",
        force=True,
    )
    logging.getLogger("kfold").setLevel(logging.INFO)
    torch.cuda.set_device(gpu_id)
    torch.set_float32_matmul_precision("highest")
    worker(gpu_id, args, jobs)
