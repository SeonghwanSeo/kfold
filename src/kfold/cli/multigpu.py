"""Distribute independent jobs across visible GPUs."""

import torch


def launch(worker, args, jobs):
    num_visible = torch.cuda.device_count()
    if any(gpu_id >= num_visible for gpu_id in args.gpu_ids):
        raise ValueError(
            f"Requested GPU IDs {args.gpu_ids}; visible GPU count: {num_visible}."
        )
    workers = min(len(args.gpu_ids), len(jobs))
    if workers == 1:
        worker(args.gpu_ids[0], args, jobs, workers)
        return
    import multiprocessing
    from multiprocessing.connection import wait

    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=worker,
            args=(args.gpu_ids[rank], args, jobs[rank::workers], workers),
        )
        for rank in range(workers)
    ]
    try:
        for process in processes:
            process.start()
        pending = {process.sentinel: process for process in processes}
        while pending:
            for sentinel in wait(pending):
                process = pending.pop(sentinel)
                process.join()
                if process.exitcode:
                    raise RuntimeError(
                        f"GPU worker {process.pid} failed (exit {process.exitcode})."
                    )
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
