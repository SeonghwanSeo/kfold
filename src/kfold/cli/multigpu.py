"""Distribute independent jobs across visible GPUs."""

import torch


def launch(worker, args, jobs):
    if args.num_gpus > torch.cuda.device_count():
        raise ValueError(
            f"Requested {args.num_gpus} GPUs; visible: {torch.cuda.device_count()}."
        )
    workers = min(args.num_gpus, len(jobs))
    if workers == 1:
        worker(0, args, jobs)
        return
    import multiprocessing
    from multiprocessing.connection import wait

    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=worker, args=(rank, args, jobs[rank::workers]))
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
