import os
import random
import time

import pynvml


def is_nvml_initialized() -> bool:
    """
    Checks if NVML is initialized by trying to get the GPU count.
    """
    try:
        pynvml.nvmlDeviceGetCount()
        return True
    except pynvml.NVMLError:
        return False


def get_free_gpus() -> list[int]:
    """
    Check each GPU and return indices of GPUs with no active processes.

    Returns:
    - List of indices of GPUs that are not currently running any processes.
    """

    nvml_was_inited = is_nvml_initialized()
    if not nvml_was_inited:
        pynvml.nvmlInit()
    free_gpus = []
    gpu_count = pynvml.nvmlDeviceGetCount()

    for i in range(gpu_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        if len(processes) == 0:
            free_gpus.append(i)

    if not nvml_was_inited:
        pynvml.nvmlShutdown()

    return free_gpus


def set_gpu(job_id: int = -1) -> None:
    pynvml.nvmlInit()
    free_gpus = []
    first_error = True
    while len(free_gpus) == 0:
        free_gpus = get_free_gpus()
        if len(free_gpus) == 0:
            if first_error:
                print("There are no available GPUs. Waiting 60 sec for next request.")
                first_error = False
            time.sleep(60)
    num_gpus = len(free_gpus)
    if job_id >= 0:
        gpu_id = free_gpus[job_id % num_gpus]
    else:
        gpu_id = random.choice(free_gpus)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"The process has been assigned to GPU {gpu_id}")
    pynvml.nvmlShutdown()
