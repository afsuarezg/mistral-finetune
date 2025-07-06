import logging
import os
import platform
from functools import lru_cache
from typing import List, Union

import torch
import torch.distributed as dist

logger = logging.getLogger("distributed")

BACKEND = "nccl"


@lru_cache()
def get_rank() -> int:
    # Windows-compatible rank detection
    if platform.system() == "Windows" and "RANK" in os.environ:
        return int(os.environ["RANK"])
    try:
        return dist.get_rank()
    except:
        return 0


@lru_cache()
def get_world_size() -> int:
    # Windows-compatible world size detection
    if platform.system() == "Windows" and "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    try:
        return dist.get_world_size()
    except:
        return 1


def visible_devices() -> List[int]:
    return [int(d) for d in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]


def set_device():
    logger.info(f"torch.cuda.device_count: {torch.cuda.device_count()}")
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    logger.info(f"local rank: {int(os.environ['LOCAL_RANK'])}")

    assert torch.cuda.is_available()

    assert len(visible_devices()) == torch.cuda.device_count()

    if torch.cuda.device_count() == 1:
        # gpus-per-task set to 1
        torch.cuda.set_device(0)
        return

    local_rank = int(os.environ["LOCAL_RANK"])
    logger.info(f"Set cuda device to {local_rank}")

    assert 0 <= local_rank < torch.cuda.device_count(), (
        local_rank,
        torch.cuda.device_count(),
    )
    torch.cuda.set_device(local_rank)


def avg_aggregate(metric: Union[float, int]) -> Union[float, int]:
    buffer = torch.tensor([metric], dtype=torch.float32, device="cuda")
    dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
    return buffer[0].item() / get_world_size()


def is_torchrun() -> bool:
    return "TORCHELASTIC_RESTART_COUNT" in os.environ
