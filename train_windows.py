import dataclasses
import logging
import os
import pprint
import platform
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import torch.cuda
import torch.distributed as dist
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from torch.optim import AdamW, lr_scheduler

from finetune.args import TrainArgs
from finetune.checkpointing import Checkpointer
from finetune.data.data_loader import build_data_loader
from finetune.distributed import (
    BACKEND,
    avg_aggregate,
    get_rank,
    get_world_size,
    is_torchrun,
    set_device,
)
from finetune.eval import evaluate
from finetune.loss import compute_loss_with_mask
from finetune.mixed_precision import (
    downcast_mixed_precision,
    prepare_mixed_precision,
    upcast_mixed_precision,
)
from finetune.monitoring.metrics_logger import (
    MetricsLogger,
    eval_log_msg,
    get_eval_logs,
    get_train_logs,
    train_log_msg,
)
from finetune.monitoring.utils import set_logger
from finetune.utils import (
    TrainState,
    logged_closing,
    set_random_seed,
)
from finetune.wrapped_model import load_model, load_args

if TYPE_CHECKING:
    from mistral_common.tokens.tokenizers.sentencepiece import InstructTokenizerBase

logger = logging.getLogger("train")


def main_logger_info(message: str) -> None:
    # For Windows, always log since we're single-process
    logger.info(message)


def train(config: str):
    args: TrainArgs = TrainArgs.load(config, drop_extra_fields=False)
    print(f"args: {args}")
    set_logger(logging.INFO)

    with ExitStack() as exit_stack:
        _train(args, exit_stack)
    logger.info("Closed everything!")


def _train(
    args: TrainArgs,
    exit_stack: ExitStack,
):
    # 1. Initial setup and checks
    set_random_seed(args.seed)

    # Windows-compatible distributed setup
    if platform.system() == "Windows":
        print("Windows detected - using single-process training mode")
        
        # Set up environment variables for single-process mode
        os.environ["LOCAL_RANK"] = "0"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        
        # Set device for single GPU
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            logger.info(f"Using CUDA device: {torch.cuda.current_device()}")
        else:
            raise RuntimeError("CUDA is required for training")
            
        # Skip distributed initialization on Windows
        logger.info("Skipping distributed training initialization on Windows")
    else:
        # Original distributed setup for non-Windows systems
        if "LOCAL_RANK" in os.environ:
            set_device()
            logger.info("Going to init comms...")
            dist.init_process_group(backend=BACKEND)
        else:
            logger.error(
                "PyTorch environment is not correctly initialized. This message should only be displayed when testing."
            )

    # 2. Init run dir
    main_logger_info(f"Run dir: {args.run_dir}")
    run_dir = Path(args.run_dir)

    # Modified directory check for Windows
    if platform.system() == "Windows":
        if run_dir.exists():
            print(f"Warning: Run dir {run_dir} already exists. Continuing anyway...")
    else:
        if is_torchrun():
            if run_dir.exists():
                raise RuntimeError(
                    f"Run dir {run_dir} already exists. Make sure to either rename `run_dir` or remove {run_dir}."
                )

    # Skip barrier on Windows
    if platform.system() != "Windows":
        dist.barrier()
    
    run_dir.mkdir(exist_ok=True, parents=True)

    args_path = run_dir / "args.yaml"
    if not args_path.exists():
        args.save(args_path)

    main_logger_info(f"TrainArgs: {pprint.pformat(dataclasses.asdict(args))}")

    # 3. Get loggers
    metrics_logger: MetricsLogger = MetricsLogger(
        run_dir,
        tag="train",
        is_master=True,  # Always True for Windows single-process
        wandb_args=args.wandb,
        mlflow_args=args.mlflow,
        config=dataclasses.asdict(args),
    )
    exit_stack.enter_context(logged_closing(metrics_logger, "metrics_logger"))

    eval_logger: MetricsLogger = MetricsLogger(
        run_dir,
        tag="eval",
        is_master=True,  # Always True for Windows single-process
        wandb_args=args.wandb,
        mlflow_args=args.mlflow,
        config=dataclasses.asdict(args),
    )
    exit_stack.enter_context(logged_closing(eval_logger, "eval_logger"))

    # 5. Potentially download model
    if Path(args.model_id_or_path).is_dir():
        model_folder = Path(args.model_id_or_path)
    else:
        raise ValueError(
            "Invalid folder path. Please set `args.initial_model` to a valid folder path."
        )

    # 6. Load function calling instruct tokenizer
    vocab_size = load_args(model_folder, args.lora).vocab_size
    is_tekken = vocab_size > 32768

    instruct_tokenizer: InstructTokenizerBase = MistralTokenizer.v3(
        # is_tekken=is_tekken
    ).instruct_tokenizer  # type: ignore

    # 7. Load data loaders
    data_loader = build_data_loader(
        instruct_tokenizer=instruct_tokenizer,
        args=args.data,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed=args.seed,
        rank=0,  # Always 0 for Windows single-process
        world_size=1,  # Always 1 for Windows single-process
        is_eval=False,
    )

    if not args.no_eval:
        assert (
            args.data.eval_instruct_data != ""
        ), "Either set `no_eval` to True or provide evaluation samples under `data.eval_instruct_data`"

        eval_data_loader = build_data_loader(
            instruct_tokenizer=instruct_tokenizer,
            args=args.data,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            seed=None,
            rank=0,  # Always 0 for Windows single-process
            world_size=1,  # Always 1 for Windows single-process
            is_eval=True,
        )
        # pre-load all eval tokens
        eval_batches = list(eval_data_loader)

    # 8. Load model
    # Define mixed precision
    param_dtype = torch.bfloat16
    optim_dtype = torch.float32

    assert args.lora is not None, "`args.lora` should be set to a valid value."

    model = load_model(
        folder=model_folder,
        lora=args.lora,
        checkpoint=args.checkpoint,
        param_dtype=param_dtype,
    )

    # 9. Load optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.optim.lr,
        betas=(0.9, 0.95),
        eps=1e-08,
        weight_decay=args.optim.weight_decay,
    )

    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.optim.lr,
        total_steps=args.max_steps,
        pct_start=args.optim.pct_start,
    )

    state = TrainState(args.max_steps)

    # 10. Initialize checkpointer
    checkpointer = Checkpointer(
        model=model,
        state=state,
        run_dir=run_dir,
        optimizer=optimizer,
        num_ckpt_keep=args.num_ckpt_keep,
    )
    # 11. Prepare mixed precision
    prepare_mixed_precision(
        model.parameters(), param_dtype=param_dtype, optim_dtype=optim_dtype
    )

    # 12. train!
    model.train()
    torch.cuda.empty_cache()

    while state.step < args.max_steps:
        state.start_step()
        is_last_step = state.step == args.max_steps

        optimizer.zero_grad()

        loss = torch.tensor([0.0], device="cuda")
        n_batch_tokens: int = 0

        for i in range(args.num_microbatches):
            # batch
            batch = next(data_loader)

            x = torch.from_numpy(batch.x).cuda(non_blocking=True)
            y = torch.from_numpy(batch.y).cuda(non_blocking=True)
            y_mask = (
                torch.from_numpy(batch.y_mask).cuda(non_blocking=True)
                if batch.y_mask is not None
                else None
            )

            # forward / backward
            output = model(
                input_ids=x,
                seqlens=batch.sizes,
            )
            mb_loss = compute_loss_with_mask(output, y, y_mask)

            mb_loss.backward()

            loss += mb_loss.detach()
            n_batch_tokens += x.numel()

            if i < args.num_microbatches - 1:
                # synchronize CUDA to re-run backward
                assert args.num_microbatches > 1  # should not happen
                torch.cuda.synchronize()

        if args.num_microbatches > 1:
            loss /= args.num_microbatches
            for p in model.parameters():
                if p.requires_grad:
                    assert p.grad is not None
                    p.grad.div_(args.num_microbatches)

        # upcast params for optimizer update
        upcast_mixed_precision(model.parameters(), optim_dtype=optim_dtype)

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)

        optimizer.step()
        scheduler.step()

        # downcast params back to param_dtype
        downcast_mixed_precision(model.parameters(), param_dtype=param_dtype)

        # logging
        if state.step % args.log_freq == 0:
            train_logs = get_train_logs(
                loss=loss.item(),
                n_batch_tokens=n_batch_tokens,
                lr=scheduler.get_last_lr()[0],
                step=state.step,
            )
            metrics_logger.log(train_logs)
            main_logger_info(train_log_msg(train_logs))

        # evaluation
        if not args.no_eval and state.step % args.eval_freq == 0:
            model.eval()
            with torch.no_grad():
                eval_loss = evaluate(
                    model=model,
                    eval_batches=eval_batches,
                    num_microbatches=args.num_microbatches,
                )

            eval_logs = get_eval_logs(
                loss=eval_loss,
                step=state.step,
            )
            eval_logger.log(eval_logs)
            main_logger_info(eval_log_msg(eval_logs))

            model.train()

        # checkpointing
        if state.step % args.ckpt_freq == 0:
            checkpointer.save_checkpoint()

        state.end_step()

    # final checkpoint
    if args.save_adapters:
        checkpointer.save_adapters()


if __name__ == "__main__":
    fire.Fire(train) 