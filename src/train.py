import random
from pathlib import Path

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from torch.amp import autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, default_collate
from torchinfo import summary
from tqdm import tqdm
from transformers import AutoTokenizer, BatchEncoding, get_linear_schedule_with_warmup

from .dataset import PaddedPackedTokenDataset
from .model.core import TheiaHyperionModel


def train(cfg):
    # Start run
    tqdm.write(f"Started run: {cfg.name}")
    run_id = f"{cfg.time}-{cfg.name}"
    run = wandb.init(project=cfg.project, dir=Path("outputs"), id=run_id, name=cfg.name, config=OmegaConf.to_container(cfg), group=cfg.group)

    # Set seeds
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Use tensor cores
    torch.set_float32_matmul_precision("high")

    # Initialize model
    orig_model = TheiaHyperionModel(cfg)
    model_summary = str(summary(orig_model, depth=5, col_names=["num_params", "params_percent"], verbose=0)).splitlines()
    tqdm.write("\n".join(model_summary[:30] + ["..."] + model_summary[-30:]))
    if cfg.load:
        orig_model.load_state_dict(torch.load(cfg.load, map_location="cpu", weights_only=True))
        tqdm.write(f"Loaded weights from: {cfg.load}")
    orig_model.to("cuda")
    if cfg.compile:
        model = torch.compile(orig_model)
    else:
        model = orig_model
    model.train()

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")

    # Define collate fn
    def collate_fn(samples):
        return BatchEncoding(default_collate(samples))

    if cfg.n_steps:
        # Initialize datasets and dataloaders
        dataset = PaddedPackedTokenDataset(Path("data") / cfg.dataset / "train.bin", cfg.seq_length, 1, tokenizer.pad_token_id)
        dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True, drop_last=True)
        dataloader_iter = iter(dataloader)

        # Initialize optimizer
        decay, no_decay = [], []
        for param in model.parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2:
                decay.append(param)
            else:
                no_decay.append(param)
        groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
        optimizer = torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta_1, cfg.beta_2), eps=1e-8, fused=True)

        # Initialize scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.n_warmup_steps, num_training_steps=cfg.n_steps)

        # Initialize variables
        train_bar = tqdm(total=cfg.n_steps, desc="Train Steps")

    for step in range(cfg.n_steps + cfg.test):
        # Mark step
        torch.compiler.cudagraph_mark_step_begin()

        # Set modes
        if step < cfg.n_steps:
            mode = "train"
            n_batches = cfg.n_accum_steps
        else:
            dataset = PaddedPackedTokenDataset(Path("data") / cfg.dataset / "test.bin", cfg.seq_length, 1, tokenizer.pad_token_id)
            dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True, drop_last=True)
            dataloader_iter = iter(dataloader)
            test_bar = tqdm(total=len(dataloader), desc="Test Batches")
            mode = "test"
            model.eval()
            n_batches = len(dataloader)

        # Initialize loss sum
        loss_sum = torch.zeros((), device="cuda", dtype=torch.float32)

        for _ in range(n_batches):
            # Fetch data
            data = next(dataloader_iter).to("cuda", non_blocking=True)

            # Pass through the model
            with torch.set_grad_enabled(mode == "train"), autocast(enabled=cfg.mixed_precision, device_type="cuda", dtype=torch.bfloat16):
                loss = model(**data)

            # Record loss
            loss_sum += loss.detach().float()

            # Backpropagate
            if mode == "train":
                (loss / n_batches).backward()

            # Step test bar
            else:
                test_bar.update(1)

        if mode == "train":
            # Update weights
            clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            # Step scheduler and train bar
            scheduler.step()
            train_bar.update(1)

        # Record stats
        run.log({f"{mode}/loss": (loss_sum / n_batches).item()}, step=step)

        # Checkpoint model
        if (cfg.save_interval and (step + 1) % cfg.save_interval == 0 and step < cfg.n_steps) or (cfg.save_final and (step + 1) == cfg.n_steps):
            checkpoint_dir = Path("outputs", "checkpoints", cfg.group, run_id, "models")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"{cfg.name}-{step + 1}.pth"
            torch.save(orig_model.state_dict(), checkpoint_path)
            tqdm.write(f"Model checkpoint {step + 1} saved at {checkpoint_path}")

    # Finish run
    run.finish()
    tqdm.write(f"Finished run: {cfg.name}")
