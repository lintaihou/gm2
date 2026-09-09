from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

from .train import train

PRESETS = {
    "Qwen3": {"n_layers": [0, 28]},
    "Theia-K24": {"sparsities": [2, 4, 4, 2], "n_edges": 24, "n_referrals": 0},
    "Theia-K32-R2": {"sparsities": [2, 4, 4, 2], "n_edges": 32, "n_referrals": 2},
    "Theia-K24-R3": {"sparsities": [2, 4, 4, 2], "n_edges": 24},
    "Theia-K24-R4": {"sparsities": [2, 4, 4, 2], "n_edges": 24, "n_referrals": 4, "n_accum_steps": 64, "batch_size": 2},
    "Theia-K16-R4-S": {"sparsities": [2, 4, 4, 2], "n_referrals": 4, "n_dense_refreshes": 8, "n_accum_steps": 64, "batch_size": 2},
    "Theia-K16-R6": {"sparsities": [2, 4, 4, 2], "n_referrals": 6, "n_accum_steps": 64, "batch_size": 2},
    "Hyperion-K32-R1": {"n_edges": 32, "n_referrals": 1},
    "Hyperion-K24-R3": {"n_edges": 24, "n_accum_steps": 64, "batch_size": 2},
    "Hyperion-K16-R3": {},
    "Hyperion-K16-R3-S": {"n_dense_refreshes": 8, "n_accum_steps": 64, "batch_size": 2},
    "Hyperion-K16-R4": {"n_referrals": 4, "n_accum_steps": 64, "batch_size": 2},
}


def launch(override_cfg=None):
    file_cfg = OmegaConf.load(Path("configs", "default.yaml"))
    dict_cfg = OmegaConf.create({"time": datetime.now().strftime("%Y-%m-%d-%H-%M-%S"), "name": "-", "group": "-", "project": "gm2", "seed": 0})
    cfg = OmegaConf.merge(dict_cfg, file_cfg)
    OmegaConf.set_struct(cfg, True)
    if override_cfg is not None:
        cfg = OmegaConf.merge(cfg, override_cfg)
    train(cfg)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--preset", choices=PRESETS, required=True)
    parser.add_argument("--eval", metavar="CHECKPOINT", help="Evaluate a model checkpoint")
    parser.add_argument("--smoke", action="store_true", help="Use the small dataset and a 10-step training run")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("overrides", nargs="*", metavar="key=value", help="Config overrides, applied last")
    args = parser.parse_args()

    overrides = PRESETS[args.preset] | {"name": args.preset}
    if args.smoke:
        overrides.update(
            name=f"{args.preset}-smoke", dataset="fineweb-edu-10bt-smoke", n_steps=10, n_warmup_steps=1, save_interval=0, save_final=True
        )
    if args.eval:
        overrides.update(load=args.eval, n_steps=0, test=True)
    if args.no_compile:
        overrides.update(compile=False)

    launch(OmegaConf.merge(overrides, OmegaConf.from_dotlist(args.overrides)))
