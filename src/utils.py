# Copyright (c) 2024 Atick Faisal
# Licensed under the MIT License - see LICENSE file for details

import argparse
from typing import Callable, Tuple

import torch.nn as nn

from src.unet import UNetRegressor, AttentionUNetRegressor, SelfUNetRegressor


def get_regressor(args: argparse.Namespace) -> nn.Module:
    if args.model == "unet":
        return UNetRegressor(
            init_features=args.init_features
        )
    elif args.model == "attention_unet":
        return AttentionUNetRegressor(
            init_features=args.init_features
        )
    elif args.model == "self_unet":
        return SelfUNetRegressor(
            init_features=args.init_features
        )
    else:
        raise ValueError(f"Invalid model name: {args.model}")

def get_critic(args: argparse.Namespace) -> nn.Module | None:
    return None


def get_trainer_and_validator(args: argparse.Namespace) -> Tuple[Callable, Callable]:
    if args.model == "unet" or args.model == "attention_unet" or args.model == "self_unet":
        from src.trainer import train_unet as trainer, validate_unet as validator
    else:
        raise ValueError(f"Invalid model name: {args.model}")
    return trainer, validator
