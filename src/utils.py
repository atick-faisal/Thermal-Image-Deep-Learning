# Copyright (c) 2024 Atick Faisal
# Licensed under the MIT License - see LICENSE file for details

import argparse
from typing import Callable, Tuple

import torch.nn as nn

from corenet import CoreNetDiscriminator, DenseNetNetDiscriminator, ResNetNetDiscriminator
from unet import UNetRegressor, AttentionUNetRegressor, SelfUNetRegressor, DenseNetUNet, ResNetUNet


def get_generator(args: argparse.Namespace) -> nn.Module:
    if args.model == "unet":
        return UNetRegressor()
    elif args.model == "attention_unet":
        return AttentionUNetRegressor(
            init_features=args.init_features
        )
    elif args.model == "self_unet":
        return SelfUNetRegressor(
            init_features=args.init_features
        )
    elif args.model == "corenet":
        return AttentionUNetRegressor(
            init_features=args.init_features
        )
    elif args.model == "densenet_corenet":
        return DenseNetUNet()
    elif args.model == "resnet_corenet":
        return ResNetUNet()
    else:
        raise ValueError(f"Invalid model name: {args.model}")


def get_discriminator(args: argparse.Namespace) -> nn.Module | None:
    if args.model == "corenet":
        return CoreNetDiscriminator(corenet_out_channels=args.corenet_out_channels)
    if args.model == "densenet_corenet":
        return DenseNetNetDiscriminator(corenet_out_channels=args.corenet_out_channels)
    if args.model == "resnet_corenet":
        return ResNetNetDiscriminator(corenet_out_channels=args.corenet_out_channels)
    return None


def get_trainer_and_validator(args: argparse.Namespace) -> Tuple[Callable, Callable]:
    if "unet" in args.model:
        from trainer import train_unet as trainer, validate_unet as validator
    elif "corenet" in args.model:
        from trainer import train_corenet as trainer, validate_corenet as validator
    else:
        raise ValueError(f"Invalid model name: {args.model}")
    return trainer, validator
