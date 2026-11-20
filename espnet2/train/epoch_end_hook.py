"""Opt-in module hooks for training-epoch state updates."""

from abc import ABC, abstractmethod

import torch


class TrainEpochEndHook(ABC):
    """Mixin for modules with state to update after a training epoch.

    The normal Trainer invokes each opted-in module once after validation and
    before saving checkpoints, on every rank. Implementations update only their
    own state and handle any distributed collectives themselves. All ranks must
    have the same module traversal order.

    Hooks run without gradients and must not depend on the module's training
    mode. DataParallel is unsupported because replica-local state is not retained.
    """

    @abstractmethod
    def on_train_epoch_end(self, epoch: int) -> None:
        """Update this module's state for the completed, one-based epoch."""
        raise NotImplementedError


@torch.no_grad()
def run_train_epoch_end_hooks(model: torch.nn.Module, epoch: int) -> None:
    """Invoke explicit hooks once per module, including shared modules."""
    for module in model.modules():
        if isinstance(module, TrainEpochEndHook):
            module.on_train_epoch_end(epoch)
