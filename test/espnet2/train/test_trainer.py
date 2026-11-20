import copy

import pytest
import torch

from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.distributed_utils import DistributedOption
from espnet2.train.epoch_end_hook import TrainEpochEndHook, run_train_epoch_end_hooks
from espnet2.train.reporter import Reporter
from espnet2.train.trainer import Trainer, TrainerOptions


class ToyModel(AbsESPnetModel):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x, **kwargs):
        loss = self.linear(x).pow(2).mean()
        # loss.detach() shares the storage of loss, as in the ESPnet models
        return dict(loss=loss, stats=dict(loss=loss.detach()), weight=x.new_ones(1))

    def collect_feats(self, **kwargs):
        return {}


class ToyIterFactory(AbsIterFactory):
    def build_iter(self, epoch, shuffle=None):
        return [(["utt"], dict(x=torch.ones(1, 2)))]


class EpochState(torch.nn.Module, TrainEpochEndHook):
    def __init__(self):
        super().__init__()
        self.register_buffer("value", torch.zeros(()))  # ()
        self.epochs = []

    def on_train_epoch_end(self, epoch: int) -> None:
        assert not torch.is_grad_enabled()
        self.epochs.append(epoch)
        self.value.add_(1)


class ToyHookModel(ToyModel):
    def __init__(self):
        super().__init__()
        self.epoch_state = EpochState()
        self.shared_state = self.epoch_state
        self.observed_states = []

    def forward(self, x, **kwargs):
        self.observed_states.append((self.training, self.epoch_state.value.item()))
        return super().forward(x, **kwargs)


def _options(output_dir, max_epoch, resume, patience, accum_grad=1):
    return TrainerOptions(
        ngpu=0,
        resume=resume,
        use_amp=False,
        train_dtype="float32",
        grad_noise=False,
        accum_grad=accum_grad,
        grad_clip=5.0,
        grad_clip_type=2.0,
        log_interval=None,
        no_forward_run=False,
        use_matplotlib=False,
        use_tensorboard=False,
        use_wandb=False,
        adapter="lora",
        use_adapter=False,
        save_strategy="all",
        output_dir=output_dir,
        max_epoch=max_epoch,
        seed=0,
        sharded_ddp=False,
        patience=patience,
        keep_nbest_models=1,
        nbest_averaging_interval=0,
        early_stopping_criterion=("valid", "loss", "min"),
        best_model_criterion=[("valid", "loss", "min")],
        val_scheduler_criterion=("valid", "loss"),
        unused_parameters=False,
        wandb_model_log_interval=-1,
        create_graph_in_tensorboard=False,
        gradient_as_bucket_view=True,
        ddp_comm_hook=None,
    )


def _run(output_dir, max_epoch, resume, patience, model=None):
    if model is None:
        model = ToyModel()
    # With a zero learning rate the loss never improves, so the training is
    # stopped early once the patience is exceeded.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    options = _options(output_dir, max_epoch, resume, patience)
    Trainer.run(
        model=model,
        optimizers=[optimizer],
        schedulers=[None],
        train_iter_factory=ToyIterFactory(),
        valid_iter_factory=ToyIterFactory(),
        plot_attention_iter_factory=None,
        trainer_options=options,
        distributed_option=DistributedOption(),
    )
    reporter = Reporter()
    reporter.load_state_dict(torch.load(output_dir / "checkpoint.pth")["reporter"])
    return reporter.get_epoch()


def test_resume_after_early_stopping_does_not_train_more(tmp_path):
    # The loss of epoch 1 is the best, so the training stops after epoch 3.
    assert _run(tmp_path, max_epoch=10, resume=False, patience=1) == 3
    assert _run(tmp_path, max_epoch=10, resume=True, patience=1) == 3


def test_resume_after_early_stopping_with_larger_patience_continues(tmp_path):
    assert _run(tmp_path, max_epoch=10, resume=False, patience=1) == 3
    assert _run(tmp_path, max_epoch=10, resume=True, patience=3) == 5


def test_epoch_end_hook_timing_checkpoint_and_resume(tmp_path):
    model = ToyHookModel()
    assert _run(tmp_path, 2, False, None, model=model) == 2
    assert model.observed_states == [(True, 0), (False, 0), (True, 1), (False, 1)]
    assert model.epoch_state.epochs == [1, 2]
    saved = torch.load(tmp_path / "checkpoint.pth")["model"]
    assert saved["epoch_state.value"].item() == 2

    resumed = ToyHookModel()
    assert _run(tmp_path, 3, True, None, model=resumed) == 3
    assert resumed.observed_states == [(True, 2), (False, 2)]
    assert resumed.epoch_state.epochs == [3]
    assert resumed.epoch_state.value.item() == 3

    finished = ToyHookModel()
    assert _run(tmp_path, 3, True, None, model=finished) == 3
    assert finished.observed_states == []
    assert finished.epoch_state.epochs == []
    assert finished.epoch_state.value.item() == 3


def test_epoch_end_hook_requires_explicit_opt_in():
    class UnrelatedModule(torch.nn.Module):
        def on_train_epoch_end(self, epoch):
            raise AssertionError("This module did not opt in to epoch-end hooks")

    model = torch.nn.Sequential(UnrelatedModule(), EpochState())
    run_train_epoch_end_hooks(model, epoch=1)
    assert model[1].epochs == [1]


def test_epoch_end_hook_errors_propagate():
    class FailingModule(torch.nn.Module, TrainEpochEndHook):
        def on_train_epoch_end(self, epoch: int) -> None:
            raise RuntimeError("epoch update failed")

    with pytest.raises(RuntimeError, match="epoch update failed"):
        run_train_epoch_end_hooks(FailingModule(), epoch=1)


def test_epoch_end_hook_rejects_data_parallel(tmp_path):
    model = ToyHookModel()
    options = _options(tmp_path, 1, False, None)
    options.ngpu = 2
    with pytest.raises(ValueError, match="Use DistributedDataParallel"):
        Trainer.run(
            model=model,
            optimizers=[torch.optim.SGD(model.parameters(), lr=0.0)],
            schedulers=[None],
            train_iter_factory=ToyIterFactory(),
            valid_iter_factory=ToyIterFactory(),
            plot_attention_iter_factory=None,
            trainer_options=options,
            distributed_option=DistributedOption(),
        )
    assert model.observed_states == []


def test_train_one_epoch_with_grad_accumulation(tmp_path):
    """accum_grad=2: the same updates as a hand-written loop, every step reported.

    The statistics are registered after backward(), and the reported train loss
    is the mean of the losses: the trainer does not divide the loss in place by
    accum_grad, which would divide loss.detach() in the statistics too.
    """
    torch.manual_seed(0)
    model = ToyModel()
    reference = copy.deepcopy(model)
    batches = [(["utt"], dict(x=torch.full((1, 2), float(i)))) for i in range(1, 5)]

    reporter = Reporter()
    with reporter.observe("train", 1) as sub_reporter:
        Trainer.train_one_epoch(
            model=model,
            iterator=batches,
            optimizers=[torch.optim.SGD(model.parameters(), lr=0.1)],
            schedulers=[None],
            scaler=None,
            reporter=sub_reporter,
            summary_writer=None,
            options=_options(tmp_path, 1, False, 1, accum_grad=2),
            distributed_option=DistributedOption(),
        )

    optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    losses = []
    for i, (_, batch) in enumerate(batches, 1):
        loss = reference(**batch)["loss"]
        losses.append(loss.item())
        (loss / 2).backward()
        if i % 2 == 0:
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad()

    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p, q)
    assert reporter.get_value("train", "loss", 1) == pytest.approx(
        sum(losses) / len(losses)
    )
