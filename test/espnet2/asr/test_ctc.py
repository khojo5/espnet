import math
from datetime import timedelta

import pytest
import torch

from espnet2.asr.ctc import CTC, LabelPriorCTC
from espnet2.train.epoch_end_hook import run_train_epoch_end_hooks


@pytest.fixture
def ctc_args():
    bs = 2
    h = torch.randn(bs, 10, 10)
    h_lens = torch.LongTensor([10, 8])
    y = torch.randint(0, 4, [2, 5])
    y_lens = torch.LongTensor([5, 2])
    return h, h_lens, y, y_lens


@pytest.mark.parametrize("ctc_type", ["builtin"])
def test_ctc_forward_backward(ctc_type, ctc_args):
    ctc = CTC(encoder_output_size=10, odim=5, ctc_type=ctc_type)
    ctc(*ctc_args).sum().backward()


@pytest.mark.parametrize("ctc_type", ["builtin"])
def test_ctc_softmax(ctc_type, ctc_args):
    ctc = CTC(encoder_output_size=10, odim=5, ctc_type=ctc_type)
    ctc.softmax(ctc_args[0])


@pytest.mark.parametrize("ctc_type", ["builtin"])
def test_ctc_log_softmax(ctc_type, ctc_args):
    ctc = CTC(encoder_output_size=10, odim=5, ctc_type=ctc_type)
    ctc.log_softmax(ctc_args[0])


@pytest.mark.parametrize("ctc_type", ["builtin"])
def test_ctc_argmax(ctc_type, ctc_args):
    ctc = CTC(encoder_output_size=10, odim=5, ctc_type=ctc_type)
    ctc.argmax(ctc_args[0])


def test_bayes_risk_ctc(ctc_args):
    # Skip the test if K2 is not installed
    try:
        import k2  # noqa
    except ImportError:
        return

    builtin_ctc = CTC(encoder_output_size=10, odim=5, ctc_type="builtin")
    bayes_risk_ctc = CTC(encoder_output_size=10, odim=5, ctc_type="brctc")
    bayes_risk_ctc.ctc_lo = builtin_ctc.ctc_lo

    builtin_ctc_loss = builtin_ctc(*ctc_args)
    bayes_risk_ctc_loss = bayes_risk_ctc(*ctc_args)

    assert torch.abs(builtin_ctc_loss - bayes_risk_ctc_loss) < 1e-6


def test_ctc_forced_align(ctc_args):
    _ = pytest.importorskip("torchaudio")

    ctc = CTC(encoder_output_size=10, odim=5, ctc_type="builtin")
    hs_pad, hlens, ys_pad, ys_lens = ctc_args
    # Forced alignment only works with batch size 1.
    hs_pad = hs_pad[0:1, :]
    hlens = hlens[0:1]
    ys_pad = ys_pad[0:1, :]
    ys_lens = ys_lens[0:1]
    b, t, _ = hs_pad.shape
    blank_idx = 0
    ys_pad[ys_pad == blank_idx] = (
        1  # make sure there is no blank in the target sequence
    )
    aligns, scores = ctc.forced_align(
        hs_pad=hs_pad, hlens=hlens, ys_pad=ys_pad, ys_lens=ys_lens, blank_idx=blank_idx
    )
    assert aligns.shape == (b, t)
    assert scores.shape == (b, t)


def _label_prior_ctc_inputs():
    torch.manual_seed(0)
    h = torch.randn(3, 12, 10)
    # The last example is too short for its target (infeasible)
    h_lens = torch.LongTensor([12, 9, 3])
    y = torch.LongTensor([[1, 2, 2, 3, 4], [4, 4, 1, -1, -1], [1, 2, 3, 4, 1]])
    y_lens = torch.LongTensor([5, 3, 5])
    return h, h_lens, y, y_lens


def _ref_ctc_nll(log_probs, y):
    """Differentiable CTC forward algorithm. log_probs: (T, V)"""
    neg = -1e10
    ext = [0]
    for token in y:
        ext += [token, 0]
    S = len(ext)
    skip = torch.tensor(
        [s >= 2 and ext[s] != 0 and ext[s] != ext[s - 2] for s in range(S)]
    )
    emit = log_probs[:, ext]  # (T, S)
    init = torch.full((S,), neg, dtype=log_probs.dtype)
    init[: min(S, 2)] = 0.0
    alpha = init + emit[0]
    for t in range(1, log_probs.size(0)):
        a1 = torch.cat([alpha.new_full((1,), neg), alpha[:-1]])
        a2 = torch.cat([alpha.new_full((2,), neg), alpha[:-2]])
        a2 = torch.where(skip, a2, torch.full_like(a2, neg))
        alpha = torch.logsumexp(torch.stack([alpha, a1, a2]), dim=0) + emit[t]
    return -torch.logsumexp(alpha[-2:], dim=0)


@pytest.mark.parametrize("reduce", [True, False])
@pytest.mark.parametrize(
    "prior_scaling_factor,non_uniform_prior",
    [(0.0, False), (0.0, True), (0.3, False)],
)
def test_label_prior_ctc_without_correction_matches_builtin(
    reduce, prior_scaling_factor, non_uniform_prior
):
    pytest.importorskip("k2")
    h, h_lens, y, y_lens = _label_prior_ctc_inputs()
    builtin_ctc = CTC(encoder_output_size=10, odim=5, reduce=reduce)
    prior_ctc = LabelPriorCTC(
        encoder_output_size=10,
        odim=5,
        reduce=reduce,
        prior_scaling_factor=prior_scaling_factor,
    )
    prior_ctc.ctc_lo.load_state_dict(builtin_ctc.ctc_lo.state_dict())
    if non_uniform_prior:
        prior_ctc.log_prior.copy_(torch.randn(5).log_softmax(0))

    h1 = h.clone().requires_grad_()
    h2 = h.clone().requires_grad_()
    loss1 = builtin_ctc(h1, h_lens, y, y_lens)
    loss2 = prior_ctc(h2, h_lens, y, y_lens)
    assert loss1.shape == loss2.shape
    torch.testing.assert_close(loss1, loss2)

    loss1.sum().backward()
    loss2.sum().backward()
    torch.testing.assert_close(h1.grad, h2.grad, atol=1e-6, rtol=1e-5)
    for p1, p2 in zip(builtin_ctc.ctc_lo.parameters(), prior_ctc.ctc_lo.parameters()):
        torch.testing.assert_close(p1.grad, p2.grad, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("reduce", [True, False])
@pytest.mark.parametrize("zero_infinity", [True, False])
def test_label_prior_ctc_all_infeasible(reduce, zero_infinity):
    pytest.importorskip("k2")
    ctc = LabelPriorCTC(
        encoder_output_size=4,
        odim=3,
        reduce=reduce,
        zero_infinity=zero_infinity,
    )
    h = torch.randn(2, 2, 4, requires_grad=True)  # (B, T, D)
    h_lens = torch.tensor([2, 2])  # (B,)
    y = torch.tensor([[1, 1, -1], [1, 2, 1]])  # (B, L)
    y_lens = torch.tensor([2, 3])  # (B,)
    loss = ctc(h, h_lens, y, y_lens)  # () or (B,)
    assert loss.shape == (() if reduce else (2,))
    if zero_infinity:
        torch.testing.assert_close(loss, torch.zeros_like(loss))
        loss.sum().backward()
        assert h.grad is not None
        torch.testing.assert_close(h.grad, torch.zeros_like(h))
        for parameter in ctc.ctc_lo.parameters():
            assert parameter.grad is not None
            torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter))
    else:
        assert torch.isposinf(loss).all()


def test_label_prior_ctc_all_infeasible_with_zero_probabilities():
    pytest.importorskip("k2")
    ctc = LabelPriorCTC(encoder_output_size=4, odim=3)
    log_probs = torch.tensor(
        [[[0.0, float("-inf"), float("-inf")]]], requires_grad=True
    )  # (B, T, V)
    loss = ctc._k2_ctc_loss(
        log_probs, torch.tensor([1]), torch.tensor([[1, 1]]), torch.tensor([2])
    )  # (B,)
    torch.testing.assert_close(loss, torch.zeros_like(loss))
    loss.sum().backward()
    assert log_probs.grad is not None
    torch.testing.assert_close(log_probs.grad, torch.zeros_like(log_probs))


def test_label_prior_ctc_uniform_prior_shifts_loss_only():
    pytest.importorskip("k2")
    h, h_lens, y, y_lens = _label_prior_ctc_inputs()
    alpha = 0.3
    builtin_ctc = CTC(encoder_output_size=10, odim=5, reduce=False)
    prior_ctc = LabelPriorCTC(
        encoder_output_size=10, odim=5, reduce=False, prior_scaling_factor=alpha
    )
    prior_ctc.ctc_lo.load_state_dict(builtin_ctc.ctc_lo.state_dict())
    prior_ctc.log_prior.fill_(-math.log(5))

    h1 = h.clone().requires_grad_()
    h2 = h.clone().requires_grad_()
    loss1 = builtin_ctc(h1, h_lens, y, y_lens)
    loss2 = prior_ctc(h2, h_lens, y, y_lens)
    # Each frame gets +alpha*log(V) for feasible examples
    shift = torch.tensor([1.0, 1.0, 0.0]) * alpha * math.log(5) * h_lens / 3
    torch.testing.assert_close(loss2, loss1 - shift)

    loss1.sum().backward()
    loss2.sum().backward()
    torch.testing.assert_close(h1.grad, h2.grad, atol=1e-6, rtol=1e-5)


def test_label_prior_ctc_matches_reference():
    pytest.importorskip("k2")
    h, h_lens, y, y_lens = _label_prior_ctc_inputs()
    h, h_lens, y, y_lens = h[:2], h_lens[:2], y[:2], y_lens[:2]
    alpha = 0.3
    prior_ctc = LabelPriorCTC(
        encoder_output_size=10, odim=5, reduce=False, prior_scaling_factor=alpha
    )
    log_prior = torch.tensor([0.6, 0.1, 0.1, 0.15, 0.05]).log()
    prior_ctc.log_prior.copy_(log_prior)

    h1 = h.clone().requires_grad_()
    loss = prior_ctc(h1, h_lens, y, y_lens)
    loss.sum().backward()

    h2 = h.clone().double().requires_grad_()
    log_probs = (
        torch.nn.functional.linear(
            h2, prior_ctc.ctc_lo.weight.double(), prior_ctc.ctc_lo.bias.double()
        ).log_softmax(-1)
        - alpha * log_prior.double()
    )
    ref = (
        torch.stack(
            [
                _ref_ctc_nll(log_probs[i, : h_lens[i]], y[i, : y_lens[i]].tolist())
                for i in range(2)
            ]
        )
        / 2
    )
    ref.sum().backward()

    torch.testing.assert_close(loss, ref.float())
    torch.testing.assert_close(h1.grad, h2.grad.float(), atol=1e-6, rtol=1e-5)


def test_label_prior_ctc_update_prior():
    pytest.importorskip("k2")
    h, h_lens, y, y_lens = _label_prior_ctc_inputs()
    prior_ctc = LabelPriorCTC(encoder_output_size=10, odim=5, prior_scaling_factor=0.3)
    torch.testing.assert_close(prior_ctc.log_prior, torch.zeros(5))

    # No statistics are accumulated in eval mode
    prior_ctc.eval()
    prior_ctc(h, h_lens, y, y_lens)
    prior_ctc.update_prior()
    torch.testing.assert_close(prior_ctc.log_prior, torch.zeros(5))

    prior_ctc.train()
    prior_ctc(h, h_lens, y, y_lens)
    prior_ctc(h[:2], h_lens[:2], y[:2], y_lens[:2])
    prior_ctc.eval()
    prior_ctc(h * 3, h_lens, y, y_lens)
    run_train_epoch_end_hooks(prior_ctc, epoch=1)

    with torch.no_grad():
        probs = prior_ctc.softmax(h)
    frames = [probs[i, : h_lens[i]] for i in range(3)]
    frames += [probs[i, : h_lens[i]] for i in range(2)]
    expected = torch.cat(frames).mean(0)
    torch.testing.assert_close(prior_ctc.log_prior.exp(), expected)
    assert prior_ctc._prior_stats is None
    assert "log_prior" in prior_ctc.state_dict()


def _distributed_prior_worker(rank, init_method):
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        ctc = LabelPriorCTC(encoder_output_size=2, odim=3)
        stats = [[2.0, 1.0, 1.0], [1.0, 3.0, 2.0]]
        ctc._prior_stats = torch.tensor(stats[rank], dtype=torch.float64)  # (V,)
        run_train_epoch_end_hooks(ctc, epoch=1)
        torch.testing.assert_close(ctc.log_prior.exp(), torch.tensor([0.3, 0.4, 0.3]))
        assert ctc._prior_stats is None

        if rank == 1:
            ctc._prior_stats = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float64)
        run_train_epoch_end_hooks(ctc, epoch=2)
        torch.testing.assert_close(ctc.log_prior.exp(), torch.tensor([0.25, 0.5, 0.25]))
        assert ctc._prior_stats is None
    finally:
        torch.distributed.destroy_process_group()


def test_label_prior_ctc_distributed_epoch_update(tmp_path):
    pytest.importorskip("k2")
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_gloo_available()
    ):
        pytest.skip("Gloo distributed backend is unavailable")
    torch.multiprocessing.spawn(
        _distributed_prior_worker,
        args=((tmp_path / "prior_group").as_uri(),),
        nprocs=2,
    )


def test_label_prior_ctc_update_prior_floor():
    pytest.importorskip("k2")
    prior_ctc = LabelPriorCTC(encoder_output_size=10, odim=3, log_prior_floor=-12.0)
    prior_ctc._prior_stats = torch.tensor([1.0, 1e-8, 0.0], dtype=torch.float64)
    prior_ctc.update_prior()
    expected = torch.tensor([1.0, 1e-8, 0.0], dtype=torch.float64)
    expected = (expected / expected.sum()).log().clamp_min(-12.0).float()
    torch.testing.assert_close(prior_ctc.log_prior, expected)
    assert prior_ctc.log_prior.min().item() == -12.0


def test_label_prior_ctc_invalid_ctc_type():
    pytest.importorskip("k2")
    with pytest.raises(ValueError):
        LabelPriorCTC(encoder_output_size=10, odim=5, ctc_type="builtin")


def test_label_prior_ctc_forced_align(ctc_args):
    pytest.importorskip("k2")
    pytest.importorskip("torchaudio")
    ctc = LabelPriorCTC(encoder_output_size=10, odim=5, prior_scaling_factor=0.3)
    ctc.log_prior.copy_(torch.tensor([0.6, 0.1, 0.1, 0.15, 0.05]).log())
    hs_pad, hlens, ys_pad, ys_lens = ctc_args
    hs_pad, hlens, ys_pad, ys_lens = hs_pad[:1], hlens[:1], ys_pad[:1], ys_lens[:1]
    ys_pad[ys_pad == 0] = 1
    aligns, scores = ctc.forced_align(hs_pad, hlens, ys_pad, ys_lens)
    assert aligns.shape == (1, 10)
    assert scores.shape == (1, 10)
