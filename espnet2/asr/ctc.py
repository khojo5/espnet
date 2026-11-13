import logging
import math
from typing import Any, List, Optional

import torch
import torch.nn.functional as F
from typeguard import typechecked


class CTC(torch.nn.Module):
    """CTC module.

    Args:
        odim: dimension of outputs
        encoder_output_size: number of encoder projection units
        dropout_rate: dropout rate (0.0 ~ 1.0)
        ctc_type: builtin or gtnctc
        reduce: reduce the CTC loss into a scalar
        ignore_nan_grad: Same as zero_infinity (keeping for backward compatiblity)
        zero_infinity:  Whether to zero infinite losses and the associated gradients.
    """

    @typechecked
    def __init__(
        self,
        odim: int,
        encoder_output_size: int,
        dropout_rate: float = 0.0,
        ctc_type: str = "builtin",
        reduce: bool = True,
        ignore_nan_grad: Optional[bool] = None,
        zero_infinity: bool = True,
        brctc_risk_strategy: str = "exp",
        brctc_group_strategy: str = "end",
        brctc_risk_factor: float = 0.0,
    ):
        super().__init__()
        eprojs = encoder_output_size
        self.dropout_rate = dropout_rate
        self.ctc_lo = torch.nn.Linear(eprojs, odim)
        self.ctc_type = ctc_type
        if ignore_nan_grad is not None:
            zero_infinity = ignore_nan_grad

        if self.ctc_type == "builtin":
            self.ctc_loss = torch.nn.CTCLoss(
                reduction="none", zero_infinity=zero_infinity
            )
        elif self.ctc_type == "builtin2":
            self.ignore_nan_grad = True
            logging.warning("builtin2")
            self.ctc_loss = torch.nn.CTCLoss(reduction="none")

        elif self.ctc_type == "gtnctc":
            from espnet2.legacy.nets.pytorch_backend.gtn_ctc import GTNCTCLossFunction

            self.ctc_loss = GTNCTCLossFunction.apply

        elif self.ctc_type == "brctc":
            try:
                import k2  # noqa
            except ImportError:
                raise ImportError("You should install K2 to use Bayes Risk CTC")

            from espnet2.asr.bayes_risk_ctc import BayesRiskCTC

            self.ctc_loss = BayesRiskCTC(
                brctc_risk_strategy, brctc_group_strategy, brctc_risk_factor
            )

        else:
            raise ValueError(f'ctc_type must be "builtin" or "gtnctc": {self.ctc_type}')

        self.reduce = reduce

    def loss_fn(self, th_pred, th_target, th_ilen, th_olen) -> torch.Tensor:
        if self.ctc_type == "builtin" or self.ctc_type == "brctc":
            th_pred = th_pred.log_softmax(2).float()
            loss = self.ctc_loss(th_pred, th_target, th_ilen, th_olen)
            if self.ctc_type == "builtin":
                size = th_pred.size(1)
            else:
                size = loss.size(0)  # some invalid examples will be excluded

            if self.reduce:
                # Batch-size average
                loss = loss.sum() / size
            else:
                loss = loss / size
            return loss

        # builtin2 ignores nan losses using the logic below, while
        # builtin relies on the zero_infinity flag in pytorch CTC
        elif self.ctc_type == "builtin2":
            th_pred = th_pred.log_softmax(2).float()
            loss = self.ctc_loss(th_pred, th_target, th_ilen, th_olen)

            if loss.requires_grad and self.ignore_nan_grad:
                # ctc_grad: (L, B, O)
                ctc_grad = loss.grad_fn(torch.ones_like(loss))
                ctc_grad = ctc_grad.sum([0, 2])
                indices = torch.isfinite(ctc_grad)
                size = indices.long().sum()
                if size == 0:
                    # Return as is
                    logging.warning(
                        "All samples in this mini-batch got nan grad."
                        " Returning nan value instead of CTC loss"
                    )
                    return loss
                elif size != th_pred.size(1):
                    logging.warning(
                        f"{th_pred.size(1) - size}/{th_pred.size(1)}"
                        " samples got nan grad."
                        " These were ignored for CTC loss."
                    )

                    # Create mask for target
                    target_mask = torch.full(
                        [th_target.size(0)],
                        1,
                        dtype=torch.bool,
                        device=th_target.device,
                    )
                    s = 0
                    for ind, le in enumerate(th_olen):
                        if not indices[ind]:
                            target_mask[s : s + le] = 0
                        s += le

                    # Calc loss again using maksed data
                    loss = self.ctc_loss(
                        th_pred[:, indices, :],
                        th_target[target_mask],
                        th_ilen[indices],
                        th_olen[indices],
                    )
            else:
                size = th_pred.size(1)

            if self.reduce:
                # Batch-size average
                loss = loss.sum() / size
            else:
                loss = loss / size
            return loss

        elif self.ctc_type == "gtnctc":
            log_probs = torch.nn.functional.log_softmax(th_pred, dim=2)
            return self.ctc_loss(log_probs, th_target, th_ilen, 0, "none")

        else:
            raise NotImplementedError

    def forward(self, hs_pad, hlens, ys_pad, ys_lens):
        """Calculate CTC loss.

        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            hlens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
        """
        # hs_pad: (B, L, NProj) -> ys_hat: (B, L, Nvocab)
        ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))

        if self.ctc_type == "brctc":
            loss = self.loss_fn(ys_hat, ys_pad, hlens, ys_lens).to(
                device=hs_pad.device, dtype=hs_pad.dtype
            )
            return loss

        elif self.ctc_type == "gtnctc":
            # gtn expects list form for ys
            ys_true = [y[y != -1] for y in ys_pad]  # parse padded ys
        else:
            # ys_hat: (B, L, D) -> (L, B, D)
            ys_hat = ys_hat.transpose(0, 1)
            # (B, L) -> (BxL,)
            ys_true = torch.cat([ys_pad[i, :l] for i, l in enumerate(ys_lens)])

        loss = self.loss_fn(ys_hat, ys_true, hlens, ys_lens).to(
            device=hs_pad.device, dtype=hs_pad.dtype
        )

        return loss

    def softmax(self, hs_pad):
        """softmax of frame activations

        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: softmax applied 3d tensor (B, Tmax, odim)
        """
        return F.softmax(self.ctc_lo(hs_pad), dim=2)

    def log_softmax(self, hs_pad):
        """log_softmax of frame activations

        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: log softmax applied 3d tensor (B, Tmax, odim)
        """
        return F.log_softmax(self.ctc_lo(hs_pad), dim=2)

    def argmax(self, hs_pad):
        """argmax of frame activations

        Args:
            torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: argmax applied 2d tensor (B, Tmax)
        """
        return torch.argmax(self.ctc_lo(hs_pad), dim=2)

    def forced_align(self, hs_pad, hlens, ys_pad, ys_lens, blank_idx=0):
        """Force alignment between input and target sequences (Viterbi path).

        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            hlens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
            blank_idx: index of blank symbol
            Note: B must be 1.

        Returns:
            alignments: Tuple(tensor, tensor):
                - Label for each time step in the alignment path computed
                using forced alignment.
                - Log probability scores of the labels for each time step.

        """
        import torchaudio

        if self.ctc_type != "builtin":
            raise NotImplementedError("force_align needs builtin CTC")
        log_probs = self.log_softmax(hs_pad)  # (B, Tmax, odim)
        assert log_probs.size(0) == 1, "Forced alignment needs batch size 1"
        assert not (ys_pad == blank_idx).any(), "Target has blank tokens."
        align_label, align_prob = torchaudio.functional.forced_align(
            log_probs, ys_pad, hlens, ys_lens, blank=blank_idx
        )
        return align_label, align_prob


class LabelPriorCTC(CTC):
    """CTC with label priors for less peaky alignments.

    Implementation of "Less peaky and more accurate CTC forced alignment by
    label priors" (Huang et al., ICASSP 2024, arXiv:2406.02560).
    Each alignment path is scored by ``prod_t y_{pi_t}^t / P(pi_t)^alpha``, i.e.,
    ``alpha * log P(k)`` is subtracted from the log-posteriors before the CTC
    forward-backward. Since the resulting scores are no longer normalized,
    ``k2.CtcLoss`` is used instead of ``torch.nn.CTCLoss``.

    The priors ``P(k)`` are initialized to a uniform distribution. During
    training, the frame-level posteriors (without priors) are accumulated, and
    ``update_prior()`` must be called at the end of every training epoch to
    replace the priors with the accumulated average posterior. Statistics are
    accumulated for every forward call in training mode (including
    intermediate CTC losses sharing this module). With DDP, every rank must
    call ``update_prior()`` since the statistics are summed by ``all_reduce``.

    ``softmax``, ``log_softmax`` and ``argmax`` return the plain posteriors
    (e.g., for ASR decoding), while ``forced_align`` applies the priors.

    Args:
        odim: dimension of outputs
        encoder_output_size: number of encoder projection units
        dropout_rate: dropout rate (0.0 ~ 1.0)
        ctc_type: must be "label_prior"
        reduce: reduce the CTC loss into a scalar
        ignore_nan_grad: Same as zero_infinity (keeping for backward compatiblity)
        zero_infinity: Whether to zero infinite losses and the associated gradients.
        prior_scaling_factor: The scale of the label priors (alpha in the paper).
            alpha=0 is identical to the standard CTC loss.
        output_beam: Beam of k2 intersection. A large value disables pruning.
        log_prior_floor: Lower bound of the log-priors ``log P(k)`` to avoid
            excessive boosts of rare tokens (-12.0 as in the official recipe).
        **kwargs: Other keyword arguments of CTC (ignored by this class).
    """

    @typechecked
    def __init__(
        self,
        odim: int,
        encoder_output_size: int,
        dropout_rate: float = 0.0,
        ctc_type: str = "label_prior",
        reduce: bool = True,
        ignore_nan_grad: Optional[bool] = None,
        zero_infinity: bool = True,
        prior_scaling_factor: float = 0.0,
        output_beam: float = 1e20,
        log_prior_floor: float = -12.0,
        **kwargs: Any,
    ):
        if ctc_type != "label_prior":
            raise ValueError(f'ctc_type must be "label_prior": {ctc_type}')
        try:
            import k2
        except ImportError:
            raise ImportError("You should install K2 to use LabelPriorCTC")

        super().__init__(
            odim=odim,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            ctc_type="builtin",
            reduce=reduce,
            ignore_nan_grad=ignore_nan_grad,
            zero_infinity=zero_infinity,
            **kwargs,
        )
        self.ctc_type = ctc_type
        if ignore_nan_grad is not None:
            zero_infinity = ignore_nan_grad
        self.zero_infinity = zero_infinity
        self.prior_scaling_factor = prior_scaling_factor
        self.log_prior_floor = log_prior_floor
        self.ctc_loss = k2.CtcLoss(
            output_beam=output_beam, reduction="none", use_double_scores=True
        )

        # log P(k), initialized to uniform: (V,)
        self.register_buffer("log_prior", torch.full((odim,), -math.log(odim)))
        # Sum of posteriors over training frames: (V,), float64
        # Plain attributes (not buffers) so that DDP does not overwrite the
        # per-rank statistics by broadcasting buffers from rank 0.
        self._prior_stats: Optional[torch.Tensor] = None

    def apply_prior(self, log_probs: torch.Tensor) -> torch.Tensor:
        """Return ``log_probs - alpha * log P(k)``.

        Args:
            log_probs: log-posteriors (..., V), where V = odim
        """
        if self.prior_scaling_factor == 0.0:
            return log_probs
        # (..., V) - (V,) -> (..., V)
        return log_probs - self.prior_scaling_factor * self.log_prior.to(
            log_probs.dtype
        )

    @torch.no_grad()
    def accumulate_prior(self, log_probs: torch.Tensor, hlens: torch.Tensor):
        """Accumulate the posteriors of valid frames for the label priors.

        Args:
            log_probs: log-posteriors without priors (B, Tmax, V)
            hlens: batch of lengths of hidden state sequences (B)
        """
        _, T, V = log_probs.shape
        mask = (
            torch.arange(T, device=log_probs.device)[None, :]
            < hlens.to(log_probs.device)[:, None]
        )  # (B, Tmax)
        # (B, Tmax, V)
        log_probs = log_probs.detach().masked_fill(~mask[..., None], float("-inf"))
        # Sum over frames in the log domain first to avoid float32 underflow
        # (B * Tmax, V) -> (V,)
        stats = torch.logsumexp(log_probs.reshape(-1, V), dim=0).double().exp()
        if self._prior_stats is None or self._prior_stats.device != stats.device:
            self._prior_stats = torch.zeros_like(stats)
        self._prior_stats += stats  # (V,)

    @torch.no_grad()
    def update_prior(self):
        """Update the label priors with the statistics accumulated so far.

        Call this at the end of every training epoch. The statistics are
        reset afterwards.
        """
        V = self.log_prior.numel()
        # Numerator: sum of posteriors over all frames in the epoch (V,)
        stats = torch.zeros(V, dtype=torch.float64, device=self.log_prior.device)
        if self._prior_stats is not None:
            stats += self._prior_stats.to(stats.device)
        self._prior_stats = None

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(stats)

        # Denominator: total number of frames, since sum_k y_k^t = 1 (scalar)
        total = stats.sum()
        if total <= 0:
            logging.warning("No statistics for label priors. Keep the priors.")
            return
        prior = (stats / total).clamp_min(torch.finfo(torch.float64).tiny)  # (V,)
        log_prior = prior.log().clamp_min(self.log_prior_floor)  # (V,)
        self.log_prior.copy_(log_prior.to(self.log_prior.dtype))
        logging.info(f"Label priors are updated: P(blank)={prior[0].item():.4f}")

    def forward(self, hs_pad, hlens, ys_pad, ys_lens):
        """Calculate CTC loss with label priors.

        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            hlens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
        """
        ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))  # (B, Tmax, V)
        log_probs = ys_hat.log_softmax(dim=2).float()  # (B, Tmax, V)
        if self.training:
            self.accumulate_prior(log_probs, hlens)

        # (B,)
        loss = self._k2_ctc_loss(self.apply_prior(log_probs), hlens, ys_pad, ys_lens)
        size = log_probs.size(0)
        if self.reduce:
            # Batch-size average
            loss = loss.sum() / size  # ()
        else:
            loss = loss / size  # (B,)
        return loss.to(device=hs_pad.device, dtype=hs_pad.dtype)

    def _k2_ctc_loss(
        self,
        log_probs: torch.Tensor,
        hlens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute utterance-level CTC losses with k2 (blank must be 0).

        Infeasible examples (too short inputs) get an infinite loss, which is
        zeroed if ``zero_infinity`` is True, as in ``torch.nn.CTCLoss``.

        Args:
            log_probs: log-posteriors with priors (B, Tmax, V)
            hlens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)

        Returns:
            torch.Tensor: losses (B,)
        """
        import k2

        B = log_probs.size(0)
        hlens_list: List[int] = hlens.tolist()  # B x int
        # B x List[int] (unpadded label sequences)
        ys = [ys_pad[i, :l].tolist() for i, l in enumerate(ys_lens.tolist())]

        def _is_feasible(i: int) -> bool:
            y = ys[i]
            num_repeats = sum(a == b for a, b in zip(y[:-1], y[1:]))
            return 0 < hlens_list[i] and len(y) + num_repeats <= hlens_list[i]

        # k2.intersect_dense requires durations in descending order
        indices = sorted(range(B), key=lambda i: -hlens_list[i])
        # N (<= B) feasible utterance ids, sorted by length
        indices = [i for i in indices if _is_feasible(i)]

        loss = log_probs.new_full((B,), float("inf"))  # (B,)
        if len(indices) > 0:
            # (N, 3): [utterance id in batch, start frame, num frames]
            supervision = torch.tensor(
                [[i, 0, hlens_list[i]] for i in indices], dtype=torch.int32
            )
            dense_fsa_vec = k2.DenseFsaVec(log_probs, supervision)
            # N CTC graphs, in the same order as supervision
            graphs = k2.ctc_graph(
                [ys[i] for i in indices], modified=False, device=log_probs.device
            )
            loss_valid = self.ctc_loss(graphs, dense_fsa_vec).to(loss.dtype)  # (N,)
            loss = loss.index_put(
                (torch.tensor(indices, device=loss.device),), loss_valid
            )

        if len(indices) < B:
            logging.warning(
                f"{B - len(indices)}/{B} samples are too short for CTC alignment."
            )
        if self.zero_infinity:
            loss = torch.where(torch.isinf(loss), torch.zeros_like(loss), loss)
        return loss

    def forced_align(
        self, hs_pad, hlens, ys_pad, ys_lens, blank_idx=0, apply_prior=True
    ):
        """Force alignment between input and target sequences (Viterbi path).

        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            hlens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
            blank_idx: index of blank symbol
            apply_prior: whether to apply the label priors in the Viterbi search
            Note: B must be 1.

        Returns:
            alignments: Tuple(tensor, tensor):
                - Label for each time step in the alignment path computed
                using forced alignment.
                - Scores of the labels for each time step (log probabilities
                with the label priors applied if apply_prior is True).
        """
        import torchaudio

        log_probs = self.log_softmax(hs_pad)  # (B, Tmax, V)
        if apply_prior:
            log_probs = self.apply_prior(log_probs)  # (B, Tmax, V)
        assert log_probs.size(0) == 1, "Forced alignment needs batch size 1"
        assert not (ys_pad == blank_idx).any(), "Target has blank tokens."
        align_label, align_prob = torchaudio.functional.forced_align(
            log_probs, ys_pad, hlens, ys_lens, blank=blank_idx
        )
        return align_label, align_prob
