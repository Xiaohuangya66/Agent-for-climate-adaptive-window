from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


NEG_INF = -1e9


def _load_transopto(workspace: Path):
    path = workspace / "model.py"
    spec = importlib.util.spec_from_file_location("thermo_agent_legacy_model", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TransOpto


class TTTPolicyAdapter:
    """Constrained Transformer sampler with test-time policy adaptation."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        checkpoint: str | Path,
        vocab_file: str | Path,
        device: str = "cuda",
        max_layers: int = 20,
        learning_rate: float = 1e-4,
        kl_coefficient: float = 0.01,
        train_mode: str = "decoder_only",
    ) -> None:
        self.workspace = Path(workspace).resolve()
        trl = self.workspace / "Transopto_TRL"
        if str(trl) not in sys.path:
            sys.path.insert(0, str(trl))
        from ttt_discover_for_MaterialsGenaration.entropic_advantage import (
            compute_entropic_advantages_adaptive_beta,
        )
        from ttt_discover_for_MaterialsGenaration.generation_wvo2 import (
            build_structural_next_token_mask,
            generate_rollouts,
        )

        self.compute_advantages = compute_entropic_advantages_adaptive_beta
        self.build_mask = build_structural_next_token_mask
        self.generate_rollouts = generate_rollouts
        self.device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        self.max_layers = max_layers
        self.kl_coefficient = kl_coefficient
        self.token2id = {
            token.strip(): index
            for index, token in enumerate(Path(vocab_file).read_text(encoding="utf-8").splitlines())
            if token.strip()
        }
        self.id2token = {value: key for key, value in self.token2id.items()}
        TransOpto = _load_transopto(self.workspace)
        self.model = TransOpto(
            spectrum_dim=221,
            vocab_size=len(self.token2id),
            d_model=256,
            nhead=4,
            num_encoder_layers=3,
            num_decoder_layers=6,
            dim_feedforward=512,
            dropout=0.01,
        )
        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        state = {key.removeprefix("model."): value for key, value in payload["state_dict"].items()}
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)
        self.reference = copy.deepcopy(self.model).eval()
        for parameter in self.reference.parameters():
            parameter.requires_grad = False
        if train_mode == "decoder_only":
            self.model.spectrum_projector.requires_grad_(False)
            self.model.view_embeddings.requires_grad = False
            self.model.transformer.encoder.requires_grad_(False)
        elif train_mode != "full":
            raise ValueError(train_mode)
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.Adam(trainable, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
        self.bos_id = self.token2id["BOS"]
        self.eos_id = self.token2id["EOS"]
        self.pad_id = self.token2id["PAD"]

    def generate(self, spectrum: torch.Tensor, count: int, temperature: float):
        return self.generate_rollouts(
            self.model,
            spectrum.to(self.device),
            count,
            self.token2id,
            self.id2token,
            max_layers=self.max_layers,
            temperature=temperature,
            device=str(self.device),
        )

    def restart_from_reference(self) -> None:
        """Restore the frozen checkpoint while retaining trainable parameter masks."""
        self.model.load_state_dict(self.reference.state_dict(), strict=True)
        self.optimizer.state.clear()

    def _masked_logits(self, model, spectrum, rollout):
        target = rollout.token_ids.to(self.device)
        count, sequence_length = target.shape
        bos = torch.full((count, 1), self.bos_id, dtype=torch.long, device=self.device)
        inputs = torch.cat([bos, target[:, :-1]], dim=1) if sequence_length > 1 else bos
        logits = model(
            spectrum.to(self.device).unsqueeze(0).expand(count, -1, -1),
            inputs,
            inputs == self.pad_id,
        )
        finished = torch.zeros(count, dtype=torch.bool, device=self.device)
        masked = logits.clone()
        positions = rollout.required_wvo2_positions.to(self.device)
        for step in range(sequence_length):
            legal = self.build_mask(
                inputs[:, : step + 1], finished, self.token2id, step, self.max_layers, positions
            )
            masked[:, step, :] = masked[:, step, :].masked_fill(~legal, NEG_INF)
            finished |= target[:, step] == self.eos_id
        return masked, target, rollout.masks.to(self.device)

    def adapt(self, spectrum: torch.Tensor, rollout, rewards: torch.Tensor) -> dict[str, float]:
        rewards = rewards.to(self.device, dtype=torch.float32)
        advantages = self.compute_advantages(rewards, gamma=float(torch.log(torch.tensor(2.0))))
        self.model.train()
        self.optimizer.zero_grad()
        logits, target, masks = self._masked_logits(self.model, spectrum, rollout)
        log_prob = F.log_softmax(logits, dim=-1).gather(2, target.unsqueeze(-1)).squeeze(-1)
        rollout_log_prob = (log_prob * masks).sum(dim=1)
        with torch.no_grad():
            ref_logits, _, _ = self._masked_logits(self.reference, spectrum, rollout)
            ref_log_prob = F.log_softmax(ref_logits, dim=-1).gather(2, target.unsqueeze(-1)).squeeze(-1)
        kl = ((log_prob - ref_log_prob) * masks).sum(dim=1)
        loss = -(advantages * rollout_log_prob).mean() + self.kl_coefficient * kl.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad], 1.0
        )
        self.optimizer.step()
        return {
            "loss": float(loss.detach().cpu()),
            "mean_reward": float(rewards.mean().detach().cpu()),
            "max_reward": float(rewards.max().detach().cpu()),
            "mean_kl": float(kl.mean().detach().cpu()),
        }
