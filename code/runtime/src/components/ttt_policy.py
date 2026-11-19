"""Real autoregressive sampling and decoder policy-gradient adaptation."""
from __future__ import annotations
import copy
import importlib.util
from pathlib import Path
import torch
from torch.nn import functional as F
from src.agent.baseline_common import ROOT
from src.vendor.thermo_agent.entropic_advantage import compute_entropic_advantages_adaptive_beta


class TTTPolicy:
    def __init__(self, space, checkpoint=None, model_file=None, device='cpu', nhead=4,
                 learning_rate=1e-4, kl_coefficient=0.01, model=None):
        self.space = space
        self.device = torch.device(device)
        self.nhead = nhead
        self.checkpoint = Path(checkpoint or ROOT / 'data/checkpoints/model_S-epoch=1198-val_loss_epoch=3.436.ckpt').resolve()
        if model is None:
            from src.support_settings import asset_path
            model_file = Path(model_file or asset_path('model_file', ROOT.parent / 'topic3/model.py'))
            spec = importlib.util.spec_from_file_location('baseline_transopto_model', model_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            model = module.TransOpto(spectrum_dim=221, vocab_size=len(space.token2id), d_model=256,
                nhead=nhead, num_encoder_layers=3, num_decoder_layers=6, dim_feedforward=512, dropout=0.01)
            payload = torch.load(self.checkpoint, map_location='cpu', weights_only=False)
            model.load_state_dict({k.removeprefix('model.'): v for k, v in payload['state_dict'].items()}, strict=True)
        self.model = model.to(self.device).eval()
        self.reference = copy.deepcopy(self.model).eval()
        for parameter in self.reference.parameters():
            parameter.requires_grad = False
        if hasattr(model, 'spectrum_projector'):
            self.model.spectrum_projector.requires_grad_(False)
            self.model.view_embeddings.requires_grad = False
            self.model.transformer.encoder.requires_grad_(False)
        self.trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(self.trainable, lr=learning_rate)
        self.kl_coefficient = kl_coefficient
        self.bos, self.eos, self.pad = [space.token2id[t] for t in ['BOS', 'EOS', 'PAD']]
        self.material_ids = [space.token2id[t] for t in space.tokens]
        self.wvo2_ids = [space.token2id[t] for t in space.wvo2]
        self.terminal_id = space.token2id[space.terminal] if space.terminal else None

    def _legal(self, prefix):
        allowed = torch.zeros((prefix.shape[0], len(self.space.token2id)), dtype=torch.bool, device=self.device)
        for row, sequence in enumerate(prefix.tolist()):
            materials = [t for t in sequence if t in self.material_ids]
            if self.eos in sequence:
                allowed[row, self.pad] = True
                continue
            if self.terminal_id is not None and materials and materials[-1] == self.terminal_id:
                allowed[row, self.eos] = True
                continue
            if len(materials) >= self.space.max_layers:
                allowed[row, self.eos] = True
                continue
            allowed[row, self.material_ids] = True
            count = sum(t in self.wvo2_ids for t in materials)
            required = self.space.required_wvo2
            missing = max(0, required-count)
            if required and count >= required:
                allowed[row, self.wvo2_ids] = False
            if self.space.maximum_wvo2 is not None and count >= self.space.maximum_wvo2:
                allowed[row, self.wvo2_ids] = False
            if self.terminal_id is None and materials and not missing:
                allowed[row, self.eos] = True
            if self.terminal_id is not None:
                if missing:
                    allowed[row, self.terminal_id] = False
                elif len(materials) == self.space.max_layers-1:
                    allowed[row] = False
                    allowed[row, self.terminal_id] = True
            if missing and self.space.max_layers-len(materials) <= missing+int(self.terminal_id is not None):
                allowed[row] = False
                allowed[row, self.wvo2_ids] = True
        if not allowed.any(dim=1).all():
            raise RuntimeError('No legal next tokens for configured search space')
        return allowed

    @torch.no_grad()
    def sample(self, spectrum, count, temperature=1.0):
        if temperature <= 0:
            raise ValueError('temperature must be positive')
        self.model.eval()
        spectrum = spectrum.to(self.device).unsqueeze(0).expand(count, -1, -1)
        prefix = torch.full((count, 1), self.bos, dtype=torch.long, device=self.device)
        for _ in range(self.space.max_layers+1):
            logits = self.model(spectrum, prefix, prefix == self.pad)[:, -1, :] / temperature
            logits = logits.masked_fill(~self._legal(prefix), -1e9)
            token = torch.multinomial(torch.softmax(logits, -1), 1)
            prefix = torch.cat([prefix, token], 1)
            if (prefix == self.eos).any(dim=1).all():
                break
        generated = prefix[:, 1:]
        structures = []
        for sequence in generated.tolist():
            tokens = []
            for token in sequence:
                if token == self.eos:
                    break
                if token in self.material_ids:
                    tokens.append(self.space.vocab[token])
            structures.append(' '.join(tokens))
        return {'token_ids': generated.detach(), 'structures': structures, 'temperature': temperature}

    def adapt(self, spectrum, rollout, rewards):
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        if rewards.ndim != 1 or len(rewards) != len(rollout['structures']) or len(rewards) != rollout['token_ids'].shape[0]:
            raise ValueError('Exactly one scalar reward is required per sampled trajectory')
        if not torch.isfinite(rewards).all():
            raise ValueError('TTT rewards must be finite')
        target = rollout['token_ids'].to(self.device)
        bos = torch.full((target.shape[0], 1), self.bos, dtype=torch.long, device=self.device)
        inputs = torch.cat([bos, target[:, :-1]], 1)
        spectra = spectrum.to(self.device).unsqueeze(0).expand(target.shape[0], -1, -1)
        # Keep dropout disabled so sampling and differentiated replay use the same policy.
        self.model.eval()
        logits = self.model(spectra, inputs, inputs == self.pad) / rollout['temperature']
        with torch.no_grad():
            ref_logits = self.reference(spectra, inputs, inputs == self.pad) / rollout['temperature']
        for step in range(target.shape[1]):
            legal = self._legal(inputs[:, :step+1])
            logits[:, step, :] = logits[:, step, :].masked_fill(~legal, -1e9)
            ref_logits[:, step, :] = ref_logits[:, step, :].masked_fill(~legal, -1e9)
        log_pi = F.log_softmax(logits, -1)
        log_ref = F.log_softmax(ref_logits, -1)
        # Include EOS, exclude PAD/post-EOS positions.
        mask = (target != self.pad).float()
        log_sequence = (log_pi.gather(-1, target.unsqueeze(-1)).squeeze(-1) * mask).sum(1)
        advantages = compute_entropic_advantages_adaptive_beta(rewards).detach()
        kl = (log_pi.exp() * (log_pi-log_ref)).sum(-1)
        mean_kl = (kl*mask).sum()/mask.sum().clamp_min(1)
        loss = -(advantages*log_sequence).mean() + self.kl_coefficient*mean_kl
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite TTT policy loss')
        self.optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.trainable, 1.0, error_if_nonfinite=True)
        before = [p.detach().clone() for p in self.trainable]
        self.optimizer.step()
        delta = sum((p.detach()-old).double().square().sum().item() for p, old in zip(self.trainable, before))**0.5
        return {'loss': float(loss.detach()), 'mean_reward': float(rewards.mean()),
                'reward_std': float(rewards.std(unbiased=False)), 'kl': float(mean_kl.detach()),
                'gradient_norm_before_clip': float(norm), 'parameter_delta_l2': delta,
                'parameters_changed': delta > 0}
