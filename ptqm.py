import math
import os
from typing import Tuple

import numpy as np
import torch
import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as T

import pesqm as pesqm_mod
from ptqm_model import PTQM


SR = 16000
MAX_DURATION = 6.0
MAX_LEN = int(SR * MAX_DURATION)
DMAP_FRAMES = 374
DMAP_BANDS = pesqm_mod._BANDS.n_bands


class PTQMPreprocessor:
    """Shared preprocessing for PTQM inference."""

    def __init__(self):
        self.mfcc = T.MFCC(
            sample_rate=SR,
            n_mfcc=13,
            melkwargs={
                "n_fft": 400,
                "hop_length": 160,
                "n_mels": 40,
                "center": False,
            },
        )

    @staticmethod
    def _load_audio(path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)
        wav = wav.mean(0)

        if sr != SR:
            wav = AF.resample(wav, sr, SR)

        return wav

    @staticmethod
    def _crop_or_pad_pair(ref: torch.Tensor, deg: torch.Tensor):
        pair_len = min(len(ref), len(deg))
        ref, deg = ref[:pair_len], deg[:pair_len]

        if pair_len > MAX_LEN:
            start = (pair_len - MAX_LEN) // 2
            ref = ref[start:start + MAX_LEN]
            deg = deg[start:start + MAX_LEN]
        elif pair_len < MAX_LEN:
            pad = MAX_LEN - pair_len
            ref = torch.nn.functional.pad(ref, (0, pad))
            deg = torch.nn.functional.pad(deg, (0, pad))

        return ref, deg

    @staticmethod
    def _log_rms_db(wav, frame_len=400, hop=160):
        frames = wav.unfold(0, frame_len, hop)
        rms = torch.sqrt(torch.mean(frames ** 2, dim=1) + 1e-12)
        return 20.0 * torch.log10(rms + 1e-12)

    @staticmethod
    def _mean_pitch_hz(wav):
        pitch = AF.detect_pitch_frequency(wav.unsqueeze(0), SR).squeeze(0)
        voiced = pitch > 0
        return pitch[voiced].mean() if voiced.any() else torch.tensor(0.0)

    def _mfcc_mean(self, wav):
        return self.mfcc(wav.unsqueeze(0)).squeeze(0).mean(dim=1)

    @staticmethod
    def _gcc_phat(ref, deg, max_lag_s=1.0):
        ref_np = ref.detach().cpu().numpy().astype(np.float64)
        deg_np = deg.detach().cpu().numpy().astype(np.float64)

        n = len(ref_np) + len(deg_np)
        R = np.fft.rfft(deg_np, n) * np.conj(np.fft.rfft(ref_np, n))
        R /= np.maximum(np.abs(R), 1e-12)

        cc = np.fft.irfft(R, n)
        max_lag = int(min(max_lag_s * SR, len(ref_np) - 1))
        lags = np.arange(-max_lag, max_lag + 1)
        cc = np.concatenate([cc[-max_lag:], cc[:max_lag + 1]])

        mask = lags > int(0.002 * SR)
        if not mask.any():
            return np.nan, 0.0

        k = np.argmax(np.abs(cc[mask]))
        lag_ms = lags[mask][k] * 1000.0 / SR
        corr = float(
            np.abs(cc[mask])[k] /
            (np.median(np.abs(cc)) + 1e-12)
        )
        return lag_ms, corr

    def process(self, mic_path: str, headset_path: str):
        ref_wav = self._load_audio(mic_path)
        deg_wav = self._load_audio(headset_path)
        ref_wav, deg_wav = self._crop_or_pad_pair(ref_wav, deg_wav)

        eps = 1e-12

        ref_logrms = self._log_rms_db(ref_wav).mean()
        deg_logrms = self._log_rms_db(deg_wav).mean()

        ref_f0 = self._mean_pitch_hz(ref_wav)
        deg_f0 = self._mean_pitch_hz(deg_wav)

        d_mfcc = (
            self._mfcc_mean(deg_wav)
            - self._mfcc_mean(ref_wav)
        ).float()

        att_db = 20.0 * torch.log10(
            (torch.sqrt(torch.mean(deg_wav ** 2) + eps) + eps)
            / (torch.sqrt(torch.mean(ref_wav ** 2) + eps) + eps)
        ).view(1)

        ref_np = ref_wav.numpy().astype(np.float64)
        deg_np = deg_wav.numpy().astype(np.float64)

        pesqm_score = pesqm_mod.pesqm(ref_np, ref_np + deg_np)
        echo_lag_ms, echo_corr = self._gcc_phat(ref_wav, deg_wav)

        prior_feat = torch.tensor(
            [
                float(pesqm_score) * 100.0,
                float(echo_lag_ms) / 1000.0 if np.isfinite(echo_lag_ms) else 0.0,
                math.log10(max(float(echo_corr), 1.0)) / 3.0,
            ],
            dtype=torch.float32,
        )

        extra_feat = torch.cat(
            [
                torch.tensor(
                    [
                        ref_logrms,
                        deg_logrms,
                        deg_logrms - ref_logrms,
                        ref_f0,
                        deg_f0,
                        deg_f0 - ref_f0,
                    ],
                    dtype=torch.float32,
                ),
                d_mfcc,
                att_db.float(),
                prior_feat,
            ]
        )

        dmap = pesqm_mod.disturbance_map(ref_np, ref_np + deg_np)

        if dmap.shape[0] < DMAP_FRAMES:
            dmap = np.pad(
                dmap,
                ((0, DMAP_FRAMES - dmap.shape[0]), (0, 0)),
            )

        dmap = np.log1p(10.0 * dmap[:DMAP_FRAMES])
        dmap = torch.from_numpy(dmap).unsqueeze(0).float()

        return extra_feat, dmap


def load_ptqm(checkpoint_path: str, device: str = "cpu") -> PTQM:
    model = PTQM().to(device)

    state = torch.load(checkpoint_path, map_location=device)

    # Legacy no-AST checkpoints may still contain frozen AST parameters.
    state = {
        k: v for k, v in state.items()
        if not k.startswith("ast.")
    }

    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint does not match the final PTQM architecture.\n"
            f"Missing keys: {missing}\n"
            f"Unexpected keys: {unexpected}"
        )

    model.eval()
    return model


@torch.no_grad()
def predict_audio(
    model: PTQM,
    mic_path: str,
    headset_path: str,
    device: str = "cpu",
    preprocessor: PTQMPreprocessor | None = None,
) -> Tuple[float, float]:
    if preprocessor is None:
        preprocessor = PTQMPreprocessor()

    extra_feat, dmap = preprocessor.process(mic_path, headset_path)

    out = model(
        extra_feat.unsqueeze(0).to(device),
        dmap.unsqueeze(0).to(device),
    )

    ios, dos = PTQM.md_to_iosdos(out)

    return float(ios.item()), float(dos.item())
