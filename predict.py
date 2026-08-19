"""Run PTQM on already prepared ref/deg segments.

Expected input
--------------
A folder containing:
    manifest.csv
    ref/
    deg/

The manifest must contain:
    ref_segment
    deg_segment
    session_tag
    pesqm
    echo_lag_ms
    echo_corr

Example
-------
python predict.py \
    --data unseen_run \
    --checkpoint checkpoints/ptqm.pt \
    --out predictions.csv
"""

import argparse
import math
import os

import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader

import pesqm as pesqm_mod
from ptqm_model import PTQM


SR = 16000
MAX_DURATION = 6.0
MAX_LEN = int(SR * MAX_DURATION)
DMAP_FRAMES = 374
DMAP_BANDS = pesqm_mod._BANDS.n_bands


class PTQMDataset(Dataset):
    """Dataset used for PTQM inference on pre-segmented audio."""

    def __init__(self, csv_path, root):
        self.df = pd.read_csv(csv_path)
        self.root = root

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

    def __len__(self):
        return len(self.df)

    def _load_pair(self, ref_path, deg_path):
        ref, sr_ref = torchaudio.load(os.path.join(self.root, ref_path))
        deg, sr_deg = torchaudio.load(os.path.join(self.root, deg_path))

        ref = ref.mean(0)
        deg = deg.mean(0)

        if sr_ref != SR:
            ref = AF.resample(ref, sr_ref, SR)
        if sr_deg != SR:
            deg = AF.resample(deg, sr_deg, SR)

        pair_len = min(len(ref), len(deg))
        ref, deg = ref[:pair_len], deg[:pair_len]

        # Deterministic center crop, matching evaluation in the paper.
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

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        ref_wav, deg_wav = self._load_pair(
            row["ref_segment"],
            row["deg_segment"],
        )

        eps = 1e-12

        # 20 acoustic descriptors
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

        # 3 precomputed PESQM/GCC features
        prior_feat = torch.tensor(
            [
                float(row["pesqm"]) * 100.0,
                float(row["echo_lag_ms"]) / 1000.0,
                math.log10(max(float(row["echo_corr"]), 1.0)) / 3.0,
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

        # PESQM disturbance map on the same 6 s crop.
        ref_np = ref_wav.numpy().astype(np.float64)
        deg_np = deg_wav.numpy().astype(np.float64)

        dmap = pesqm_mod.disturbance_map(ref_np, ref_np + deg_np)

        if dmap.shape[0] < DMAP_FRAMES:
            dmap = np.pad(
                dmap,
                ((0, DMAP_FRAMES - dmap.shape[0]), (0, 0)),
            )

        dmap = np.log1p(10.0 * dmap[:DMAP_FRAMES])
        dmap = torch.from_numpy(dmap).unsqueeze(0).float()

        return {
            "extra_feat": extra_feat,
            "dmap": dmap,
            "session_tag": str(row["session_tag"]),
        }


def load_model(checkpoint_path, device):
    model = PTQM().to(device)

    state = torch.load(checkpoint_path, map_location=device)

    # Older no-AST checkpoints may still contain frozen AST weights because
    # the training class instantiated AST even when use_ast=False.
    # They are not part of PTQM and can safely be discarded for inference.
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
def predict(args):
    dataset = PTQMDataset(
        os.path.join(args.data, "manifest.csv"),
        args.data,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
    )

    model = load_model(args.checkpoint, args.device)

    records = []

    for batch in loader:
        out = model(
            batch["extra_feat"].float().to(args.device),
            batch["dmap"].float().to(args.device),
        )

        ios, dos = PTQM.md_to_iosdos(out)

        for i, tag in enumerate(batch["session_tag"]):
            records.append(
                {
                    "call_id": tag,
                    "ios_pred": float(ios[i]),
                    "dos_pred": float(dos[i]),
                }
            )

    segment_predictions = pd.DataFrame(records)

    segment_path = args.out.replace(".csv", "_segments.csv")
    segment_predictions.to_csv(segment_path, index=False)

    # Mean pooling matches the conversational-level aggregation described
    # in the paper.
    per_call = (
        segment_predictions
        .groupby("call_id")
        .agg(
            n_segments=("ios_pred", "size"),
            IOS=("ios_pred", "mean"),
            DOS=("dos_pred", "mean"),
        )
        .reset_index()
    )

    per_call.to_csv(args.out, index=False)

    print(per_call.round(3).to_string(index=False))
    print(f"\nSaved per-call predictions to: {args.out}")
    print(f"Saved segment predictions to: {segment_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict IOS and DOS using PTQM."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Folder containing manifest.csv, ref/, and deg/.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the trained PTQM checkpoint.",
    )
    parser.add_argument(
        "--out",
        default="predictions.csv",
        help="Output CSV file.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help='Inference device, e.g. "cpu", "cuda", or "mps".',
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )

    predict(parser.parse_args())
