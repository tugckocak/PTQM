import argparse
from pathlib import Path

import pandas as pd

from ptqm import PTQMPreprocessor, load_ptqm, predict_audio


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Predict IOS and DOS for multiple synchronized mic/headset pairs. "
            "Each subfolder must contain mic.wav and headset.wav."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Folder containing one subfolder per sample.",
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
    args = parser.parse_args()

    root = Path(args.input)
    model = load_ptqm(args.checkpoint, args.device)
    preprocessor = PTQMPreprocessor()

    rows = []

    for sample_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        mic_path = sample_dir / "mic.wav"
        headset_path = sample_dir / "headset.wav"

        if not mic_path.exists() or not headset_path.exists():
            print(
                f"Skipping {sample_dir.name}: "
                "expected mic.wav and headset.wav"
            )
            continue

        ios, dos = predict_audio(
            model,
            mic_path=str(mic_path),
            headset_path=str(headset_path),
            device=args.device,
            preprocessor=preprocessor,
        )

        rows.append(
            {
                "sample": sample_dir.name,
                "IOS": ios,
                "DOS": dos,
            }
        )

        print(
            f"{sample_dir.name}: "
            f"IOS={ios:.3f}, DOS={dos:.3f}"
        )

    if not rows:
        raise SystemExit(
            "No valid sample folders found. "
            "Each subfolder must contain mic.wav and headset.wav."
        )

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print(f"\nSaved predictions to: {args.out}")


if __name__ == "__main__":
    main()
