import argparse

from ptqm import load_ptqm, predict_audio


def main():
    parser = argparse.ArgumentParser(
        description="Predict IOS and DOS for one synchronized mic/headset pair."
    )
    parser.add_argument("--mic", required=True, help="Talker's microphone WAV file.")
    parser.add_argument("--headset", required=True, help="Talker's headset WAV file.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the trained PTQM checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help='Inference device, e.g. "cpu", "cuda", or "mps".',
    )
    args = parser.parse_args()

    model = load_ptqm(args.checkpoint, args.device)

    ios, dos = predict_audio(
        model,
        mic_path=args.mic,
        headset_path=args.headset,
        device=args.device,
    )

    print("PTQM prediction")
    print("----------------")
    print(f"IOS: {ios:.3f}")
    print(f"DOS: {dos:.3f}")


if __name__ == "__main__":
    main()
