"""Command line for grimace_from_va.

Examples
--------
python -m grimace_from_va --self-test
python -m grimace_from_va --valence 0.81 --arousal 0.66 -o happy.png
python -m grimace_from_va --csv example/sample_va.csv -o out.mp4
python -m grimace_from_va --csv va.csv --weights-out weights.csv
python -m grimace_from_va --csv va.csv --lead 1 --trail 1 -o match_c1.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .fkcn import self_check, va_to_weights
from .trajectory import DEFAULT_ALPHA, load_va_csv, map_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="grimace_from_va",
        description="Render a Grimace face from valence and arousal.",
    )
    parser.add_argument("--self-test", action="store_true", help="check the VA mapping and exit")
    parser.add_argument("--csv", type=Path, help="CSV with valence, arousal, optional timestamp_s")
    parser.add_argument("--valence", type=float, help="single valence in [-1, 1]")
    parser.add_argument("--arousal", type=float, help="single arousal in [-1, 1]")
    parser.add_argument("-o", "--out", type=Path, help="PNG for one point, MP4 for a CSV")
    parser.add_argument(
        "--weights-out",
        type=Path,
        help="also write the 6-D weights as CSV (no browser when this is the only output)",
    )
    parser.add_argument("--rows-out", type=Path, help="write one EMA'd face per input row")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="EMA alpha (default 0.6)")
    parser.add_argument("--lead", type=float, default=0.0, help="seconds of the first face before the clip")
    parser.add_argument("--trail", type=float, default=0.0, help="seconds of the last face after the clip")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument(
        "--no-amplify",
        action="store_true",
        help="draw raw FKCN mixtures instead of stretching the strongest weight",
    )
    parser.add_argument(
        "--no-prefer-raw",
        action="store_true",
        help="read valence/arousal even if raw_valence/raw_arousal exist",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        self_check()
        print("self-test ok")
        return 0

    single = args.valence is not None or args.arousal is not None
    if single and args.csv is not None:
        parser.error("pass either --csv or --valence/--arousal, not both")
    if single and (args.valence is None or args.arousal is None):
        parser.error("--valence and --arousal are both required")
    if not single and args.csv is None:
        parser.error("pass --csv, or --valence and --arousal, or --self-test")

    if single:
        weights = va_to_weights(args.valence, args.arousal)
        names = ("joy", "surprise", "fear", "sadness", "disgust", "anger")
        print(" ".join(f"{n}={w:.4f}" for n, w in zip(names, weights)))
        if args.out is None and args.weights_out is None:
            return 0
        if args.weights_out is not None:
            args.weights_out.parent.mkdir(parents=True, exist_ok=True)
            header = ",".join(names)
            values = ",".join(f"{w:.6f}" for w in weights)
            args.weights_out.write_text(header + "\n" + values + "\n", encoding="utf-8")
            print(f"wrote {args.weights_out}")
        if args.out is not None:
            from . import render_png

            render_png(
                args.valence,
                args.arousal,
                str(args.out),
                width=args.width,
                height=args.height,
                amplify=not args.no_amplify,
            )
            print(f"wrote {args.out}")
        return 0

    valence, arousal, timestamps, tokens = load_va_csv(
        args.csv, prefer_raw=not args.no_prefer_raw
    )
    if args.rows_out is not None:
        mapped = map_rows(
            valence, arousal, timestamps, tokens, alpha=args.alpha
        )
        mapped.save_csv(args.rows_out)
        print(f"wrote {args.rows_out} ({len(mapped.weights)} rows)")

    if args.out is None and args.weights_out is None:
        if args.rows_out is None:
            parser.error("pass -o and/or --weights-out")
        return 0

    # Weights-only: skip Chromium.
    if args.out is None:
        from .trajectory import display_trajectory

        traj = display_trajectory(
            valence,
            arousal,
            timestamps,
            alpha=args.alpha,
            lead_s=args.lead,
            trail_s=args.trail,
        )
        traj.save_csv(args.weights_out)
        print(
            f"wrote {args.weights_out} ({len(traj.weights)} frames, "
            f"{traj.times[-1]:.2f}s, rate-limit {traj.rate_limit_fraction:.2f})"
        )
        return 0

    if args.out.suffix.lower() != ".mp4":
        parser.error("--out for a CSV must be an .mp4 path")

    from . import render_video

    render_video(
        str(args.csv),
        str(args.out),
        alpha=args.alpha,
        lead_s=args.lead,
        trail_s=args.trail,
        width=args.width,
        height=args.height,
        amplify=not args.no_amplify,
        prefer_raw=not args.no_prefer_raw,
        weights_out=str(args.weights_out) if args.weights_out else None,
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
