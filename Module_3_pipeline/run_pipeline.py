"""
run_pipeline.py
===============
Runs the full VisionX Sinhala Comic Audio-Video Narrative pipeline
end to end in a single command.

Stages:
  1. OCR Post-Correction   → corrected Excel output
  2. Emotion Detection     → emotion Excel + master.json updated
  3. TTS Generation        → audio MP3s + TTS Excel + master.json updated
  4. Video Generation      → final MP4

Usage:
    python run_pipeline.py

    # Run specific stages only:
    python run_pipeline.py --stages ocr emotion
    python run_pipeline.py --stages tts video

    # Skip a stage:
    python run_pipeline.py --skip ocr
"""

import sys
import time
import argparse
import traceback
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


# ── Stage definitions ─────────────────────────────────────────────────────────

def run_ocr():
    print("\n" + "█" * 55)
    print("  STAGE 1 — OCR Post-Correction")
    print("█" * 55)
    from runners.run_ocr_post_correction import main
    main()


def run_emotion():
    print("\n" + "█" * 55)
    print("  STAGE 2 — Emotion Detection")
    print("█" * 55)
    from runners.run_emotion_detection import main
    main()


def run_tts():
    print("\n" + "█" * 55)
    print("  STAGE 3 — TTS Generation")
    print("█" * 55)
    from runners.run_tts import main
    main()


def run_video():
    print("\n" + "█" * 55)
    print("  STAGE 4 — Video Generation")
    print("█" * 55)
    from runners.run_video import main
    main()


STAGES = {
    "ocr":     run_ocr,
    "emotion": run_emotion,
    "tts":     run_tts,
    "video":   run_video,
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VisionX full pipeline runner"
    )
    parser.add_argument(
        "--stages", nargs="+",
        choices=list(STAGES.keys()),
        default=list(STAGES.keys()),
        help="Stages to run (default: all)"
    )
    parser.add_argument(
        "--skip", nargs="+",
        choices=list(STAGES.keys()),
        default=[],
        help="Stages to skip"
    )
    args = parser.parse_args()

    stages_to_run = [s for s in args.stages if s not in args.skip]

    print("=" * 55)
    print("  VisionX — Sinhala Comic Audio-Video Pipeline")
    print("=" * 55)
    print(f"\nStages to run: {' → '.join(stages_to_run)}")

    results  = {}
    t_start  = time.time()

    for stage in stages_to_run:
        t0 = time.time()
        try:
            STAGES[stage]()
            elapsed       = time.time() - t0
            results[stage] = ("✓", elapsed)
            print(f"\n  [DONE] {stage} completed in {elapsed:.1f}s")
        except Exception as e:
            elapsed        = time.time() - t0
            results[stage] = ("✗", elapsed)
            print(f"\n  [ERROR] {stage} failed after {elapsed:.1f}s")
            print(f"  {type(e).__name__}: {e}")
            traceback.print_exc()
            print(f"\n  Pipeline stopped at stage: {stage}")
            print("  Fix the error above and re-run with:")
            remaining = stages_to_run[stages_to_run.index(stage):]
            print(f"  python run_pipeline.py --stages {' '.join(remaining)}")
            break

    # ── Summary ───────────────────────────────────────────────────────────────
    total = time.time() - t_start
    print("\n" + "=" * 55)
    print("  PIPELINE SUMMARY")
    print("=" * 55)
    for stage, (status, elapsed) in results.items():
        print(f"  {status}  {stage:<12} {elapsed:>6.1f}s")
    print(f"\n  Total time: {total:.1f}s")

    if all(s == "✓" for s, _ in results.values()):
        print("\n  All stages completed successfully.")
        print(f"  Video → outputs/video/comic_narrative.mp4")
    else:
        print("\n  Pipeline did not complete — see errors above.")


if __name__ == "__main__":
    main()