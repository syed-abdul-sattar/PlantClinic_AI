"""
PlantClinic AI — Reproducible Demo Runner

This file provides a simple entry point for running the verified
production pipeline from a cloned repository.

Usage:
    python run_demo.py image1.jpg
    python run_demo.py image1.jpg image2.jpg image3.jpg
"""

from pathlib import Path
import argparse
import sys


# ------------------------------------------------------------------
# Project root
# ------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ------------------------------------------------------------------
# Production pipeline
# ------------------------------------------------------------------

from src.pipeline.plant_clinic_system import PlantClinicSystem
from src.pipeline.plant_clinic_consensus import PlantClinicConsensusSystem


# ------------------------------------------------------------------
# Required project files
# ------------------------------------------------------------------

DISEASE_MODEL = (
    REPO_ROOT
    / "models"
    / "disease_model"
    / "SwinLarge_384_ROBUST_FINAL.pth"
)

LEAF_MODEL = (
    REPO_ROOT
    / "models"
    / "leaf_gate"
    / "mobilenetv3_leaf_gate_v2.pth"
)

CLASS_MAPPING = (
    REPO_ROOT
    / "environment"
    / "class_mapping.json"
)


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

def validate_files(image_paths):
    required_files = {
        "Disease model": DISEASE_MODEL,
        "Leaf model": LEAF_MODEL,
        "Class mapping": CLASS_MAPPING,
    }

    missing = []

    for name, path in required_files.items():
        if not path.exists():
            missing.append(f"{name}: {path}")

    if missing:
        print()
        print("Missing required project files:")

        for item in missing:
            print("  -", item)

        print()
        print(
            "Please follow docs/RUNNING_THE_PROJECT.md "
            "to download the final model weights."
        )

        raise SystemExit(1)

    missing_images = [
        str(path)
        for path in image_paths
        if not path.exists()
    ]

    if missing_images:
        print()
        print("Missing input image(s):")

        for path in missing_images:
            print("  -", path)

        raise SystemExit(1)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run PlantClinic AI multi-image "
            "plant disease diagnosis."
        )
    )

    parser.add_argument(
        "images",
        nargs="+",
        help="One to three input image paths."
    )

    args = parser.parse_args()

    if not 1 <= len(args.images) <= 3:
        parser.error(
            "Provide between 1 and 3 images."
        )

    image_paths = [
        Path(image).expanduser().resolve()
        for image in args.images
    ]

    validate_files(image_paths)

    print("=" * 70)
    print("PLANTCLINIC AI")
    print("=" * 70)
    print("Loading production models...")

    base_system = PlantClinicSystem(
        disease_model_path=str(DISEASE_MODEL),
        class_mapping_path=str(CLASS_MAPPING),
        leaf_model_path=str(LEAF_MODEL),
        upsampler=None,
    )

    consensus_system = PlantClinicConsensusSystem(
        base_system
    )

    print("Models loaded.")
    print(f"Images supplied: {len(image_paths)}")
    print("=" * 70)

    result = consensus_system.run_multi_image(
        [str(path) for path in image_paths]
    )

    consensus_system.display_result(result)


if __name__ == "__main__":
    main()
