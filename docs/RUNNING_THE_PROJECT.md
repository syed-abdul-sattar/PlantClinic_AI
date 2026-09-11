# Running PlantClinic AI

This guide explains how to run the final production-focused PlantClinic AI system from a fresh Git clone.

## 1. Clone the repository

```bash
git clone https://github.com/syed-abdul-sattar/PlantClinic_AI.git
cd PlantClinic_AI
```

## 2. Create a Python environment

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
```

Activate the environment.

Windows:

```powershell
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU execution, install the appropriate PyTorch build for the available CUDA environment when required.

## 4. Download the final model weights

The final trained weights are hosted on Hugging Face:

https://huggingface.co/sasattar/PlantClinic_AI_Models

Download these two files:

- `disease_model/SwinLarge_384_ROBUST_FINAL.pth`
- `leaf_gate/mobilenetv3_leaf_gate_v2.pth`

Place them inside the cloned repository as follows:

```text
PlantClinic_AI/
└── models/
    ├── disease_model/
    │   └── SwinLarge_384_ROBUST_FINAL.pth
    └── leaf_gate/
        └── mobilenetv3_leaf_gate_v2.pth
```

Do not rename the files.

## 5. Run the project

The repository provides `run_demo.py` as the reproducible entry point.

For one image:

```bash
python run_demo.py image.jpg
```

For multiple images:

```bash
python run_demo.py image1.jpg image2.jpg image3.jpg
```

The final consensus system accepts up to three images.

## 6. Production pipeline

The runner creates the existing production pipeline:

1. `PlantClinicSystem`
2. `PlantClinicConsensusSystem`

The verified production architecture uses:

- MobileNetV3 Small for leaf/non-leaf validation
- Swin Transformer Large at 384 × 384 for disease classification
- the verified class mapping
- existing image-quality processing
- confidence and uncertainty handling
- severity estimation
- multi-image consensus
- final diagnostic reporting

## 7. Required files

The runner expects these files:

```text
models/disease_model/SwinLarge_384_ROBUST_FINAL.pth
models/leaf_gate/mobilenetv3_leaf_gate_v2.pth
environment/class_mapping.json
```

If a required model is missing, `run_demo.py` stops with an explicit message.

## 8. Input images

One to three image paths can be supplied.

Images that fail the existing plant-image validation can be excluded according to the production pipeline logic.

## 9. Production source

The actual production implementation remains in:

```text
src/pipeline/plant_clinic_system.py
src/pipeline/plant_clinic_consensus.py
```

`run_demo.py` only provides the local file paths and command-line entry point. It does not replace the production pipeline.

## 10. Research history

Historical experiments are preserved under:

```text
experiments/
```

The complete development journey is documented in:

```text
docs/DEVELOPMENT_JOURNEY.md
```

Experimental files are preserved for research history and should not automatically be interpreted as active production components.

## 11. Large datasets

The original research datasets are intentionally not stored in GitHub.

This repository distributes the production source and documentation without requiring a 100 GB dataset checkout.

## 12. Model hosting

Final model weights:

https://huggingface.co/sasattar/PlantClinic_AI_Models

GitHub source repository:

https://github.com/syed-abdul-sattar/PlantClinic_AI

## 13. Important reproducibility note

The final weights and source code are separated intentionally:

```text
GitHub
  -> source code + documentation + research history

Hugging Face
  -> final trained model weights
```

A fresh-clone test should be completed before deleting the original research datasets or model files from Google Drive.