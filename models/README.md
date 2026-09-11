# PlantClinic AI Model Weights

The final trained model weights are hosted separately from GitHub because large binary checkpoints should not be stored directly in the source repository.

## Hugging Face Model Repository

https://huggingface.co/sasattar/PlantClinic_AI_Models

The repository contains:

PlantClinic_AI_Models/
├── disease_model/
│   └── SwinLarge_384_ROBUST_FINAL.pth
└── leaf_gate/
    └── mobilenetv3_leaf_gate_v2.pth

## 1. Disease Classifier

Architecture:

**Swin Transformer Large**

Input resolution:

**384 × 384**

Checkpoint:

`SwinLarge_384_ROBUST_FINAL.pth`

Place it at:

`models/disease_model/SwinLarge_384_ROBUST_FINAL.pth`

## 2. Leaf Validation Model

Architecture:

**MobileNetV3 Small**

Task:

**Leaf / non-leaf validation**

Checkpoint:

`mobilenetv3_leaf_gate_v2.pth`

Place it at:

`models/leaf_gate/mobilenetv3_leaf_gate_v2.pth`

## Download Instructions

1. Open the Hugging Face model repository.
2. Download both `.pth` files.
3. Create the following directories inside the cloned project:

`models/disease_model/`
`models/leaf_gate/`

4. Place each checkpoint in its corresponding directory.
5. Do not rename the files.

## Why The Weights Are Not In GitHub

The final disease checkpoint is approximately 745 MB.

Large binary model files are intentionally excluded from this Git repository through `.gitignore`.

The source code and documentation remain on GitHub, while the trained model artifacts are hosted on Hugging Face.

## Experimental Checkpoints

Experimental and historical model work is documented under:

`experiments/`

The experimental checkpoints are not required to run the final production pipeline.

## Final Production Components

The verified production pipeline uses:

- MobileNetV3 Small for leaf/non-leaf validation
- Swin Transformer Large at 384 × 384 for disease classification
- the verified class mapping
- the existing image-quality and confidence/uncertainty processing
- severity estimation
- multi-image consensus

The production implementation is located under:

`src/pipeline/`