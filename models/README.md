# Model Files

PlantClinic AI uses two trained models in the demonstrated Phase 1 pipeline.

## 1. Leaf Gatekeeper

Architecture:
- MobileNetV3 Small

Purpose:
- Classifies the uploaded image as a leaf or non-leaf image.

Checkpoint:
`mobilenetv3_leaf_gate_v2.pth`

## 2. Disease Classifier

Architecture:
- Swin Transformer Large
- Input resolution: 384 × 384

Checkpoint:
`SwinLarge_384_ROBUST_FINAL.pth`

The disease checkpoint is intentionally not included in this repository because of its large file size.

The path used during the current Colab deployment is:

`/content/drive/MyDrive/Major_Project/models/SwinLarge_384_ROBUST_FINAL.pth`

For public deployment, obtain/store the checkpoint through the project’s documented model distribution method.
