# PlantClinic AI

PlantClinic AI is an AI-assisted plant disease diagnosis system using deep learning and multi-image consensus.

## Production Pipeline

1. MobileNetV3 Small - leaf / non-leaf validation
2. Swin Transformer Large (384x384) - plant disease classification
3. Image preprocessing and quality validation
4. Severity estimation
5. Multi-image consensus
6. Final diagnostic report

## Active Models

Leaf Gate:
- MobileNetV3 Small
- mobilenetv3_leaf_gate_v2.pth

Disease Classifier:
- Swin Transformer Large
- Input resolution: 384x384
- SwinLarge_384_ROBUST_FINAL.pth

The large disease-model checkpoint is stored separately because of its size.

## Disease Classes

The class mapping used by the pipeline is:

environment/class_mapping.json

## Main Source

src/pipeline/plant_clinic_system.py
src/pipeline/plant_clinic_consensus.py

## Large Files

Large datasets, backup models, research checkpoints, and archived development files are intentionally excluded from this repository.

## Status

This repository contains the cleaned, production-focused version of PlantClinic AI.
