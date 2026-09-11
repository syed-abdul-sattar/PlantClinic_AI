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

## Supported Plants & Diseases

The final PlantClinic AI disease classifier contains **32 supported classes**.

### Supported Plants

- Apple
- Blueberry
- Cherry
- Corn (maize)
- Grape
- Peach
- Pepper (bell)
- Potato
- Raspberry
- Soybean
- Squash
- Strawberry
- Tomato

### Supported Diseases / Conditions

#### Apple
- Apple Scab
- Black Rot
- Healthy

#### Blueberry
- Healthy

#### Cherry
- Powdery Mildew

#### Corn (Maize)
- Cercospora Leaf Spot / Gray Leaf Spot
- Common Rust
- Northern Leaf Blight
- Healthy

#### Grape
- Black Rot
- Esca (Black Measles)
- Leaf Blight (Isariopsis Leaf Spot)

#### Peach
- Bacterial Spot
- Healthy

#### Pepper (Bell)
- Healthy

#### Potato
- Early Blight
- Late Blight

#### Raspberry
- Healthy

#### Soybean
- Healthy

#### Squash
- Powdery Mildew

#### Strawberry
- Leaf Scorch
- Healthy

#### Tomato
- Bacterial Spot
- Early Blight
- Healthy
- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites (Two-spotted Spider Mite)
- Target Spot
- Tomato Yellow Leaf Curl Virus
- Tomato Mosaic Virus

The exact final class-to-index mapping is stored in:

`environment/class_mapping.json`

## Multi-Image Diagnosis

PlantClinic AI is designed to analyze **up to 3 images per diagnosis session**.

Each supplied image is processed individually before the system combines the valid results into a final consensus.

### If only one image is supplied

The system can still produce a diagnosis, but it explicitly warns that **multi-image consensus reliability may be limited** because only one image is available.

### If two images are supplied

Both images can be analyzed and their evidence can contribute to the final consensus.

### If three images are supplied

All three images can be analyzed, providing the intended maximum amount of multi-image evidence for one session.

### If more than three images are supplied

Only the first three images are used for analysis. Additional images are ignored and reported in the image-limit notice.

For example, if 6 images are uploaded:

```text
Uploaded:  Image 1, Image 2, Image 3, Image 4, Image 5, Image 6

Analyzed: Image 1, Image 2, Image 3
Ignored:  Image 4, Image 5, Image 6
```

### Rejected / Invalid Images

An uploaded image may be excluded if it cannot be validated as a suitable plant image or if image processing rejects it.

Excluded images are reported with their image number and reason rather than silently being treated as valid disease evidence.

For example:

```text
Total Images Uploaded        : 3
Successfully Analyzed Images : 2
Excluded Images              : 1

Image 3
Reason : The uploaded image could not be validated as a plant image.
```

## Consensus Interpretation

The final diagnosis is calculated from the successfully analyzed images.

| Consensus Strength | Agreement |
|---|---:|
| HIGH | >= 80% |
| MODERATE | >= 60% |
| LOW | < 60% |

The system also identifies images that disagree with the final consensus as outliers and reports them separately.

### Important

More images do not automatically guarantee a correct diagnosis.

The final prediction depends on:

- image quality,
- whether the image passes the system's plant-image validation,
- the diseases represented in the trained model,
- and the model's learned classification capability.

PlantClinic AI should therefore be treated as an **AI-assisted plant disease diagnosis system**, not as a replacement for expert agricultural diagnosis.

## Supported-Class Limitation

PlantClinic AI is a **closed-set classifier**.

It can classify images among the disease/healthy classes represented by its final 32-class model.

It should not be interpreted as a general-purpose detector capable of identifying every possible plant species or every possible plant disease.

If a plant disease is not represented in the trained classes, the system may not be able to provide a reliable identification.

For the complete list of model classes, refer to:

`environment/class_mapping.json`
