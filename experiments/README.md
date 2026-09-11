# PlantClinic AI — Research Experiments

This directory preserves selected research and development artifacts from the development of PlantClinic AI.

These files document the iterative engineering and research process that led to the final system. They are not all components of the final production pipeline.

## Why This Directory Exists

PlantClinic AI was developed through repeated research, implementation, training, testing, evaluation, and redesign.

The development process involved:

1. Identifying a problem
2. Researching possible approaches
3. Implementing an approach
4. Training or testing it
5. Evaluating its limitations
6. Redesigning the approach when necessary
7. Retesting
8. Integrating components that met the project requirements

## Directory Structure

```text
experiments/
├── disease_models/
│   ├── training_log.csv
│   ├── training_log_stage2.csv
│   └── production_training_log.csv
├── segmentation/
│   ├── sam/
│   │   └── leaf_pseudo_label_generator.py
│   └── segformer/
│       └── config.json
├── legacy_pipeline/
│   └── plant_clinic_core.py
└── environmental_prototype/
    └── geo_specialist.py
```

## Disease Model Experiments

The disease-classification development included multiple model generations:

```text
ResNet18
   ↓
ResNet50
   ↓
ViT
   ↓
ViT384
   ↓
Swin Transformer
   ↓
Swin Large 384
```

The training logs preserved here provide evidence of the iterative training and evaluation process.

## SAM Segmentation / Pseudo-Label Experiment

The SAM experiment investigated automated leaf-region extraction and pseudo-label generation.

The preserved script is:

`segmentation/sam/leaf_pseudo_label_generator.py`

This was part of research into improving leaf detection and segmentation.

**SAM should not be interpreted as part of the final production inference pipeline.**

## SegFormer Experiment

The SegFormer configuration represents experimentation with semantic segmentation and refinement of leaf regions.

This experiment investigated whether segmentation could improve the downstream disease-diagnosis pipeline.

## Legacy Pipeline

`legacy_pipeline/plant_clinic_core.py` is preserved as a historical implementation.

It helps document how the system evolved before the final production-focused pipeline was consolidated.

It is not the primary entry point for the current project.

The current production pipeline is located under:

`src/pipeline/`

## Environmental Prototype

`environmental_prototype/geo_specialist.py` represents an experimental direction involving environmental and geospatial information.

It is preserved because it was part of the broader development exploration.

It is not part of the demonstrated final disease-diagnosis pipeline.

## Experimental vs Production Components

A critical distinction is maintained throughout this repository.

### Historical / Experimental

Examples include:

- ResNet18
- ResNet50
- ViT variants
- SAM
- SegFormer
- segmentation and cropping experiments
- alternative image-processing approaches
- older pipeline implementations
- environmental prototypes
- training experiments

These components document research history.

### Final / Verified

The final production-focused system uses:

- MobileNetV3 Small for leaf/non-leaf validation
- Swin Large Patch4 Window12 at 384×384 for disease classification
- the verified class mapping
- image-quality processing
- uncertainty and confidence handling
- severity estimation
- multi-image consensus
- final diagnostic reporting

The production pipeline is implemented under:

`src/pipeline/`

## Why Failed Experiments Are Preserved

An unsuccessful experiment can still be valuable when it answers an engineering or research question.

For PlantClinic AI, experiments were used to investigate:

- whether a particular architecture was suitable
- whether an input-validation strategy was too aggressive
- whether image enhancement provided meaningful benefit
- whether segmentation improved the complete pipeline
- whether explainability methods were practical
- how uncertainty should be handled
- how multiple images could be combined reliably

The repository therefore distinguishes between:

```text
What was tried
```

and:

```text
What was ultimately selected
```

## Model Weights

Large experimental model weights are intentionally not stored in this Git repository.

This keeps the source repository manageable and avoids mixing historical checkpoints with final production artifacts.

Final model weights will be hosted separately and linked from the project documentation.

## Reproducibility

The goal of this directory is not to guarantee that every historical experiment can be executed immediately.

Instead, it preserves important source-level and training-history evidence needed to understand how the project evolved.

The reproducible path for the final system will be documented separately.

## Important Note

The presence of a file in `experiments/` does **not** mean that the file is currently imported or executed by the production pipeline.

For the current architecture, refer to:

- `README.md`
- `src/pipeline/`
- `docs/DEVELOPMENT_JOURNEY.md`

The development journey explains why the architecture changed; the production pipeline shows what currently runs.