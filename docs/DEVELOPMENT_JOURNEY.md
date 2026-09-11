# PlantClinic AI — Complete Development Journey

## Purpose of This Document

This document records the development history of PlantClinic AI.

The purpose is not only to describe the final system, but to preserve the reasoning, experiments, failures, redesigns, model changes, dataset investigations, and engineering decisions that led to the current version.

PlantClinic AI was developed iteratively.

The general development cycle was:

Research
→ Experiment
→ Implement
→ Train
→ Evaluate
→ Identify limitations
→ Redesign
→ Retest
→ Integrate

This history is important because several models and modules exist in the project that were investigated during development but are not part of the verified final production-focused pipeline.

---

# 1. Original Project Objective

The original objective was not simply to train an image classifier.

The project was designed around the idea of building an AI-assisted plant disease diagnosis system that could eventually handle more realistic agricultural images.

The intended system needed to address problems that a simple image classifier does not solve:

- invalid or non-plant images
- non-leaf images
- poor image quality
- blur
- lighting variation
- background noise
- multiple leaves
- uncertainty
- disease classification
- explainability
- severity estimation
- multiple images of the same plant

The project therefore evolved into a multi-stage system instead of remaining a single:

Image → Model → Disease

pipeline.

---

# 2. The First Major Challenge: Real Images Are Different From Clean Datasets

One of the first important observations was that plant-disease datasets can be much cleaner than images encountered in real agricultural environments.

PlantVillage became an important dataset during development.

Many PlantVillage images have:

- relatively clean backgrounds
- centered leaves
- controlled conditions
- clear disease symptoms
- limited environmental noise

This is useful for training and benchmarking, but high performance on such data does not automatically mean that the system will behave reliably on arbitrary field photographs.

This led to a major design principle:

> High classification accuracy on a clean dataset is not enough to build a robust plant-disease diagnosis system.

This realization influenced almost every later development decision.

---

# 3. Dataset Investigation and Cleaning

A significant amount of project work involved understanding what data was actually available rather than blindly training on everything stored in the project folders.

The project eventually contained several different categories of data:

### Disease classification data

Used for plant-disease classification.

### Leaf / non-leaf data

Used to determine whether an uploaded image contains a plant leaf.

### Background / negative images

Used to expose the validation system to images containing objects other than leaves.

Examples found in the project included:

- animals
- cars
- people
- buildings
- other non-leaf content

### Segmentation data

A separate collection of images and masks was investigated for leaf segmentation.

The project records show approximately 535 segmentation images/masks.

### Field / real-world images

Additional real-world images were investigated to understand how the system behaved outside controlled dataset conditions.

---

# 4. Discovery of Dataset Imbalance

One of the most important dataset problems discovered during development was class imbalance in the leaf/non-leaf dataset.

A dataset scan recorded approximately:

21,411 leaf images

and a much smaller number of non-leaf images.

Another development record reported approximately 316 non-leaf images, while earlier experiments contained different counts.

The exact count changed as datasets were expanded and modified.

The important discovery was not the exact number.

The important discovery was:

> The leaf/non-leaf training data was severely imbalanced.

This caused a serious failure mode.

The leaf classifier could learn that predicting "leaf" most of the time was a very easy way to obtain high apparent accuracy.

This led to the decision to deliberately increase negative examples and investigate hard negatives.

---

# 5. Leaf / Non-Leaf Classifier — First Approach

The first leaf-detection experiments used a ResNet18 binary classifier.

The task was:

Leaf
vs
Non-Leaf

The motivation was straightforward:

If an uploaded image is not a plant leaf, it should not immediately be passed to the disease classifier.

The first experiments revealed the importance of dataset balance.

A development record specifically notes that the initial classifier suffered from severe class imbalance and could effectively predict everything as leaf.

This was a major failure because high training accuracy would be misleading if the negative class was poorly represented.

---

# 6. Building a Better Negative Dataset

After observing the imbalance problem, the project investigated additional non-leaf sources.

These included:

- COCO/background images
- manually collected non-leaf images
- hard negative examples
- other natural/background images

The idea behind hard negatives was important.

Random non-leaf images are easy to reject.

The difficult cases are images that visually resemble plant images but should not be treated as valid disease-classification inputs.

Examples include:

- backgrounds
- objects
- photographs with vegetation but no useful leaf
- diagrams
- slides
- visually confusing images

The development therefore moved from simply asking:

"Can the model recognize a leaf?"

toward:

"Can the system safely determine whether an image contains useful plant-leaf evidence?"

---

# 7. Biological Validation / Bio-Gate

A second validation concept was introduced in addition to learned leaf classification.

The Bio-Gate used image characteristics intended to determine whether an image looked biologically consistent with a plant leaf.

The investigated signals included:

- green ratio
- edge density
- brightness

Green ratio was investigated using colour-space thresholding.

Edge density used image edges as a structural signal.

Brightness was used to identify images that were excessively dark.

The intention was to create a second line of protection before disease classification.

The conceptual pipeline became:

Image
↓
Leaf Classifier
↓
Bio-Gate
↓
Disease Classification

---

# 8. Failure of the Initial Hard Bio-Gate

The Bio-Gate introduced an important problem.

The filters were initially treated too much like hard rules.

For example:

if a condition failed → reject the image

This looked attractive from a safety perspective, but real plant images do not always satisfy simple visual assumptions.

A genuine leaf can:

- have unusual colours
- be photographed under poor lighting
- contain shadows
- contain disease-related discoloration
- have low green content
- have unusual backgrounds

The development records explicitly identify the Bio-Gate as too aggressive and note that valid leaves could be rejected.

This was one of the major lessons of the project:

> A safety filter can itself become a source of false rejection.

---

# 9. Hard Filtering vs Scoring

The excessive rejection problem led to a different design idea.

Instead of treating validation as only:

PASS / FAIL

the project investigated a scoring approach.

The proposed concept combined signals such as:

leaf classifier probability

and

biological feature score

into a combined score.

One documented experimental formulation was:

combined_score =
0.6 × leaf_score +
0.4 × bio_score

with experimental decision regions:

score < 0.25
→ reject

0.25–0.45
→ uncertain

> 0.45
→ pass

These values belong to the documented experimental redesign and should not automatically be interpreted as the final calibrated thresholds.

The important development decision was the move toward graded evidence instead of blindly applying independent hard filters.

---

# 10. Disease Model — ResNet18

After establishing an initial classification baseline, ResNet18 was investigated for disease classification.

This provided a conventional CNN baseline.

The purpose was not to immediately choose the largest possible architecture.

The model was trained, tested, and evaluated so that its behaviour could be compared with later architectures.

The development records describe the ResNet18 stage as successful but eventually limited by a performance plateau.

This created the motivation for the next experiment.

---

# 11. Disease Model — ResNet50

ResNet50 was then investigated as a deeper CNN architecture.

The goal was to determine whether increasing CNN depth and representational capacity would provide a meaningful improvement.

The experiment was part of the architecture comparison process.

The development notes indicate that ResNet50 remained limited compared with the later Transformer-based approaches.

Therefore the project continued toward Transformer architectures.

---

# 12. Moving From CNNs to Transformers

The next major architectural decision was to investigate Vision Transformers.

This was a substantial change in the model-development direction.

Instead of relying only on convolutional feature extraction, the project investigated Transformer-based visual representations.

Multiple ViT training runs were performed.

The project contains records/checkpoints corresponding to multiple training stages, including:

vit_epoch_1
vit_epoch_2
...
vit_epoch_29

as well as:

vit_best

and later higher-resolution ViT experiments.

The existence of these checkpoints reflects that the project did not simply train one model once.

Multiple experiments were performed and evaluated.

---

# 13. ViT 384 Experiments

The project then investigated a higher-resolution ViT configuration.

The motivation was to preserve more visual detail.

Plant-disease symptoms can sometimes occupy relatively small portions of a leaf.

Therefore the question was:

Would higher input resolution allow the model to retain more disease-related information?

Multiple ViT384 training stages were recorded.

Examples included:

vit384_epoch_30
through
vit384_epoch_48

and a best checkpoint.

The ViT experiments improved the research understanding of the problem but were ultimately not selected as the final disease architecture.

---

# 14. Why the Model Evolution Matters

The model history can therefore be summarized as:

ResNet18
↓
ResNet50
↓
Vision Transformer
↓
ViT 384
↓
Swin Transformer
↓
Swin Large 384
↓
Robust Final Checkpoint

This is not simply a list of models.

Each transition represented a new hypothesis followed by training and evaluation.

The project therefore contains a genuine model-evolution study:

CNN
→
deeper CNN
→
Transformer
→
higher-resolution Transformer
→
hierarchical Transformer

This is one of the most important parts of the research history.

---

# 15. Swin Transformer Experiments

The project then moved to the Swin Transformer family.

The final architecture investigated was:

swin_large_patch4_window12_384

with an input resolution of:

384 × 384

Multiple Swin checkpoints existed during development.

Examples include:

SwinLarge_384_BEST.pth

SwinLarge_384_FINAL_99PC.pth

SwinLarge_384_ROBUST_FINAL.pth

The final verified active disease checkpoint is:

SwinLarge_384_ROBUST_FINAL.pth

The active pipeline creates the model using the Swin Large Patch4 Window12 384 architecture.

The final class mapping contains 32 classes.

---

# 16. Accuracy and the Clean-Dataset Problem

One development record reports approximately 99.78% accuracy for the Swin model on the cleaned dataset.

The project also generated confusion matrices showing a strong diagonal and relatively low observed misclassification on that evaluation.

However, an important limitation was recognized:

> Clean-dataset performance can be biased toward the conditions represented by the dataset.

Therefore, the project did not treat a high dataset score as proof that the system was automatically robust in the field.

This distinction became one of the reasons for developing:

- leaf validation
- image-quality analysis
- uncertainty handling
- confidence thresholds
- multiple-image consensus

The reported accuracy should therefore be understood as a dataset evaluation result, not as a universal real-world accuracy claim.

---

# 17. Image Quality Became a Separate Problem

After working on disease classification, another major problem became clear.

A classifier may perform well when an image is clear, but real images can contain:

- blur
- poor focus
- poor lighting
- low contrast
- shadows
- background noise

Therefore image quality was introduced as another stage of the system.

---

# 18. Blur Detection

The project investigated blur detection using Laplacian variance.

The basic concept was:

cv2.Laplacian(image).var()

The recorded experimental ranges included:

< 50
→ severe blur

< 120
→ moderate blur

> 120
→ acceptable

These thresholds were part of the image-quality experiments and were used to determine whether additional processing should be considered.

---

# 19. CLAHE Enhancement

CLAHE was investigated for moderate image-quality problems.

The processing approach included:

RGB
↓
LAB
↓
CLAHE on L channel
↓
Merge channels
↓
Sharpen

The purpose was to improve local contrast without applying expensive enhancement to every image.

The system therefore attempted to make enhancement conditional on image quality.

---

# 20. ESRGAN / Super-Resolution Experiment

RealESRGAN was also investigated.

The model used during experimentation was:

RealESRGAN_x4plus

The goal was to determine whether heavily degraded images could benefit from super-resolution before disease classification.

The experiments showed an important practical limitation:

Many plant images were already sufficiently high resolution.

Therefore super-resolution did not always provide a meaningful benefit relative to its additional computational cost.

This resulted in an important design decision:

> Enhancement should be conditional rather than automatically applied to every image.

In the current verified demonstration run, the system is initialized with:

upsampler=None

Therefore ESRGAN exists as an optional capability in the code, but it was not active in the demonstrated final run.

---

# 21. Test-Time Augmentation

The disease prediction process also investigated Test-Time Augmentation.

The recorded transformations included:

- original image
- horizontal flip
- brightness adjustment
- contrast adjustment
- small rotation

Predictions from the different views were averaged.

The objective was to make the final prediction less dependent on a single exact presentation of the image.

This became part of the inference strategy investigated during development.

---

# 22. Confidence Threshold Experiments

The project then moved beyond simply returning the highest-probability class.

Dynamic confidence thresholds were investigated.

Experimental examples recorded during development included:

healthy
→ 90%

virus
→ 95%

blight/rot
→ 92%

other classes
→ 85%

If confidence was below the applicable threshold:

UNCERTAIN

Otherwise:

SUCCESS

The reasoning was that not every prediction should automatically be presented as a confident diagnosis.

---

# 23. Large-Scale Evaluation Exposed Another Failure

The system was evaluated using multiple datasets and image categories.

The recorded evaluation groups included:

COCO
→ non-leaf rejection

PlantVillage
→ clean/lab-style leaf evaluation

PlantDoc
→ real-world/field-style evaluation

One logged evaluation result reported:

COCO:
SUCCESS = 605
UNCERTAIN = 1496
REJECTED = 705

PlantVillage:
SUCCESS = 713
UNCERTAIN = 1683
REJECTED = 710

PlantDoc:
SUCCESS = 807
UNCERTAIN = 1775
REJECTED = 760

The important observation was not simply the numbers.

The important observation was:

> Too many images were becoming uncertain or rejected.

This exposed weaknesses in the earlier hard-threshold design.

---

# 24. Failure Analysis

The development records identified several causes:

### Problem 1 — Bio-Gate too strict

Valid plant images could be rejected.

### Problem 2 — Leaf threshold calibration

The binary leaf classifier's threshold was not always appropriate.

### Problem 3 — Hard rejection

The system could reject an image immediately after one validation condition failed.

This produced excessive rejection.

The resulting philosophy became:

> Reject obvious invalid images, but do not reject uncertain cases too aggressively.

---

# 25. Segmentation Research

Another major branch of the project investigated leaf segmentation.

The motivation was straightforward.

Real agricultural photographs may contain:

- several leaves
- branches
- soil
- sky
- background objects
- overlapping plant structures

If the disease classifier receives the entire image, it may use background information instead of disease-specific evidence.

Therefore the project investigated whether explicit leaf segmentation could improve the pipeline.

---

# 26. SAM Experiment

Segment Anything Model (SAM) was investigated as a possible way of generating object masks.

A SAM checkpoint existed in the project.

The idea was:

Image
↓
SAM
↓
Candidate object masks
↓
Identify leaf regions

The purpose was to isolate useful plant regions before disease classification.

---

# 27. SegFormer Experiment

SegFormer was also investigated.

The role considered for SegFormer was semantic segmentation and mask refinement.

The proposed concept became:

SAM
↓
Candidate masks
↓
SegFormer
↓
Leaf mask refinement
↓
Leaf extraction

The project therefore explored a hybrid segmentation strategy rather than relying on only one segmentation approach.

---

# 28. Leaf Cropping / Instance Extraction

After segmentation, the project investigated extracting individual leaf regions.

The intended workflow was:

Plant image
↓
Segmentation
↓
Leaf masks
↓
Connected components / regions
↓
Leaf 1
Leaf 2
Leaf 3
↓
Disease classification per leaf

This was motivated by the observation that real farm photographs can contain multiple leaves.

---

# 29. Segmentation Failure / Non-Integration

The segmentation work was valuable as an experiment, but it did not become part of the verified production-focused pipeline.

The reason for documenting this distinction is important.

A module can be:

- implemented
- trained
- tested
- interesting

without being:

- stable
- sufficiently useful
- computationally appropriate
- successfully integrated into the final pipeline

The final audit therefore confirms that SAM and SegFormer should not be presented as active components of the verified production pipeline merely because their checkpoints exist.

This is an important part of the project's research story.

---

# 30. Why Segmentation Was Still Valuable

Even though segmentation was not retained in the verified final pipeline, the experiments answered an important research question:

> Does explicit leaf extraction provide enough practical benefit to justify the additional complexity?

The experiments helped identify the trade-off between:

- better spatial isolation
- additional processing
- segmentation reliability
- integration complexity
- downstream classification behaviour

Therefore the segmentation work was not wasted.

It was part of the process of determining what should and should not enter the final system.

---

# 31. Grad-CAM Research

Grad-CAM was implemented to investigate explainability.

The objective was to generate a heatmap indicating which image regions contributed to the model prediction.

The intended output was:

Original image
+
Grad-CAM heatmap
=
Visual evidence

Grad-CAM support remains present in the pipeline.

---

# 32. Grad-CAM Limitation With Transformers

The experiments revealed an important issue.

The Grad-CAM output for the Transformer-based disease classifier was not always localized cleanly.

The documented issues included:

- heatmap spreading into background regions
- low localization precision
- difficulty obtaining a clean disease-lesion explanation

This meant that although Grad-CAM could be implemented technically, it did not automatically become a high-quality user-facing explainability feature.

The project therefore investigated possible future alternatives such as:

- Grad-CAM++
- Score-CAM
- EigenCAM
- LayerCAM

But these remained upgrade directions rather than verified final user-facing components.

---

# 33. Unknown Disease Problem

Another conceptual problem was identified.

A closed-set classifier is trained on known classes.

If an unknown disease is presented, the classifier may still assign it to one of the known classes.

This means:

High confidence
does not necessarily mean
the disease is actually in the training distribution.

The project therefore investigated the need for:

- open-set recognition
- entropy-based rejection
- out-of-distribution detection
- better confidence calibration

The current system contains uncertainty/OOD-related logic, but a fully validated unknown-disease research module remains a future extension rather than something that should be overstated.

---

# 34. Severity Estimation

Severity became another important requirement.

The project records describe the desire to distinguish levels such as:

Early
Moderate
Severe

The research direction considered disease-region information and infected-area estimation.

However, it is important to distinguish the experimental concept from the currently implemented pipeline.

The verified current system produces a severity level through its existing processing logic, but the more advanced lesion-area-based severity research was not fully established as a separate segmentation-driven module.

Therefore the project history records both:

- current severity output
- future lesion-area-based severity research

without pretending they are the same implementation.

---

# 35. Multi-Image Diagnosis

A major later development was the move from single-image prediction toward multiple-image consensus.

The motivation was practical.

One photograph may contain:

- poor focus
- an unrepresentative region
- occlusion
- lighting problems
- insufficient symptoms

Multiple photographs of the same plant can provide complementary evidence.

The system therefore introduced a consensus layer.

---

# 36. Consensus Architecture

The consensus system does not create another disease model.

Instead:

Image 1
↓
Base PlantClinicSystem

Image 2
↓
Base PlantClinicSystem

Image 3
↓
Base PlantClinicSystem

The resulting individual predictions are then passed to the consensus layer.

The verified audit specifically confirms that the consensus class reuses the existing base PlantClinicSystem rather than instantiating a second disease model.

---

# 37. Consensus Scoring

The consensus stage considers information including:

- confidence
- entropy
- image quality
- agreement
- valid results
- rejected images
- outlier predictions
- final prediction
- final confidence
- consensus strength
- severity

The recorded trust concept combines prediction confidence with quality and entropy-related weighting.

The final agreement ratio is then used to characterize consensus.

---

# 38. Consensus Strength

The documented consensus logic categorizes agreement approximately as:

agreement ≥ 0.80
→ HIGH

agreement ≥ 0.60
→ MODERATE

otherwise
→ LOW

The system can also identify images that disagree with the final prediction as outliers.

This is an important difference from simply averaging model probabilities blindly.

---

# 39. Handling Invalid Images During Consensus

One of the important design decisions was that one invalid image should not necessarily destroy the entire diagnosis.

The final system can:

- reject an invalid image
- continue analysing valid images
- calculate consensus from valid results
- report which images were excluded

This behaviour was demonstrated successfully.

---

# 40. Verified Multi-Image Demonstration

A verified final run used three uploaded images.

The report recorded:

Total Images Uploaded:
3

Successfully Analyzed Images:
2

Excluded Images:
1

The excluded image was rejected because it could not be validated as a plant image.

The two valid images both predicted:

Corn maize Northern Leaf Blight

Image 1:

Confidence:
97.97%

Image 2:

Confidence:
48.91%

The final consensus report produced:

Predicted Disease:
Corn maize Northern Leaf Blight

Confidence Score:
73.44%

Severity:
MODERATE

Consensus Strength:
HIGH

Reliability:
High diagnostic reliability

This was an important validation of the multi-image architecture.

---

# 41. Interface and Communication Development

Another part of the project involved redesigning system messages.

The goal was to make the output understandable to a normal user rather than exposing internal developer terminology.

Examples of message categories that were reviewed included:

- no image uploaded
- more than three images uploaded
- single-image warning
- processing image
- non-leaf rejection
- high uncertainty
- all images rejected
- high consensus
- failed analysis
- next-step guidance
- agricultural disclaimer

The design principle became:

> Explain what happened, explain why it happened when useful, and tell the user what to do next.

For example, a rejected image should communicate that the image could not be verified as a clear plant leaf rather than simply exposing an internal classification label.

---

# 42. Maximum Image Limit

The consensus system currently processes a maximum of three images per diagnosis session.

This was deliberately communicated to the user rather than silently ignoring additional files.

The system can report:

- how many images were uploaded
- how many were analysed
- how many were excluded
- which images were rejected
- whether the three-image limit was reached

This was part of the effort to make the system behaviour transparent.

---

# 43. Why the Interface Was Redesigned

The project was not only about making predictions.

A diagnostic system also needs to communicate uncertainty correctly.

Developer-style messages such as:

PROCESSING IMAGE 1

or

HIGH UNCERTAINTY DETECTED

may technically be correct but can sound robotic or alarming.

The interface work therefore aimed to use clearer language such as:

Analyzing Image 1...

or:

This image contained insufficient visual evidence for reliable disease analysis.

This was part of making the system more trustworthy and understandable.

---

# 44. Project Architecture Became More Focused

During development, the project accumulated multiple folders, scripts, models, checkpoints, backups, and experimental implementations.

At one point, the project effectively contained more than one complete system structure.

This created a new engineering problem:

> Which code is actually the final system?

A complete audit was therefore performed.

The audit searched the project for:

- Python files
- model architectures
- checkpoints
- class mappings
- pipeline references
- experimental modules

This was necessary before cleaning the GitHub repository.

---

# 45. Final Pipeline Audit

The audit confirmed the verified main pipeline:

src/pipeline/plant_clinic_system.py

and consensus pipeline:

src/pipeline/plant_clinic_consensus.py

The disease model was confirmed as:

Swin Large Patch4 Window12 384

The active disease checkpoint was confirmed as:

Major_Project/models/SwinLarge_384_ROBUST_FINAL.pth

The leaf gate was confirmed as:

MobileNetV3 Small

The active leaf checkpoint was confirmed as:

PlantClinic_AI/models/leaf_gate/mobilenetv3_leaf_gate_v2.pth

The class mapping was confirmed as:

PlantClinic_AI/environment/class_mapping.json

The class mapping contains:

32 classes.

The consensus system reuses the base PlantClinicSystem.

These findings are based on the final pipeline audit. :contentReference[oaicite:2]{index=2}

---

# 46. Important Historical vs Final Distinction

The project contains many historical checkpoints.

Examples include:

- ResNet18 checkpoints
- ResNet50
- ViT checkpoints
- ViT384 checkpoints
- SAM checkpoints
- SegFormer checkpoints
- RealESRGAN
- other experimental models

These files are evidence of research and experimentation.

They should not automatically be interpreted as active production components.

The final audit specifically warns against assuming that ResNet, ViT, SAM, SegFormer, or SwinIR are active simply because their checkpoint files exist. :contentReference[oaicite:3]{index=3}

This distinction is deliberately preserved in this document.

---

# 47. Important Correction to Earlier Project Notes

Some earlier development notes describe the leaf detector as ResNet18.

That reflects an earlier stage of development.

The final verified pipeline uses:

MobileNetV3 Small

for the active leaf/non-leaf gate.

The final audit confirms this architecture and its checkpoint. :contentReference[oaicite:4]{index=4}

Therefore the project history should not erase the earlier ResNet18 experiment, but it also should not incorrectly present it as the final leaf model.

This is an example of why maintaining a development history is useful.

---

# 48. Final Verified Architecture

The current verified production-focused architecture is:

Input Image
↓
MobileNetV3 Small
Leaf / Non-Leaf Gate
↓
Image Quality / Blur Analysis
↓
Conditional Processing
├── CLAHE when required
└── ESRGAN only when an upsampler is supplied
↓
Swin Large 384
Disease Classification
↓
Confidence / Entropy
↓
OOD / Threshold Logic
↓
Grad-CAM support
↓
Severity
↓
Individual Result
↓
Multi-Image Consensus
↓
Final Diagnostic Report

The verified audit records this structure explicitly. :contentReference[oaicite:5]{index=5}

---

# 49. What Is Currently Active

## Active / Verified

### Leaf validation

MobileNetV3 Small

Checkpoint:

mobilenetv3_leaf_gate_v2.pth

### Disease classification

Swin Large Patch4 Window12 384

Checkpoint:

SwinLarge_384_ROBUST_FINAL.pth

### Class mapping

32-class mapping.

### Image quality

Blur analysis.

### Enhancement

CLAHE support.

### Optional enhancement

ESRGAN support when an upsampler is explicitly supplied.

### Prediction robustness

Test-time augmentation and confidence/entropy processing.

### Uncertainty handling

Threshold/OOD-related logic.

### Severity

Current severity output.

### Multi-image reasoning

PlantClinicConsensusSystem.

### Reporting

Final diagnostic report.

---

# 50. What Remained Experimental

The following should remain documented as research history rather than being presented as verified active production components:

- ResNet18 disease experiments
- ResNet50 disease experiments
- ViT experiments
- ViT384 experiments
- SAM segmentation experiments
- SegFormer experiments
- hybrid segmentation
- leaf cropping/instance extraction experiments
- SwinIR experiments
- experimental enhancement configurations
- advanced Grad-CAM alternatives
- future open-set recognition experiments
- future lesion segmentation
- future advanced severity estimation
- future crop-specific models
- future multi-task learning

---

# 51. The Main Research Lesson

The most important result of the project is not simply:

"Swin Large achieved a high accuracy."

The larger research story is:

A clean dataset classifier was not considered sufficient.

The project therefore investigated:

1. Better disease architectures.
2. Input validation.
3. Negative examples.
4. Hard negatives.
5. Biological filtering.
6. Image-quality analysis.
7. Image enhancement.
8. Test-time augmentation.
9. Confidence handling.
10. Segmentation.
11. Explainability.
12. Unknown-disease handling.
13. Multi-image reasoning.
14. Consensus.
15. User-facing reporting.

Several of these experiments failed, were limited, or were not worth integrating.

That is a normal and valuable part of research development.

---

# 52. Why the Failed Experiments Matter

A failed experiment is not wasted work if it answers a question.

For example:

### ResNet experiments

Question:

Can a conventional CNN provide the required disease representation?

Result:

Useful baseline, but later architectures were investigated.

### ViT experiments

Question:

Can Transformer-based global visual representation improve the problem?

Result:

Useful improvement and research evidence, but not selected as final.

### Higher-resolution ViT

Question:

Does more spatial detail improve representation?

Result:

Useful experiment, but Swin Large 384 became the final direction.

### Hard Bio-Gate

Question:

Can simple biological rules prevent invalid inputs?

Result:

Useful concept, but overly strict rules caused false rejection.

### ESRGAN

Question:

Can super-resolution help degraded images?

Result:

Potentially useful for selected cases, but not universally valuable.

### SAM / SegFormer

Question:

Can explicit leaf segmentation improve downstream diagnosis?

Result:

Interesting research direction, but not sufficiently integrated into the verified final pipeline.

### Grad-CAM

Question:

Can the classifier provide disease-region explanations?

Result:

Implemented, but Transformer localization was not sufficiently clean for a polished user-facing feature.

### Multi-image consensus

Question:

Can multiple independent image observations improve diagnostic reliability?

Result:

Successfully integrated and demonstrated in the final working system.

---

# 53. Development Philosophy

The project followed a practical research philosophy:

Do not keep a component simply because it is technically impressive.

Do not remove a component simply because the first implementation failed.

Instead:

Research it.

Implement it.

Test it.

Find its weaknesses.

Measure whether the weaknesses can be solved.

If the component becomes useful, integrate it.

If it does not provide sufficient value, preserve it as research history and move forward.

This philosophy prevented the final system from becoming unnecessarily complicated.

---

# 54. Reproducibility Philosophy

The final GitHub repository is intentionally production-focused.

The repository should contain:

- source code
- class mapping
- requirements
- documentation
- development history
- dataset references
- model download instructions

Large datasets should not be placed directly into GitHub.

Large model checkpoints should also be hosted separately.

This allows the repository to remain manageable while still making the project reproducible.

---

# 55. Final Project Structure

The cleaned repository currently follows the production-focused structure:

PlantClinic_AI/

README.md

requirements.txt

.gitignore

datasets/
    README.md

environment/
    class_mapping.json

models/
    README.md

src/
    __init__.py
    pipeline/
        __init__.py
        plant_clinic_system.py
        plant_clinic_consensus.py

docs/
    DEVELOPMENT_JOURNEY.md

---

# 56. Current Repository Philosophy

The GitHub repository is not intended to pretend that every historical experiment is part of the final system.

Instead, it has two responsibilities:

### Final system

Show exactly what someone needs to understand and run the verified pipeline.

### Research history

Show how the system evolved and why particular approaches were retained or abandoned.

This separation makes the project more honest and easier to evaluate.

---

# 57. What Another Researcher Should Understand

A researcher reading this project should be able to understand:

Why ResNet was investigated.

Why the project moved toward Transformers.

Why Swin Large 384 became the final disease model.

Why a dedicated leaf gate was required.

Why negative examples were important.

Why hard filters caused problems.

Why image-quality processing was investigated.

Why ESRGAN remained optional.

Why segmentation was investigated but not promoted into the verified final pipeline.

Why Grad-CAM was implemented but remained limited.

Why confidence and uncertainty became important.

Why multiple images were introduced.

Why consensus became part of the final system.

Why the repository does not contain every experimental checkpoint.

---

# 58. Current State

The current verified production-focused system is functional.

The demonstrated consensus workflow successfully:

- accepted multiple images
- rejected an invalid image
- analysed valid images
- obtained individual predictions
- calculated consensus
- produced a final disease
- calculated final confidence
- produced a severity level
- generated a final diagnostic report

The verified example produced:

Corn maize Northern Leaf Blight

73.44% final confidence

MODERATE severity

HIGH consensus

High diagnostic reliability

This is the current system state that should be used for the repository's reproducibility documentation.

---

# 59. What We Will Do Next

The next repository phase is reproducibility.

The sequence is:

1. Host the final disease model.
2. Host the final leaf-gate model.
3. Record the real download links.
4. Update models/README.md.
5. Update datasets/README.md with the original dataset links.
6. Update the main README.
7. Add exact installation instructions.
8. Add exact model-placement instructions.
9. Test the project from a clean clone.
10. Verify that another person can reproduce the pipeline.
11. Only after this is verified, clean the large files from Google Drive.

The large original project data should not be deleted before this process is verified.

---

# 60. Final Summary

PlantClinic AI was not developed by choosing one model and writing one prediction script.

It evolved through multiple research and engineering cycles.

The major progression was:

Dataset investigation
↓
Data cleaning
↓
Leaf/non-leaf experiments
↓
Negative and hard-negative investigation
↓
Bio-Gate
↓
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
↓
Image quality analysis
↓
CLAHE
↓
ESRGAN experiments
↓
TTA
↓
Confidence / uncertainty work
↓
Segmentation experiments
↓
SAM
↓
SegFormer
↓
Leaf extraction experiments
↓
Grad-CAM
↓
Explainability analysis
↓
Threshold/OOD investigation
↓
Multi-image reasoning
↓
Consensus
↓
User-facing report refinement
↓
Final production-focused pipeline
↓
GitHub cleanup
↓
Reproducibility

The final system is therefore the result of repeated experimentation and failure-driven redesign.

The important distinction is:

**Everything that was tried is part of the development history.**

**Only what was actually verified and integrated belongs to the final production architecture.**

That distinction will be maintained throughout the repository.

---

# 61. Source and Audit Notes

The development history is reconstructed from project development records, training notes, dataset investigations, pipeline audits, and verified execution results.

Some older project notes describe earlier versions of components. Where an older implementation conflicts with the final audited implementation, the historical version is retained as history and the final audited implementation is identified as the current state.

In particular:

- Earlier notes describe a ResNet18 leaf classifier.
- The final verified pipeline uses MobileNetV3 Small for the leaf gate.
- Earlier research plans describe SAM/SegFormer integration.
- The final audit confirms that these should not be assumed to be active production dependencies.
- The final disease classifier is verified as Swin Large Patch4 Window12 384.
- The final class mapping contains 32 classes.

This document intentionally preserves those transitions instead of silently rewriting history.
