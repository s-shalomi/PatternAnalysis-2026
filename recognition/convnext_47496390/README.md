# ConvNeXt for classification of Alzheimer's Diseases and Congitive Normal on 2D MRI brain slice scans


## Project Description

Alzheimer’s Disease (AD) causes irreversible neurode generation marked by progressive hippocampal and cortical atrophy with ventricular enlargement.
Automated classification of T1-weighted structural brain MRIs into AD versus Cognitively Normal (CN) aids early screening. 
The primary challenge is extracting discriminative anatomical biomarkers amidst significant natural head shape and age variation between patients, while strictly avoiding patient data leakage across multi-slice extractions.

## User Need and Scope

The intended users are neurologists and radiologists who analyse MRI brain scans to detect and diagnose the onset of Alzheimer's disease.
Automated classifications of T1-weighted structural brain MRIs into AD vs CN will help these users in early screening and diagnosis of Alzheimer's disease in patients.

This project will allow the intended users to have an automated assistance that classifies only high confidence cases. Ambiguous cases are re-routed to the specialists for human review.

# The Model

## ConvNeXt

This project implements ConvNeXt (Liu et al., 2022). ConvNeXt is a modernised pure CNN architecture that integrates elements from Vision Transformers.  It modernises a standard ResNet model towards a Vision Transformer model, however is constructed entirely from standard ConvNet modules.

The key innovations that led to ConvNeXt are as follows:
- **Patchify Stem**: replaces 7x7 convolutions with 4x4 convolutions (with stride = 4) to create non-overlapping patches to improve efficiency
- **Stagewise Architecture**: 4 stage structure where feature maps are processed at progressively lower resolutions
- **Inverted Bottleneck Blocks**: expansion occurs before depthwise convolutions. Each block expands hidden dimensions using pointwise linear convolutions.
- **Layer Normalisation** instead of batch normalisation
- **GELU Activation**
- **Depthwise Separable Convolutions**: reduces the number of parameters, which reduces computation and allows the network to be deeper and wider

ConvNeXt architectures have been proven to match or surpass standard vision transformers on accuracy, while retaining the computational efficiency and hardware optimisations of CNNs.

### Architecture

[image]

Figure 1 shows the ConvNeXt architecture. MRI brain scans were passed into the network as input. It was then prepared for deeper processing by passing it through a 2D convolutional layer and layer normalisation (4x4 non-overlapping patchify convolution). 
It was then passed through 4 ConvNeXt blocks, each designed to extract increasingly complex features. Downsampling layers in between each ConvNeXt block allow the model to find higher level patterns by reducing spatial dimensions and increasing feature channels. Finally, global average pooling and a linear classifier produce the final output.
 
Instead, of duplicating the grayscale MRI images across 3 channels (as was done in the ImageNet trained ConvNeXt), my implementation modies the patch stem to accept 1 input channel. This reduces the number of parameters and memory usage. 

# Project Data

The ADNI dataset was used in this project. 

The provided dataset (on the Rangpur cluster) has been preprocessed and contains 2 classes: Alzheimer's Disease (AD) and Cognitive Normal (CN). Each patient is associated with 20 2D slices of their MRI brain scan.
Filenames in the dataset are <patient_id>_<slice_id>.jpeg. The patient id was used to split data and verify there no leakage between the preprocessed training and testing sets (verify_leakage.py).

## Data Splits

The training set contains 1076 patients. This will be split so that the training set contains 80%: 860 patients. The validation set will then have 216 patients. The testing set contains 450 patients. Since the sample size is quite small, augmentations were be performed, only on the training dataset.
To ensure uniform distribution when splitting training and validation sets, pixel intensity was analysed. The training set was split into 5 bins and the validation set was created by randomly sampling from each bin.

## Data Preprocesing

Images were normalised and augmentations were performed on the traininng data only. These included:
- RandomRotation and RandomAffine to simulate natural variations in patient head positioning
- ColorJitter to simulate tissue contrast between different scanners
- RandomErasing to act as a dropout, and ensure the network did not overfit to localised regions

## Acceptance Criteria

The Dilemma: models frequently achieve deceptively high aggregate test accuracy while concealing catastrophic overconfidence on borderline, ambiguous, or out-of-distribution cases.

To address this, the following acceptance criteria were defined:
- **Zero Data Leakage**: 100% patient level separation across data splits
- **Classification Accuracy**: 80% classification accuracy of AD and CN images.
- **Operational Feasibility**: Training completes in half a day with peak GPU VRAM limited to 16GB

## Risks

Risks include:
- Data leakage between training sets
- Class imbalance may cause bias towards majority class
- Unequal distribution between datasets may cause inflated accuracy readings

Fallback Model: Since ConvNeXt is build on a ResNet Architecture, a suitable fallback would be ResNet-18 or ResNet-50.



 Core Investigation: Does your chosen model architecture reliably quantify uncertainty on ambiguous
boundary cases compared to a simple baseline? Does its predicted confidence score (or embedding
metric distance) correlate with diagnostic correctness, or does the network remain stubbornly
overconfident even when making incorrect predictions?
• Engineering Recommendation: Establish an empirical confidence threshold or reject-option rule
that safely automates high-confidence cases while routing ambiguous cases to human specialists.
Does the added architectural complexity and computational cost of your advanced model justify its
operational advantage over the baseline


 Data Hygiene & Leakage Prevention: Enforce strict patient-level separation for medical images
(all slices/lesions from one patient reside strictly in one split). 

Baseline Implementation & Comparison: Implement a simple baseline model (e.g., a standard
convolutional network, MLP, or basic linear classifier) to establish an initial performance benchmark.
You will compare your chosen model from the table below against this baseline on the same evaluation
data.
3. Quantitative Benchmarking: Evaluate beyond raw accuracy. Report per-class precision, recall,
macro-F1, and AUROC. For severely imbalanced classes (melanoma), report Precision-Recall curves
(PR-AUC). 

 Failure Case Autopsy & Manager Recommendation: Inspect 3–5 representative failure
cases (false positives and false negatives) from the dataset to diagnose failure triggers. Propose a
calibrated operating threshold or decision rule to route low-confidence predictions to human experts,
and deliver an actionable recommendation to your project manager.

# init readme 
Sufficiently documenting your project, working principles, feasibility review, experiments, and instructions 

# ConvNeXt for classification of Alzheimer's Diseases and Congitive Normal on 2D MRI brain slice scans

# The algorithm



a description of the algorithm and the problem that it solves (approximately a paragraph),
how it works in a paragraph, and a figure/visualisation

## The problem solved

# Feasibility review

# Dependencies
dependencies required, including exact versions, and address reproducibility of results.

## Reproducibility of results

# Running the code
 Example inputs, outputs, and visual plots of your algorithm

# Preprocessing
Description of any specific pre-processing used, with references if any. Justify your training,
validation, and testing splits of the data.

# Open research dilemna 
## CNN vs ConvNeXt
Quantitative benchmarking of your chosen model against your implemented baseline model under
identical evaluation splits

## Resource profiling
Resource profiling table documenting computational efficiency (peak GPU VRAM, parameter
count, and inference runtime/latency).

## Error autopsies
 In-depth qualitative error autopsies analyzing 3–5 representative failure cases from the dataset
(diagnosing root triggers and failure modes).

## Engineering recommendation
Actionable engineering recommendation and trade-off synthesis

# AI usage disclosure









Augmentation Design & Justification:
    - RandomRotation (degrees=5) & RandomAffine: Model natural patient head tilt and alignment
      variations inside the MRI scanner bore without violating stereotaxic coordinate frames.
    - ColorJitter (brightness=0.1, contrast=0.1): Simulates inter-scanner B0/B1 field inhomogeneities
      and RF receive-coil sensitivity variations.
    - RandomErasing (p=0.25): Acts as spatial dropout, encouraging the network to discover distributed
      neurodegenerative biomarkers across both hemispheres and the ventricles.
    - Strictly Prohibited: Vertical flipping is omitted because the superior-inferior anatomical axis
      is fixed in clinical acquisition; inverting the brain introduces out-of-distribution artifacts.
