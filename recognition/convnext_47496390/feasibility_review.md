# ADNI ConvNeXt Feasibility Review

## Project Description
Alzheimer’s Disease (AD) causes irreversible neurode generation marked by progressive hippocampal and cortical atrophy with ventricular enlargement.
Automated classification of T1-weighted structural brain MRIs into AD versus Cognitively Normal (CN) aids early screening. 
The primary challenge is extracting discriminative anatomical biomarkers amidst significant natural head shape and age variation between patients, while strictly avoiding patient data leakage across multi-slice extractions.

## User Need and Scope
The intended users are neurologists and radiologists who analyse MRI brain scans to detect and diagnose the onset of Alzheimer's disease.
Automated classifications of T1-weighted structural brain MRIs into AD vs CN will help these users in early screening and diagnosis of Alzheimer's disease in patients.

This project will allow the intended users to have an automated assistance that classifies only high confidence cases. Ambiguous cases are re-routed to the specialists for human review.

## Acceptance Criteria
**The Dilemma**: models frequently achieve deceptively high aggregate test accuracy while concealing catastrophic overconfidence on borderline, ambiguous, or out-of-distribution cases.

The following acceptance critera is defined to address this dilemma:
- 100% patient level separation across data splits
- 80% classification accuracy of AD and CN images
- Ambiguous MRI scans are re-routed for human review
- Training runtime is 60 seconds per epoch (Reyes & Sánchez, 2024)

## Model Choice and Course Concepts
**Chosen Difficulty**: Hard

**Dataset Configuration**:
Data will be split at the patient level into training, validation and testing.

**Model Architecture**:
The ConvNeXt architecture was introduced by Meta AI and UC Berkley. It modernises a standard ResNet model towards a Vision Transformer model, however is constructed entirely from standard ConvNet modules.

**Course Concepts**:
- Data splitting and preventing data leakage
- CNNs
- Testing/evaluation of model performance

 proposed dataset configuration, model architecture,
chosen difficulty band, and how course concepts and literature support the choices.

## Preliminary Feasibility Evidence


 initial data audit, baseline smoke-test experiment, or
measured resource estimate. A fully trained final model is not required

## Risks, Budget and Fallback


**Fallback Model**: 
Since ConvNeXt is build on a ResNet Architecture, a suitable fallback would be ResNet-18 or ResNet-50.

Key technical risks, computing budget, the next planned experiment,
and a workable fallback.

## References
- https://www.sciencedirect.com/science/article/pii/S2405844024014993
- https://share.gemini.google/OlLE9OZGsMfJ

# questions
- referencing - does it need to be apa
- is acceptance criteria fine, does it need more detail or justification
- model architecture is convnext, do i need to describe it
- what does course concepts expect
- what does dataset config mean
- what does budget mean
- what experiments should we be running?
