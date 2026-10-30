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

The model used was the convnext-tiny variant.

### Architecture

[image]

Figure 1 shows the ConvNeXt architecture. MRI brain scans were passed into the network as input. It was then prepared for deeper processing by passing it through a 2D convolutional layer and layer normalisation (4x4 non-overlapping patchify convolution). 
It was then passed through 4 ConvNeXt blocks, each designed to extract increasingly complex features. Downsampling layers in between each ConvNeXt block allow the model to find higher level patterns by reducing spatial dimensions and increasing feature channels. Finally, global average pooling and a linear classifier produce the final output.
 
Instead, of duplicating the grayscale MRI images across 3 channels (as was done in the ImageNet trained ConvNeXt), my implementation modies the patch stem to accept 1 input channel. This reduces the number of parameters and memory usage. 

# Project Data

The ADNI dataset was used in this project. 

The provided dataset (on the Rangpur cluster) has been preprocessed and contains 2 classes: Alzheimer's Disease (AD) and Cognitive Normal (CN). Each patient is associated with 20 2D slices of their MRI brain scan.
Filenames in the dataset are <patient_id>_<slice_id>.jpeg. The patient id was used to split data and verify there no leakage between the preprocessed training and testing sets (`verify_leakage.py`).

The training set contains 1076 patients. This will be split so that the training set contains 80%: 860 patients. The validation set will then have 216 patients. The testing set contains 450 patients. Since the sample size is quite small, augmentations were be performed, only on the training dataset.

Early experiements showed high validation accuracy (up to 95%) while test accuracy reamined in the low 70s.
To ensure uniform distribution when splitting training and validation sets, pixel intensity was analysed at the patient level. The training set was split into 5 bins and the validation set was created by randomly sampling from each bin.

## Data Preprocesing

Augmentations were performed on the traininng data only. These included:
- RandomRotation and RandomAffine to simulate natural variations in patient head positioning
- ColorJitter to simulate tissue contrast between different scanners
- RandomErasing to act as a dropout, and ensure the network did not overfit to localised regions

Images were also normalised to have a mean and standard deviation of 0.5, to ensure pixel intensities were scaled to a consistent range. This stabilises training and allows the model to converge more efficiently.

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

# Training and Inference

## Final Hyperparameters

The model was trained using the following:
- Learning rate: 0.0005
- Batch size: 16
- Epochs: 100
- Weight decay: 0.05 on AdamW optimiser
- Cross entropy loss with 0.1 label smoothing
- Cosine Annealing learning rate scheduler with linear warmup (10 epochs) 
- Drop path: 0.1

## Training Results

[training curves image]

Figure x shows loss and accuracy for training, validation and testing.

Training loss drops to 0.23 from 0.69 and validation loss follows closely, dropping to 0.18 from 0.61. The variations in the validation loss can be explained by the batch size of 16, which causes noisy weight updates.
Validation loss is lower than training loss because a dropout of 0.1 was used during training only.

Training and validation accuracy increase with minimal divergence. This suggests that the model was not overfitting.

## Test Results

The model recieved a final test accuracy of **77.18%**, with a patient level test accuracy of **78.22**.
Although this is much lower than validation accuracy and loss, it can be explained by a few possible reasons:
- The test set may have a different distribution to the testing and validation sets
- There may be a different class imbalance distribution between the test and training sets

[confusion matrix]

[classification report]

# The Open Research Dilemma

## CNN vs ConvNeXt

## Resource Profiling

## Error Autopsies

## Actionable Engineering recommendation


# Dependencies
All experiements were conducted on GPUs on the Rangpur cluster.

Software versions used to run the experiements were

```python >= 3.10
torch >= 2.0.1
torchvision >= 0.15.2
numpy >= 1.24.3
scikit-learn >= 1.3.0
matplotlib >= 3.7.1
pillow >= 9.5.0
```

## Reproducibility

To ensure reproducibility, a fixed random seed (42) was used..

To run the model:

1. Clone the repository, and ensure all dependencies are intalled
2. Navigate to the convenext_47496390 directory
3. To run ConvNeXt with the final hyperparameters:
        `python train.py --model convnext --epochs 100 --batch_size 16 --lr 5e-4 --weight_decay 0.05 --data_root /home/groups/comp3710/ADNI/AD_NC` 
4. To run iference using trained weights and visualise results:
        `python predict.py --model convnext --weights convnext_best.pth --data_root /home/groups/comp3710/ADNI/AD_NC --batch_size 16 --save_dir ./diagnostic_plots`

The inference script (`predict.py`) generates:
- `convnext_test_cm.png`: Annotated confusion matrix heatmap.
- `convnext_test_roc.png`: Receiver Operating Characteristic curve with AUROC calculation.
- `convnext_misclassified_slices.png`: 8-sample qualitative failure case autopsy grid.
- `convnext_sample_predictions.png`: 8-sample grid of correct diagnostic predictions.

# References
https://www.geeksforgeeks.org/computer-vision/convnext/
https://arxiv.org/abs/2201.03545
https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py

# AI Disclosure
AI was used to assist in building the dataloaders, ensure a patient level split and create a validation set uniformly by creating bins of the training set. It was then used to adapt the PyTorch ConvNeXt model and ensure alignment with the ConvNeXt paper.
AI was also used to create plotting functions, evaluate accuracy at the patient level, determine computational profiling and implement argument parsing.


