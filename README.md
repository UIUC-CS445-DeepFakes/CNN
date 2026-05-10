# CNN

## Intro

This repository conatains the CNN(VGG16) implemenataion for Spaital DeepFake Analysis.

## Preprocessing

All images are resized and normalized before being fed into the model.

**Training augmentations** (applied only during training to improve generalization):

- Resize to 256×256, then random crop to 224×224
- Random horizontal flip
- Color jitter (brightness, contrast, saturation)
- Normalize using ImageNet mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`

**Validation / Test** (deterministic):

- Resize directly to 224×224
- Same normalization as above

The dataset is split into **train / val / test** sets:  22,702 images with an approximate 70% training, 15% testing, and 15% validation.

## Model Design

The model is a fine-tuned **VGG-16** pretrained on ImageNet, adapted for binary deepfake classification.

| Component       | Detail                                        |
| --------------- | --------------------------------------------- |
| Backbone        | VGG-16 (ImageNet pretrained)                  |
| Frozen layers   | Conv blocks 0–16                              |
| Classifier head | `512 → → 1` with ReLU + Dropout (p=0.5) |
| Output          | Single logit → sigmoid at inference           |
| Loss            | `BCEWithLogitsLoss`                           |
| Optimizer       | Adam (`lr=3e-4`, `weight_decay=1e-4`)         |
| LR schedule     | `ReduceLROnPlateau` (factor=0.5, patience=2)  |
| Early stopping  | Patience of 20 epochs                         |

## Training and Datasets

Our datasets were preprocessed, zipped, and stored in Google Drive. See the [Data Extraction Repository](https://github.com/UIUC-CS445-DeepFakes/Data-extraction) for more details.
During runtime, these zips files are copied from [Google Drive](https://drive.google.com/drive/folders/1D9VHNO4FI2E1Mxd78-GUCj0alPPMAdY0?usp=sharing) and into the Colab enviroment. This allows us to decrease file reading time.

`DataLoader` shuffles the imagages since the files are in video or person ID order. `Num_workers=2` and `pin_memory=True` to decrease runtime.

The model is trained until a set epoch is reached. For each batches (`batch_size=32`) we predict the labels and compute the average loss. These are stored for visualization and optimization processes. The trained model is stored for version control.

When testing and validating the model is shifted to evaluation mode. This function iterates over the data loader. We pass a state paramter to the `test_model()` function to decide which values to return. Regardless of "state" a classification report is returned to exam the performance of the model. During Validation, the predictions and labels are returned which allows us to see the images and their predicted and actual label.
<img width="1170" height="400" alt="image" src="https://github.com/user-attachments/assets/06a23db2-6565-46c1-8188-1d807ae96a9b" />

For visualization:
* `from sklearn.metrics import classification_report` creates classification reports
* `pandas` convert classification reports dictionary to table.
* `from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay` creates Confusion Matrices

* The following libraries was used to create Gradient-weighted Class Activations heatmap overlaying the images to identify the regions used by the model to classify each datasets.
  - `from pytorch_grad_cam import GradCAM`
  - `from pytorch_grad_cam.utils.image import show_cam_on_image`
  - `from PIL import Image`
  - `from torchvision.transforms.functional import to_pil_image`
  - `import torchcam`
  - `from torchcam.utils import overlay_mask`
  - `import numpy as np`
    <img width="1570" height="309" alt="image" src="https://github.com/user-attachments/assets/4f4910df-abf2-4142-86ea-02b770a40e57" />

    
## Contributers

- Ryan Lee : rdlee4@illinos.edu
- Aleem Prince: Aleemap2@illinois.edu

## Sources

1. Project Motivation:
   - Shahzad, H. F., Rustam, F., Flores, E. S., Luís Vidal Mazón, J., de la Torre Diez, I., & Ashraf, I. (2022). A Review of Image Processing Techniques for Deepfakes. Sensors (Basel, Switzerland), 22(12), 4556. https://doi.org/10.3390/s22124556
2. Deep Learning with Pytorch:
   - https://github.com/udacity/deep-learning-v2-pytorch/blob/master/intro-to-pytorch/Part%207%20-%20Loading%20Image%20Data%20(Solution).ipynb
3. VGG16 refernence:
   - https://towardsdatascience.com/an-implementation-of-vgg-dea082804e14/
   - https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
   - https://gist.github.com/KushajveerSingh/7773052dfb6d8adedc53d0544dedaf60
4. GRAD-CAM
   - https://www.kaggle.com/code/kenny3s/pytorch-grad-cam
   - https://jacobgil.github.io/pytorch-gradcam-book/Pixel%20Attribution%20for%20embeddings.html
