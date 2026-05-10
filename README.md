# CNN

## Intro

This repository conatains the CNN implemenataion for DeepFake Analysis.

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

The dataset is split into **train / val / test** sets

## Model Design

The model is a fine-tuned **VGG-16** pretrained on ImageNet, adapted for binary deepfake classification.

| Component       | Detail                                        |
| --------------- | --------------------------------------------- |
| Backbone        | VGG-16 (ImageNet pretrained)                  |
| Frozen layers   | Conv blocks 0–16                              |
| Classifier head | `4096 → 1024 → 1` with ReLU + Dropout (p=0.5) |
| Output          | Single logit → sigmoid at inference           |
| Loss            | `BCEWithLogitsLoss`                           |
| Optimizer       | Adam (`lr=1e-4`, `weight_decay=1e-4`)         |
| LR schedule     | `ReduceLROnPlateau` (factor=0.5, patience=2)  |
| Early stopping  | Patience of 5 epochs                          |

## Training and Datasets

## Contributers

- Ryan Lee : @illinos.edu
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
