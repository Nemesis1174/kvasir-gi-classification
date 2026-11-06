# Gastrointestinal Image Classification with ResNet50 (Kvasir v2)

A deep learning classifier for endoscopic images of the gastrointestinal tract. I fine-tuned an ImageNet-pretrained ResNet50 with fastai on the Kvasir v2 dataset and used Albumentations for data augmentation. This was a course project for Intelligent Agents at Universidad Panamericana (Aguascalientes, Mexico, 2025).

## Results

The model reaches **91.2% accuracy and about 0.91 macro F1** on the validation set. The validation set has 1,600 images, 200 per class. There is no separate test set, so these are validation results.

| Class | Precision | Recall | F1 |
|---|---|---|---|
| dyed-lifted-polyps | 0.914 | 0.905 | 0.910 |
| dyed-resection-margins | 0.928 | 0.905 | 0.916 |
| esophagitis | 0.871 | 0.775 | 0.820 |
| normal-cecum | 0.929 | 0.980 | 0.954 |
| normal-pylorus | 0.975 | 0.995 | 0.985 |
| normal-z-line | 0.797 | 0.885 | 0.839 |
| polyps | 0.928 | 0.905 | 0.916 |
| ulcerative-colitis | 0.959 | 0.945 | 0.952 |

The most common confusion is between **esophagitis and normal-z-line** (45 esophagitis images predicted as normal-z-line). The highest-loss predictions are often visually ambiguous cases predicted with very high confidence, which suggests trying label smoothing or focal loss as a next step.

## Method

1. Kvasir v2 images are reorganized into `train` and `val` folders, one subfolder per class, so fastai can read them with `ImageDataLoaders.from_folder`.
2. Augmentation with Albumentations: CLAHE, random brightness and contrast, Gaussian blur, horizontal flip, rotation, resize to 224x224, and ImageNet normalization.
3. Model: `vision_learner` with a pretrained ResNet50, batch size 32, and a learning rate chosen with `lr_find()`.
4. Training in two phases: 10 epochs with the base frozen (learning rate 1e-3), then 20 epochs with all layers unfrozen and discriminative learning rates `slice(1e-6, 1e-4)`. `SaveModelCallback` keeps the best model by validation loss.
5. Evaluation with accuracy and macro precision, recall, and F1 (fastai and scikit-learn), a confusion matrix, the top-loss images, and per-class TP, TN, FP, FN.

## Repository contents

- `notebooks/kvasir_resnet50.ipynb`: the full Colab notebook with outputs.
- `kvasir_resnet50.py`: the same code exported as a script.
- `results/per_class_metrics.csv`: per-class precision, recall, F1, and confusion counts.
- `requirements.txt`: Python dependencies.

The trained weights (`kvasir_v2_resnet50.pkl`, about 100 MB) are not included because of GitHub's file size limit.

## How to run

The code was written for Google Colab. Download Kvasir v2 from the [Simula datasets page](https://datasets.simula.no/kvasir/), split it into `train` and `val` folders with one subfolder per class, update the dataset path in the first cells, and run the notebook with a GPU runtime.

## Dataset

Kvasir v2, published by Simula Research Laboratory in Norway: 8 classes (dyed-lifted-polyps, dyed-resection-margins, esophagitis, normal-cecum, normal-pylorus, normal-z-line, polyps, ulcerative-colitis). The images are not redistributed here. Please cite: Pogorelov et al., "KVASIR: A Multi-Class Image Dataset for Computer Aided Gastrointestinal Disease Detection," ACM MMSys 2017.

## Author

Christopher Márquez Reyes, [GitHub](https://github.com/Nemesis1174)
