# CAMTS-UNet for Brain Tumor Segmentation

This repository contains the implementation accompanying the paper on **CAMTS-UNet**, a deep-learning model for brain MRI tumor segmentation.

CAMTS-UNet is a U-Net-style architecture that combines:

- Residual convolutional blocks in the encoder
- Atrous Spatial Pyramid Pooling (ASPP) for multi-scale context
- Cross-attention gates for refining encoder skip connections
- Deep supervision during training
- Dice + binary cross-entropy loss
- Explainability with Grad-CAM

> **Note:** This repository currently contains the exported training script. The script was generated from notebook code and includes the complete data preparation, training, evaluation, visualization, baseline comparison, and inference pipeline.

## Repository contents

- [`CAMTS_UNet_Full_extracted.py`](CAMTS_UNet_Full_extracted.py) — complete training and evaluation pipeline
- `dataset_description.png` — generated dataset overview
- `methodology_pipeline.png` — generated methodology diagram
- `training_curves.png` — generated training curves
- `quantitative_donut_charts.png` — generated metric visualization
- `qualitative_analysis.png` — generated qualitative predictions
- `predictions_gradcam.png` — generated Grad-CAM visualizations
- `mhd_comparison.png` — generated mean Hausdorff distance comparison
- `model_comparison_table.png` — generated baseline comparison table
- `single_inference.png` — generated single-image inference visualization

The image files above are created when the corresponding sections of the script are run; they may not be present in a fresh checkout.

## Dataset

The pipeline expects the LGG brain MRI segmentation dataset in the following directory structure:

```text
lgg-mri-segmentation/
└── kaggle_3m/
    ├── TCGA_*/
    │   ├── image.tif
    │   ├── image_mask.tif
    │   └── ...
    └── ...
```

The default path used by the script is:

```text
/content/lgg-mri-segmentation/kaggle_3m
```

To use a local dataset, update `Config.DATA_DIR` near the beginning of `CAMTS_UNet_Full_extracted.py`.

## Installation

Python 3.9+ is recommended. Install the required packages with:

```bash
pip install torch torchvision opencv-python-headless albumentations matplotlib scikit-learn tqdm scipy segmentation-models-pytorch pillow
```

A CUDA-enabled PyTorch installation is recommended for training, but the script can fall back to CPU when CUDA is unavailable.

## Usage

Run the complete pipeline with:

```bash
python CAMTS_UNet_Full_extracted.py
```

The script will:

1. Load and pair MRI images with segmentation masks.
2. Balance positive and negative samples.
3. Split the data into training, validation, and test sets using a 70/15/15 split.
4. Resize images to `256 × 256` and apply augmentation to the training set.
5. Train CAMTS-UNet for 50 epochs by default.
6. Save the best checkpoint as `camts_unet_best.pth`.
7. Evaluate Dice, IoU, precision, recall, F1, and mean Hausdorff distance.
8. Generate qualitative, quantitative, baseline-comparison, and Grad-CAM visualizations.

The main training settings can be changed in the `Config` class:

```python
class Config:
    IMAGE_SIZE = 256
    BATCH_SIZE = 8
    EPOCHS = 50
    LR = 1e-4
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
```

## Model overview

The model uses the following high-level structure:

```text
Input MRI
   ↓
Residual encoder
   ↓
ASPP bottleneck
   ↓
Cross-attention gated skip connections
   ↓
Multi-stage decoder
   ↓
Binary tumor segmentation mask
```

The implementation also evaluates several pretrained U-Net baselines from `segmentation-models-pytorch`, including VGG16, ResNet34, ResNet50, EfficientNet-B0, and MobileNetV2 encoders.

## Inference

After training, the script demonstrates single-image inference with:

- A predicted probability map
- A binary tumor mask
- Tumor detection status
- Predicted tumor area in pixels and as a percentage of the image

To run inference on a different image, update `sample_path` near the end of the script and ensure the trained checkpoint is available.

## Reproducibility

The default random seed is `42`. Results can vary depending on the PyTorch version, CUDA environment, dataset preparation, and hardware.

## Citation

If you use this code, please cite the associated paper:

```bibtex
@article{camtsunet,
  title   = {CAMTS-UNet for Brain MRI Tumor Segmentation},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO}
}
```

Please replace the placeholder citation fields with the final bibliographic information for the paper.

## License

No license file is currently included in this repository. Please contact the repository owner before redistributing or using the code in a way that requires explicit licensing terms.
