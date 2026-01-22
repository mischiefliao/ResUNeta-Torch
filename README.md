# ResUNet-a: PyTorch Implementation

This repository contains a **PyTorch implementation** of the ResUNet-a model, converted from the original [TensorFlow/Keras version](https://github.com/Akhilesh64/ResUnet-a).

ResUNet-a is a deep learning framework for semantic segmentation of remotely sensed data, originally proposed in the paper:

> **ResUNet-a: a deep learning framework for semantic segmentation of remotely sensed data**  
> Foivos I. Diakogiannis, François Waldner, Peter Caccetta, Chen Wu  
> ISPRS Journal of Photogrammetry and Remote Sensing, Volume 162, Pages 94-114, 2020

## Features

- **PyTorch Implementation**: Complete conversion from TensorFlow/Keras to PyTorch
- **Same Interface**: Maintains the same command-line interface as the original TensorFlow version
- **Residual Blocks**: Implements ResNet v2 style residual blocks with BatchNorm before activation
- **U-Net Architecture**: Encoder-decoder structure with skip connections
- **Multiple Loss Functions**: Supports BCE, Dice, and Tanimoto loss functions
- **Data Augmentation**: Built-in data augmentation using Albumentations
- **Comprehensive Metrics**: Calculates IoU, Precision, Recall, and F1-score

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.9.0+
- CUDA (optional, for GPU acceleration)

### Install Dependencies

```bash
pip install -r requirements.txt
```

For PyTorch with CUDA support, install PyTorch from the [official website](https://pytorch.org/get-started/locally/) according to your CUDA version.

## Usage

### Training

To train the model, use the `main.py` script with the following arguments (same as the TensorFlow version):

```bash
python main.py \
    --image_size 256 \
    --batch_size 8 \
    --num_classes 2 \
    --validation_split 0.2 \
    --epochs 100 \
    --image_path ./images \
    --gt_path ./gt \
    --layer_norm batch \
    --model_save_path ./ \
    --checkpoint_mode epochs \
    --learning_rate 1e-4 \
    --loss_function tanimoto
```

#### Arguments

- `--image_size`: Input image size (default: 256)
- `--batch_size`: Batch size for training (default: 8)
- `--num_classes`: Number of output classes (default: 2)
- `--validation_split`: Fraction of data for validation (default: 0.2)
- `--epochs`: Number of training epochs (default: 100)
- `--image_path`: Path to directory containing training images (required)
- `--gt_path`: Path to directory containing ground truth masks (required)
- `--layer_norm`: Normalization type - 'batch', 'instance', or 'layer' (default: 'batch')
- `--model_save_path`: Directory to save model checkpoints (default: './')
- `--checkpoint_mode`: When to save checkpoints - 'epochs' or 'best' (default: 'epochs')
- `--learning_rate`: Learning rate for optimizer (default: 1e-4)
- `--loss_function`: Loss function - 'bce', 'dice', or 'tanimoto' (default: 'tanimoto')

### Prediction

To generate predictions on test images:

```bash
python predict.py \
    --image_size 256 \
    --num_classes 2 \
    --image_path ./test \
    --model_path ./best_model.pth \
    --output_path ./results
```

#### Arguments

- `--image_size`: Input image size (default: 256)
- `--num_classes`: Number of classes (default: 2)
- `--image_path`: Path to directory containing test images (required)
- `--model_path`: Path to trained model file (.pth) (required)
- `--output_path`: Directory to save prediction results (required)

## Project Structure

```
resunet-a-pytorch/
├── model.py              # ResUNet-a model definition
├── loss.py               # Loss functions (Tanimoto, Dice, BCE)
├── main.py               # Training script
├── predict.py            # Prediction script
├── batch_preprocess.py   # Data loading and preprocessing
├── utils.py              # Utility functions (metrics, visualization)
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── LICENSE               # MIT License
```

## Differences from TensorFlow Version

### Model Format

- **TensorFlow**: Models saved as `.h5` files using Keras `model.save()`
- **PyTorch**: Models saved as `.pth` files using `torch.save(model.state_dict())`

### Data Format

- **TensorFlow**: Default channel-last format (H, W, C)
- **PyTorch**: Default channel-first format (C, H, W)

### Training Loop

- **TensorFlow**: Uses `model.fit()` with callbacks
- **PyTorch**: Manual training loop with explicit gradient computation

### Model Loading

- **TensorFlow**: `tf.keras.models.load_model('model.h5')`
- **PyTorch**: `model.load_state_dict(torch.load('model.pth'))`

### Device Management

- **TensorFlow**: `with tf.device('/gpu:0')`
- **PyTorch**: `model.to(device)` and `data.to(device)`

## Model Architecture

The ResUNet-a model consists of:

1. **Encoder Path**: Downsampling path with residual blocks
2. **Decoder Path**: Upsampling path with skip connections
3. **Residual Blocks**: ResNet v2 style blocks with BatchNorm before activation
4. **Atrous Convolution**: Optional dilated convolutions in deeper layers

## Loss Functions

### Tanimoto Loss

The Tanimoto coefficient (also known as Jaccard index) measures the similarity between predicted and ground truth masks:

```
T = intersection / (pred_sum + target_sum - intersection)
Loss = 1 - T
```

### Tanimoto with Complement

Combines the original Tanimoto loss with the complement Tanimoto loss:

```
Loss = α * (1 - T) + (1 - α) * (1 - T_complement)
```

## Results

The model performance depends on your dataset and training configuration. The original paper reports state-of-the-art results on the ISPRS Potsdam dataset.

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{DIAKOGIANNIS202094,
    title = {ResUNet-a: A deep learning framework for semantic segmentation of remotely sensed data},
    journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
    volume = {162},
    pages = {94 - 114},
    year = {2020},
    issn = {0924-2716},
    doi = {https://doi.org/10.1016/j.isprsjprs.2020.01.013},
    url = {http://www.sciencedirect.com/science/article/pii/S0924271620300149},
    author = {Foivos I. Diakogiannis and François Waldner and Peter Caccetta and Chen Wu},
    keywords = {Convolutional neural network, Loss function, Architecture, Data augmentation, Very high spatial resolution}
}
```

## Acknowledgments

- Original TensorFlow implementation: [Akhilesh64/ResUnet-a](https://github.com/Akhilesh64/ResUnet-a)
- Original MXNet implementation: [feevos/resuneta](https://github.com/feevos/resuneta)
- Original paper authors: Foivos I. Diakogiannis, François Waldner, Peter Caccetta, Chen Wu

## License

MIT License - see [LICENSE](LICENSE) file for details.
