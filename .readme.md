
# AlexNet Implementation

## Overview

This project provides an implementation of the **AlexNet architecture**, as described in the paper 
[ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) by Alex Krizhevsky et al.

The implementation includes:
- **AlexNet model definition** using PyTorch.
- **Training pipeline** with dataset preparation, logging (via TensorBoard), and checkpointing.
- **Scripts for dataset organization** and model evaluation.

---

## Project Structure

Below is an overview of the key files and folders in this repository:

```
AlexNet/
│
├── model.py               # AlexNet model implementation and training loop
├── extract_imagenet.sh    # Script to organize and extract the ImageNet dataset
├── rearrange_tinyimagenet.py  # Helper script for TinyImageNet dataset preparation
├── happy.h                # (Optional: Unused header file, can be ignored or removed)
├── requirements.txt       # List of required Python libraries for this project
└── README.md              # This README file
```

---

## Setup Instructions

### **1. Install Dependencies**
Make sure you have Python 3.7+ installed. You can install the required Python libraries using:

```bash
pip install -r requirements.txt
```

### **2. Prepare the Dataset**
- Ensure that the **ImageNet** dataset or a subset like **TinyImageNet** is downloaded.
- Use the provided scripts to organize the dataset:
  - **For ImageNet**:
    ```bash
    bash extract_imagenet.sh
    ```
  - **For TinyImageNet**:
    Use `rearrange_tinyimagenet.py` to rearrange the dataset folders for PyTorch's `ImageFolder` format.

### **3. Train the Model**
Run the `model.py` script to start training the AlexNet model:

```bash
python model.py
```

Key training features:
- Supports GPU acceleration (multi-GPU training enabled).
- Logs metrics like loss and accuracy using TensorBoard.
- Saves model checkpoints after each epoch.

---

## Logging and Monitoring
TensorBoard is used for logging metrics like **loss**, **accuracy**, **gradients**, and **weights**.

### To start TensorBoard:
```bash
tensorboard --logdir=<LOG_DIR>
```
Replace `<LOG_DIR>` with the directory path where logs are saved (configured in `model.py`).

---

## File Descriptions

### **1. model.py**
- Contains the implementation of the AlexNet model.
- Includes training logic:
  - Dataset loading
  - Loss calculation
  - Backpropagation and optimization
  - Logging and checkpointing

### **2. extract_imagenet.sh**
- Bash script to extract and organize the ImageNet dataset into a format compatible with PyTorch’s `ImageFolder`.

### **3. rearrange_tinyimagenet.py**
- Python script to restructure the **TinyImageNet dataset** into training and validation folders.

### **4. requirements.txt**
- Contains the Python library dependencies for this project:
  - `torch`
  - `torchvision`
  - `tensorboardX`

---

## Model Overview

AlexNet consists of:
- Five convolutional layers followed by ReLU activations and max-pooling layers.
- Three fully connected layers for classification.
- Dropout layers for regularization.

### Architecture Details
| Layer Type              | Parameters                |
|-------------------------|---------------------------|
| Conv2D + ReLU + Pooling | 96 filters, 11x11 kernel  |
| Conv2D + ReLU + Pooling | 256 filters, 5x5 kernel   |
| Conv2D + ReLU           | 384 filters, 3x3 kernel   |
| Conv2D + ReLU           | 384 filters, 3x3 kernel   |
| Conv2D + ReLU + Pooling | 256 filters, 3x3 kernel   |
| Fully Connected (FC)    | 4096 neurons              |
| Fully Connected (FC)    | 4096 neurons              |
| Fully Connected (FC)    | 1000 neurons (output)     |

---

## Future Improvements
- Add support for additional datasets like CIFAR-10 or MNIST.
- Implement testing and validation pipelines.
- Experiment with hyperparameter tuning for better performance.

---

## References
- **Paper**: [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- **PyTorch Documentation**: [https://pytorch.org/docs/](https://pytorch.org/docs/)
