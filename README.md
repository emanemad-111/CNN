# ResNet50 Image Classification Project

This repository contains a project for training and evaluating a ResNet50 model for image classification using PyTorch. The dataset is loaded from a specified directory and preprocessed using common image transformations. The final model is fine-tuned on the dataset and evaluated on test data.

## Features

- Utilizes a pre-trained ResNet50 model from PyTorch's `torchvision.models`.
- Supports data augmentation techniques (random flips, rotations, resizing).
- Splits the dataset into training, validation, and test sets.
- Includes a training loop with validation tracking.
- Saves the best-performing model based on validation accuracy.
- Computes and reports test accuracy after training.

## Prerequisites

To run this project, ensure you have the following installed:

- Python 3.7+
- PyTorch 1.9+
- torchvision 0.10+
- matplotlib

You can install the required libraries using:

```bash
pip install torch torchvision matplotlib
```

## Dataset Preparation

1. Organize your dataset into the following structure:

   ```
   path_to_dataset/
   ├── class_1/
   │   ├── img1.jpg
   │   ├── img2.jpg
   │   └── ...
   ├── class_2/
   │   ├── img1.jpg
   │   ├── img2.jpg
   │   └── ...
   └── ...
   ```

2. Replace `path_to_dataset` in the script with the actual path to your dataset directory.

## Usage

### Training the Model

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/ResNet50-Image-Classification.git
   cd ResNet50-Image-Classification
   ```

2. Run the training script:

   ```bash
   python train.py
   ```

3. The script will display training and validation losses and accuracies for each epoch.

### Saving the Best Model

- The model with the best validation accuracy is saved as `best_model.pth` in the working directory.

### Testing the Model

- After training, the script evaluates the model on the test set and prints the test accuracy.

## Customization

- **Learning Rate**: Modify the learning rate by changing the `lr` parameter in the optimizer.
- **Batch Size**: Adjust the `batch_size` in the DataLoader to handle larger or smaller datasets.
- **Number of Epochs**: Set the desired number of epochs in the `epochs` variable.
- **Transforms**: Edit the `img_transforms` to include or exclude specific preprocessing steps.

## Results

- The model achieves high accuracy on the test dataset, depending on the quality and quantity of your data.
- You can visualize the loss and accuracy trends during training by extending the script with Matplotlib plots.

## Contributing

Contributions are welcome! Feel free to fork the repository and submit a pull request for improvements or new features.


