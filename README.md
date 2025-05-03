**1. Imports and Device Setup**
      torch: Core PyTorch library.
      
      torch.nn: Tools for building neural networks.
      
      torch.optim: Optimizers like Adam or SGD.
      
      torchvision: Datasets, pre-trained models, and image transformations.
      
      transforms: Image preprocessing utilities (e.g., converting to tensors, normalization).

**2. Dataset Loading**
      transform = transforms.Compose([transforms.ToTensor()])
      
      " Wraps image preprocessing in a Compose container.
      Converts PIL images into PyTorch tensors. "
      
      train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
      train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
      
      "Loads CIFAR-10 training data (60,000 32×32 RGB images).
      batch_size=64: 64 images per training step.
      shuffle=True: Shuffles data to improve learning.
      num_workers=2: Uses 2 subprocesses for loading."

      
      test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
      test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

      
     " Loads test set in a similar fashion. "
     
**3. Model Definition – ColorizationNet**
      A simple convolutional neural network (CNN) designed for image colorization.
      Input: grayscale image (1 channel).
      Output: 2-channel tensor (likely chrominance channels a, b in Lab color space).

**4. Grayscale Conversion**
      Converts a 3-channel RGB tensor to grayscale using standard luminance weights.
      Returns a single-channel grayscale tensor.

**5. Training Setup**
    Initializes the colorization model.
    Loss: Mean Squared Error (MSE) between predicted color and ground truth.
    Optimizer: Adam with learning rate 0.001.

**6. Training Loop**
    Trains the model for 10 epochs.
    Each batch:
    Converts images to grayscale inputs.
    Predicts 2-channel color output.
    Compares with original color channels.
    Updates weights using gradient descent.

**7. Visualization of Results**
    Shows visual comparisons:
    Grayscale input
    Original color
    Colorized output
    Uses matplotlib to display side-by-side images.
    
**Summary of the Full Code**
    Trains a CNN to colorize grayscale images from the CIFAR-10 dataset.
    Converts RGB images to grayscale and then predicts color back.
    Uses MSE loss to measure pixel-wise differences between predicted and actual color channels.
    Includes both training and visual evaluation.

      


