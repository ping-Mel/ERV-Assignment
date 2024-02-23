import torch
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import math
from tqdm import tqdm


def assign_compute():
    isGPU = torch.cuda.is_available()
    return torch.device("cuda" if isGPU else "cpu")



def apply_transforms():
    transform = transforms.Compose([
        transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
        transforms.Resize((28, 28)),
        transforms.RandomRotation((-15., 15.), fill=0),
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.1307,), (0.3081,)),# Normalize the images using the well-established mean and standard deviation statistics derived from the MNIST dataset
    ])
    
    return transform

def get_dataset(MNISTPath='../data', train=False, download=False, custom_transform=None):
    """
    Load the MNIST dataset with specified parameters and optional custom transformations.

    Parameters:
    - path (str): Path to save/load the MNIST dataset. Defaults to './MNIST_data'.
    - train (bool): Whether to load the training set (True) or the test set (False). Defaults to True.
    - download (bool): Whether to download the dataset if not found at the path. Defaults to True.
    - custom_transform (callable): Custom transformation to apply to the dataset. Defaults to None.

    Returns:
    - The MNIST dataset loaded as per the specified parameters.
    """
    if custom_transform is None:
        custom_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    dataset = datasets.MNIST(MNISTPath, train = train, download = download, transform=custom_transform)

    return dataset

def load_dataset(data, **kwargs):
    return DataLoader(data, **kwargs)

def plot_images(data_loader, image_count):
    batch_data, batch_label = next(iter(data_loader))  # Fetch a batch of data and corresponding labels.
    
    # A simple strategy to calculate nrows and ncols
    ncols = 4  # Set a fixed number of columns
    nrows = math.ceil(image_count / ncols)  # Calculate the number of rows needed
    
    fig = plt.figure(figsize=(15, nrows * 2.5))  # Adjust figure size based on number of rows

    for i in range(image_count):  # Loop based on the range_number
        plt.subplot(nrows, ncols, i + 1)  # Create the grid and specify index of the subplot
        plt.tight_layout()  # Adjust subplot parameters to prevent overlap
        plt.imshow(batch_data[i].squeeze(0), cmap='gray')  # Display the image in grayscale
        plt.title(batch_label[i].item())  # Set title to the label of the image
        plt.xticks([])  # Remove x-axis tick marks
        plt.yticks([])  # Remove y-axis tick marks


def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train(model, device, train_loader, optimizer, criterion, train_acc, train_losses): #missing epoch? 
  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    # Predict
    pred = model(data)

    # Calculate loss
    loss = criterion(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()

    correct += GetCorrectPredCount(pred, target)
    processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

  train_acc.append(100*correct/processed)
  train_losses.append(train_loss/len(train_loader))

def test(model, device, test_loader, criterion, test_acc, test_losses):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item()  # remove reduction sum up batch loss

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
def plot_train_result(train_losses, train_acc, test_losses, test_acc):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Plot Training Loss
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")

    # Plot Training Accuracy
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")

    # Plot Test Loss
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")

    # Plot Test Accuracy
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")

    plt.show() # display the plot