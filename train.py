import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.optim import Adam
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
from transformer_block import VisionTransformer
from train_config import Train_Config
from transformer_config import Config
from tqdm import tqdm

from visualizer import visualize_batch, visualize_predictions

from torch.optim.lr_scheduler import CosineAnnealingLR


def train_model():
#  d_model = Train_Config.d_model
#  n_classes = Train_Config.n_classes
  img_size = Train_Config.img_size
#  patch_size = Train_Config.patch_size
#  n_channels = Train_Config.n_channels
#  n_heads = Train_Config.n_heads
#  n_layers = Train_Config.n_layers
  batch_size = Train_Config.batch_size
  epochs = Train_Config.epochs
  alpha = Train_Config.alpha

  transform = T.Compose([
    T.Resize(img_size),
    T.ToTensor()
  ])

  train_set = MNIST(
    root="./../datasets", train=True, download=True, transform=transform
  )
  test_set = MNIST(
    root="./../datasets", train=False, download=True, transform=transform
  )

  train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
  test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)

  visualize_batch(train_loader, num_images=8)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

  transformer = VisionTransformer(Config).to(device)

  optimizer = Adam(transformer.parameters(), lr=alpha)

  scheduler = CosineAnnealingLR(
    optimizer,
    T_max = epochs * len(train_loader),
    eta_min = alpha * 0.001 #min learning rate
  )

  criterion = nn.CrossEntropyLoss()

  for epoch in tqdm(range(epochs), desc="Epochs"):

    training_loss = 0.0
    for i, data in enumerate(tqdm(train_loader, desc="Training", leave = False), 0):
      inputs, labels = data
      inputs, labels = inputs.to(device), labels.to(device)

      optimizer.zero_grad()

      outputs = transformer(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      scheduler.step()

      training_loss += loss.item()

      if i % 20 == 0:  # print loss very 20 batches
        print(f'  Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

    print(f'Epoch {epoch + 1}/{epochs} loss: {training_loss  / len(train_loader) :.3f}')

    visualize_predictions(transformer, train_loader, num_images = 8, epoch = epoch + 1)

  correct = 0
  total = 0

  with torch.no_grad():
    for data in tqdm(test_loader, desc="Testing", leave = False):
      images, labels = data
      images, labels = images.to(device), labels.to(device)

      outputs = transformer(images)

      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
    print(f'\nModel Accuracy: {100 * correct // total} %')

if __name__ == "__main__":
  train_model()