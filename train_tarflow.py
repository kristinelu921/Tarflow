import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.optim import AdamW
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
from transformer_config import Config as cfg
from tqdm import tqdm
import matplotlib.pyplot as plt

from visualizer import visualize_batch, visualize_predictions,log_metrics, plot_loss_curve

from tarflow_transformer_block import Tarflow

from noise_adaptations import add_noise as add_noise
from torch.optim.lr_scheduler import CosineAnnealingLR

from generator import Generator

def train_model(model, config): #mnist trainer
  cfg  = config
  img_size = (cfg.img_size, cfg.img_size)
  batch_size = cfg.batch_size
  epochs = cfg.epochs

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

  #visualize_batch(train_loader, num_images=8)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

  my_model =  model

  #TODO: change lr
  optimizer = AdamW(my_model.parameters(), lr=cfg.initial_lr, weight_decay = cfg.weight_decay, betas = (0.9, 0.95))

  print("before scheduler", optimizer.param_groups[0]["lr"])

  scheduler = cosine_scheduler_with_warmup(cfg, optimizer)

  loss_fn = my_model.loss

  for epoch in tqdm(range(epochs), desc="Epochs"):

    training_loss = 0.0
    for i, data in enumerate(tqdm(train_loader, desc="Training", leave = False), 0):
      inputs, _ = data
      inputs = inputs.to(device)

      optimizer.zero_grad()

      outputs, alphas, log_dets = my_model.encode(inputs)
      loss = loss_fn(outputs, log_dets)
      loss.backward()
      optimizer.step()

      if cfg.has_scheduler: #if we want a lr scheduler
        scheduler.step()
        #print("stepped")

      training_loss += loss.item()
      #print("after scheduler", optimizer.param_groups[0]["lr"])
      if i % 1 == 0:  # print loss very 20 batches
        print(f'  Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

    print(f'Epoch {epoch + 1}/{epochs} loss: {training_loss  / len(train_loader) :.3f}')

    #visualizing/logging
    #visualize_predictions(transformer, train_loader, num_images = 8, epoch = epoch + 1)
    log_metrics(epoch, i, loss.item(), optimizer.param_groups[0]["lr"])

  # For example, visualize the first layer weights of alpha_head
  #alpha_weights = my_model.transformer_flow_blocks[0].alpha_head[0].weight.detach().cpu()

  #plt.figure(figsize=(10, 8))
  #plt.imshow(alpha_weights, cmap='viridis')
  #plt.colorbar()
  #plt.title("Alpha Head Weights")
  #plt.savefig("alpha_weights.png")
  #plt.show()

  # Save named parameters to a text file
  def save_params_to_txt(model, filename='model_params.txt'):
      with open(filename, 'w') as f:
          for name, param in model.named_parameters():
              f.write(f"Layer: {name} | Shape: {param.shape}\n")
              f.write(f"  Min: {param.data.min().item()}, Max: {param.data.max().item()}\n")
              f.write(f"  Mean: {param.data.mean().item()}, Std: {param.data.std().item()}\n")
              f.write("----------------------------\n")
      print(f"Parameters saved to {filename}")

  # Call this function after training
  save_params_to_txt(my_model)

  generator = Generator(cfg)
  generated_images = generator.generate_latents(my_model, num_samples = 16, device = device)
  generator.visualize_generated_images(generated_images)

  correct = 0
  total = 0

  if cfg.evaluate:
    with torch.no_grad():
      for data in tqdm(test_loader, desc="Testing", leave = False):
        images, labels = data
      images, labels = images.to(device), labels.to(device)

      outputs = my_model(images)

      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
    print(f'\nModel Accuracy: {100 * correct // total} %')

  plot_loss_curve()


def cosine_scheduler_with_warmup(cfg, optimizer):
  """
  config consists of:
  num_warmup_steps, num_training_steps, lr_min, lr_max, weight_decay, 

  creates a lambda scheduler function that returns the learning rate for a given step

  warmup goes from lr_min to lr_max in num_warmup_steps

  cosine decay goes from lr_max back to lr_min in remaining # steps
  """
  def lr_lambda(current_step):
    #warmup phase
    if current_step < cfg.num_warmup_steps:
      print("lr", cfg.lr_min + (cfg.lr_max - cfg.lr_min) * current_step / cfg.num_warmup_steps)

      return cfg.lr_min + (cfg.lr_max - cfg.lr_min) * current_step / cfg.num_warmup_steps
    
    #cosine decay phase
    progress = (current_step - cfg.num_warmup_steps) / (cfg.total_training_steps - cfg.num_warmup_steps) #which step we're on of training

    cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(torch.pi) * progress))

    lr = cfg.lr_min + (cfg.lr_max - cfg.lr_min) * cosine_decay
    return lr
  
  return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    

if __name__ == "__main__":
  train_model(Tarflow(cfg), cfg)
