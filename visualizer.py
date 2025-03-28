import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize_batch(dataloader, num_images = 5, save_path = 'initial_images.png'):

    #get batch
    data = iter(dataloader)
    images = next(data)
    #plot images
    plt.figure(figsize = (12, 4))
    for i in range(min(num_images, len(images))):
        plt.subplot(1, num_images, i+1)
        img = images[i].squeeze().numpy()
        plt.imshow(img, cmap = 'gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    
    print("PLOTTED")
    plt.savefig(save_path)
    plt.tight_layout()
    plt.close()

def visualize_predictions(model, dataloader, num_images = 8, epoch = None, save_dir = './plots'):
    #set to eval
    model.eval()

    #get batch
    dataiter = iter(dataloader)
    images, labels = next(dataiter) 

    with torch.no_grad():
        inputs = images.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)     
        
        plt.figure(figsize = (12, 4))   
        for i in range(min(num_images, len(images))):
            plt.subplot(1, num_images, i+1)
            img = images[i].squeeze().numpy()
            plt.imshow(img, cmap = 'gray')
            plt.title(f"Pred: {predicted[i]}, True: {labels[i]}")

            color = 'green' if predicted[i] == labels[i] else 'red'
            plt.title(f"Pred: {predicted[i]}, True: {labels[i]}", color = color)

            plt.axis('off')
    plt.tight_layout()

    if epoch is not None:
        import os
        os.makedirs('logs', exist_ok = True)
        plt.savefig(f'logs/predictions_epoch_{epoch}.png')

    plt.close()
        

def log_metrics(epoch, batch_idx, loss, lr, filename = 'logs/training_log.txt'):
    with open(filename, 'a') as f:
        f.write(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}, LR: {lr:.6f}\n")

def plot_loss_curve(log_file='logs/training_log.txt', save_path='logs/loss_curve.png'):
    epochs = []
    losses = []
    learning_rates = []
    
    with open(log_file, 'r') as f:
        for line in f:
            parts = line.strip().split(', ')
            epoch = int(parts[0].split()[1])
            loss = float(parts[2].split(': ')[1])
            lr = float(parts[3].split(': ')[1])
            
            epochs.append(epoch)
            losses.append(loss)
            learning_rates.append(lr)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    line1 = ax1.plot(epochs, losses, 'b-', label='Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    line2 = ax2.plot(epochs, learning_rates, 'r-', label='Learning Rate')
    ax2.set_ylabel('Learning Rate', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    
    plt.title('Training Loss and Learning Rate over Time')
    plt.savefig(save_path)
    plt.close()