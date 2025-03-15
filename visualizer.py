import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize_batch(dataloader, num_images = 5):

    #get batch
    data = iter(dataloader)
    images, labels = next(data)

    #plot images
    plt.figure(figsize = (12, 4))
    for i in range(min(num_images, len(images))):
        plt.subplot(1, num_images, i+1)
        img = images[i].squeeze().numpy()
        plt.imshow(img, cmap = 'gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    
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
