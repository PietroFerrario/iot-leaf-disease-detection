import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

# Mean and STD 
def get_mean_std(dataset, batch_size = 32): 
    
    
    channels = 1    
    channels_sum = np.zeros(channels)
    channels_squared_sum = np.zeros(channels)
    num_batches = 0
    
    for batch, _ in dataset.batch(batch_size): 
        
        # sum over batch (0), height (2) and width (3), 3 channels for RGB
        imgs = batch.numpy() #(B,W,W,C)
        channels_sum += imgs.mean(axis=(0,1,2))
        channels_squared_sum += (imgs ** 2).mean(axis=(0,1,2))
        num_batches += 1
    
    mean = channels_sum / num_batches
    std = np.sqrt(channels_squared_sum / num_batches - mean**2)
    
    return mean, std

# Function for the NN

# # Training Loop
# def train_step(model, x, y, loss_fn, optimizer):

#     # Single training step 
#     with tf.GradientTape() as tape:
#         logits = model(x, training=True)
#         loss = loss_fn(y, logits)
#     grads = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradient(zip(grads, model.trainable_variables))
#     preds = tf.argmax(logits, axis=-1)
#     acc = tf.reduce_mean(tf.cast(tf.equal(preds,y), tf.float23))
#     return loss, acc
  
# Evaluation Loop 
def evaluate(model, dataset, loss_fn):
    # Evaluate model over dataset 
    
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for x,y in dataset:
        logits = model(x, training =False)
        loss = loss_fn(y, logits)
        preds = tf.argmax(logits, axis =-1)
        acc = tf.reduce_mean(tf.cast(tf.equal(preds,y), tf.floa32))
        batch_size = x.shape[0]
        total_loss += loss.numpy() * batch_size
        total_acc += acc.numpy() * batch_size
        n += batch_size

    return total_loss/ n, total_acc/n


# def fine_tune(model, train_ds, val_ds, loss_fn, optimizer, epochs, , pruner=None, scheduler=None):
    
#     best_pruned_checkpoint = dict()
#     best_pruned_accuracy = 0
#     model.to(device)
    
#     for epoch in range(num_epochs_finetune):
#         model.train() 
            
#         epoch_loss = 0
#         correct = 0 
        
#         for x,y in tqdm(train_loader, desc='Fine-Tuning', leave = False):
            
            
#             x = x.to(device)
            
#             y = y.to(device)
            
#             optimizer.zero_grad()
            
#             y_pred = model(x)
            
#             loss = criterion(y_pred, y)
#             loss.backward()
#             optimizer.step()
            
#              # Apply pruner mask 
#             if pruner: 
#                 pruner.apply(model) 
                
#             # Apply Scheduler 
#             if scheduler:
#                 scheduler.step()
            
#             epoch_loss += loss.item()
#             correct += (y_pred.argmax(dim=1)==y).sum().item()
        
#         train_loss = epoch_loss / len(train_loader)
#         train_acc = correct / len(train_loader.dataset)
        
#         val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
#         is_best = val_acc > best_pruned_accuracy 
#         if is_best:
#             best_pruned_checkpoint['state_dict'] = copy.deepcopy(model.state_dict())
#             best_pruned_accuracy = val_acc
            
#         print(f'\t Epoch {epoch+1}/{num_epochs_finetune} | Fine Tune Train  Loss: {train_loss:.5f}')
#         print(f'\t Fine Tune Val. Loss: {val_loss:.5f} | Fine Tune Valid Accuracy = {val_acc*100:.4f}% | Best Fine Tune Accuracy: {best_pruned_accuracy*100:2f}%')
        
#     return best_pruned_checkpoint, best_pruned_accuracy
            
    
    
  
# Time Stamp Function 
def epoch_time(start, end):
    elapsed = end - start
    return int(elapsed//60), int(elapsed % 60)
  
  # Plotting Training results 
def plot_train_valid_loss_accuracy(train_losses, valid_losses, test_loss, train_accuracy, valid_accuracy, epochs):

    x = np.arange(1, epochs+1)
    plt.figure(figsize=(12,5))

    # Plotting the Loss 
    plt.subplot(1,2,1)
    plt.plot(x, train_losses, label='Training Loss', linestyle='-')
    plt.plot(x, valid_losses, label='Validation Loss', linestyle='--')
    plt.plot(x, np.repeat(test_loss,epochs), label='Test Loss', linestyle=':')

    plt.title("Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plotting the Accuracy
    plt.subplot(1,2,2)
    plt.plot(x, train_accuracy, label='Training Accuracy', linestyle='-')
    plt.plot(x, valid_accuracy, label='Validation Accuracy', linestyle='--')
    
    plt.title("Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig('Classifier_Training_Accuracy_Loss.png')
    plt.show()

def get_prediction(model, dataset):
    """
    Run model.predict on dataset, return all_images, all_true_labels, all_pred_probs
    """
    
    imgs = []
    labels = []
    probs = []
    
    for x, y in dataset:
        
        p = model.predict(x, verbose=0)
        imgs.append(x.numpy())
        labels.append(y.numpy())
        probs.append(p)
    
    return np.concatenate(imgs), np.concatenate(labels), np.concatenate(probs)

def plot_confusion_matrix(true_labels, pred_labels): 
    
    cm = metrics.confusion_matrix(true_labels, pred_labels)
    
    cm_display = metrics.ConfusionMatrixDisplay(cm, display_labels=range(2))
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1)
    
    cm_display.plot(values_format='d', cmap='Blues', ax=ax)
    
    plt.title('Confusion Matrix')
    plt.savefig('Confusion_matrix.png')
    plt.show()
    
