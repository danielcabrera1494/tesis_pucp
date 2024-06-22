import numpy as np
import torch
import torch.nn as nn
from dataset import load_dataset_from_path
from utils import set_seed, __get_device__
from utils import Wav2VecRepresentation

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from dataset import AudioDataset, get_dataloader
from model import StutterNet

import pandas as pd
import os
import shutil

def train(model, loader, optimizer, criterion):
    model.train()
    running_loss=0
    correct=0
    total=0
    
    for data in loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        # forward pass
        outputs=model(inputs)
        #loss=criterion(outputs,labels)
        loss = criterion(outputs.reshape(-1), labels.float())
        #print(loss)
        
        # backward and optimise
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Compute Training Accuracy
        #_, predicted_labels = torch.max(out)
        predicted_labels = torch.tensor(np.where(outputs.cpu().data.reshape(-1) > 0.5, 1, 0))
        #print(predicted_labels)
        total += labels.size(0)
        correct += predicted_labels.eq(labels.cpu()).sum().item()
        
    train_loss = running_loss/len(loader)
    accu = 100.*correct/total
    print(f'Train Loss: {train_loss:.3f} | Accuracy: {accu:.3f}')
    return train_loss, accu  # Add this line to return the loss and accuracy


def evaluate(model, loader, criterion):
    model.eval()
    
    running_loss=0
    correct=0
    total=0
    
    # These lists will store all the predictions and true labels
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for data in loader:
            features, labels = data[0].to(device), data[1].to(device)
            outputs=model(features)
            
            #_, predicted_valid = torch.max(outputs.data, 1)
            predicted_labels = torch.tensor(np.where(outputs.cpu().data.reshape(-1) > 0.5, 1, 0))
            #all_predictions.extend(predicted_valid.cpu().numpy())
            all_predictions.extend(predicted_labels.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            #loss= criterion(outputs,labels)
            loss = criterion(outputs.reshape(-1), labels.float())
            running_loss+=loss.item()
            
            total += labels.size(0)
            #correct += predicted_valid.eq(labels.cpu()).sum().item()
            correct += predicted_labels.eq(labels.cpu()).sum().item()

    eval_loss = running_loss/len(loader)
    accu = 100.* correct/total

    f1 = f1_score(all_labels, all_predictions)#, average='micro')

    print(f'Validation Loss: {eval_loss:.3f} | Accuracy: {accu:.3f} | F1: {f1:.3f}')
    return eval_loss, accu, f1


set_seed(42)
device = __get_device__()
wav2vec_rep = Wav2VecRepresentation(device)

subset = "augment_x2_train"
disfluency = "SoundRep"
train_balance = False

stutter_train_path = f'/content/drive/MyDrive/Ulima/Data/{subset}_data/{disfluency}'
fluent_train_path = f'/content/drive/MyDrive/Ulima/Data/{subset}_data/NoStutteredWords'
x_train, y_train = load_dataset_from_path(stutter_train_path, 
                                            fluent_train_path, 
                                            wav2vec_rep, balance=train_balance)


subset = "test"
stutter_train_path = f'/content/drive/MyDrive/Ulima/Data/{subset}_data/{disfluency}'
fluent_train_path = f'/content/drive/MyDrive/Ulima/Data/{subset}_data/NoStutteredWords'
x_test, y_test = load_dataset_from_path(stutter_train_path,
                                            fluent_train_path,
                                            wav2vec_rep, balance=False)


subset = "val"
stutter_train_path = f'/content/drive/MyDrive/Ulima/Data/{subset}_data/{disfluency}'
fluent_train_path = f'/content/drive/MyDrive/Ulima/Data/{subset}_data/NoStutteredWords'
x_val, y_val = load_dataset_from_path(stutter_train_path,
                                            fluent_train_path,
                                            wav2vec_rep, balance=False)


n_samples_train = np.shape(x_train)[0]
n_samples_val = np.shape(x_val)[0]
n_samples_test = np.shape(x_test)[0]

print('Number of samples to train = ', n_samples_train)
print('Number of samples to validate = ', n_samples_val)
print('Number of samples to test = ', n_samples_test)

batch_size = 32 #128
num_epochs = 150 #150
learning_rate = 0.001 #0.0001
output_path = f'ckp_stutternet_{disfluency}_{train_balance}_x2'

train_dataset = AudioDataset(x_train,y_train, n_samples_train)
val_dataset = AudioDataset(x_val, y_val, n_samples_val)
test_dataset = AudioDataset(x_test,y_test,n_samples_test)

train_loader = get_dataloader(train_dataset, batch_size, shuffle=True)
val_loader = get_dataloader(val_dataset, batch_size)


model = StutterNet(batch_size).to(device)
#print(model)
# Loss and optimizer
#criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.01)  

epochs=num_epochs
patience = 40
total_patience = patience
min_f1 = -1.0
best_epoch = 0

epochs_list = []
train_losses = []
val_losses = []

for epoch in range(1,epochs+1):

    if patience == 0:
        print('No more patience')
        break

    print(f'EPOCH {epoch} ...')
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    val_loss, val_acc, f1 = evaluate(model, val_loader, criterion)

    epochs_list.append(epoch)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if f1 > min_f1:
        min_f1 = f1
        patience = 40
        best_epoch = epoch
        #torch.save(model.state_dict(), output_path + f'_{epoch}.pt')
        torch.save(model.state_dict(), output_path + '.pt')
    else:
        print(f'Decreasing patience from {patience} to {(patience-1)}')
        patience -= 1

# After the training loop, save the collected data to a CSV file
results_df = pd.DataFrame({
    'epoch': epochs_list,
    'train_loss': train_losses,
    'val_loss': val_losses
})
results_df.to_csv(f'epoch_losses_{disfluency}_{train_balance}.csv', index=False)
print(f'Losses per epoch saved to epoch_losses_{disfluency}_{train_balance}.csv')

#Save checkpoint in Google Drive
source_path = '/content/tesis_pucp/Disfluency detection/scripts/'+output_path + '.pt'
destination_path = '/content/drive/MyDrive/Ulima/Data/saves/tesis/'+output_path + '.pt'
# Copy the file
shutil.copyfile(source_path, destination_path)
print("File copied successfully!")

print(f'The best model was saved at epoch {best_epoch} with an F1 score of {min_f1:.3f}')
#################################### PREDICTIONS ##############################################

def test_model(model, test_loader, device, output_csv_path):
    model.eval()
    total = correct = predicted_stutter = labels_stutter = correct_stutter = 0
    results = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            probabilities = torch.sigmoid(outputs)[:, 0]  # Apply sigmoid and get probabilities for the positive class
            predicted = (probabilities > 0.5).long()  # Convert probabilities to 0 or 1

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect data for CSV
            results.extend(zip(labels.cpu().numpy(), predicted.cpu().numpy(), probabilities.cpu().numpy()))

            print("Predicted:", predicted)

    # Calculate metrics
    final_labels, final_predictions, probabilities = zip(*results)
    final_labels = np.array(final_labels)
    final_predictions = np.array(final_predictions)
    probabilities = np.array(probabilities)
    predicted_stutter = np.sum(final_predictions == 1)
    labels_stutter = np.sum(final_labels == 1)
    correct_stutter = np.sum((final_predictions == 1) & (final_labels == 1))

    acc_test = 100 * correct / total
    recall = correct_stutter / labels_stutter if labels_stutter > 0 else 0
    precision = correct_stutter / predicted_stutter if predicted_stutter > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f'Accuracy on test dataset: {acc_test:.2f}%')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1_score:.2f}')

    # Save results to CSV
    results_df = pd.DataFrame({
        "Actual Label": final_labels,
        "Predicted Label": final_predictions,
        "Probability": probabilities
    })
    results_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")

# Set the output path and check if it exists
output_csv_path = 'test_predictions.csv'
if not os.path.exists(output_csv_path):
    print("Creating a new CSV file for predictions.")
else:
    print("CSV file already exists and will be overwritten.")

# Assuming the test dataset and model are already set up correctly
test_loader = get_dataloader(test_dataset, batch_size, shuffle=True)
test_model(model, test_loader, device, output_csv_path)