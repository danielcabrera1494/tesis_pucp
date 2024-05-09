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

    f1 = f1_score(all_labels, all_predictions, average='micro')

    print(f'Validation Loss: {eval_loss:.3f} | Accuracy: {accu:.3f} | F1: {f1:.3f}')
    return eval_loss, accu, f1


set_seed(42)
device = __get_device__()
wav2vec_rep = Wav2VecRepresentation(device)

subset = "train"
disfluency = "SoundRep"

stutter_train_path = f'/content/drive/MyDrive/Ulima/Data/{subset}_data/{disfluency}'
fluent_train_path = f'/content/drive/MyDrive/Ulima/Data/{subset}_data/NoStutteredWords'
x_train, y_train = load_dataset_from_path(stutter_train_path, 
                                            fluent_train_path, 
                                            wav2vec_rep, balance=True)


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
num_epochs = 50 #150
learning_rate = 0.0003 #0.0001
output_path = f'ckp_stutternet_{disfluency}'

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
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.998), weight_decay=0.001)  

epochs=num_epochs
patience = 100
min_f1 = 0.0

for epoch in range(1,epochs+1):

    if patience == 0:
        print('No more patience')
        break

    print(f'EPOCH {epoch} ...')
    train(model, train_loader, optimizer, criterion)
    _,_,f1 = evaluate(model, val_loader, criterion)

    if f1 > min_f1:
        min_f1 = f1
        patience = 100
        #torch.save(model.state_dict(), output_path + f'_{epoch}.pt')
        torch.save(model.state_dict(), output_path + '.pt')
    else:
        print(f'Decreasing patience from {patience} to {(patience-1)}')
        patience -= 1

