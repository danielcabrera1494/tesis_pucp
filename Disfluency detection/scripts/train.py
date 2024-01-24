import numpy as np
import torch
import torch.nn as nn
from dataset import load_dataset_from_path
from utils import set_seed, __get_device__
from utils import Wav2VecRepresentation

from sklearn.model_selection import train_test_split

from utils import AudioDataset, get_dataloader
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
        loss=criterion(outputs,labels)
        
        # backward and optimise
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Compute Training Accuracy
        _, predicted_labels = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted_labels.eq(labels).sum().item()
        
    train_loss = running_loss/len(loader)
    accu = 100.*correct/total
    print(f'Train Loss: {train_loss:3f} | Accuracy: {accu:3f}')


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
            
            _, predicted_valid = torch.max(outputs.data, 1)
            all_predictions.extend(predicted_valid.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            loss= criterion(outputs,labels)
            running_loss+=loss.item()
            
            total += labels.size(0)
            correct += predicted_valid.eq(labels).sum().item()

    eval_loss = running_loss/len(loader)
    accu = 100.* correct/total
    print(f'Validation Loss: {eval_loss:3f} | Accuracy: {accu:3f}')

set_seed(42)
device = __get_device__()
wav2vec_rep = Wav2VecRepresentation(device)

subset = "train"

stutter_train_path = f'/data/msobrevilla/audio/{subset}_data/Block'
fluent_train_path = f'/data/msobrevilla/audio/{subset}_data/NoStutteredWords'
x_train, y_train = load_dataset_from_path(stutter_train_path, 
                                            fluent_train_path, 
                                            wav2vec_rep, balance=True)


subset = "test"
stutter_train_path = f'/data/msobrevilla/audio/{subset}_data/Block'
fluent_train_path = f'/data/msobrevilla/audio/{subset}_data/NoStutteredWords'
x_test, y_test = load_dataset_from_path(stutter_train_path,
                                            fluent_train_path,
                                            wav2vec_rep, balance=False)


x_train_n, x_dev, y_train_n, y_dev = train_test_split(x_train, 
                                                        y_train, 
                                                        test_size=0.15, 
                                                        random_state=123, 
                                                        shuffle=True, 
                                                        stratify = y_train)

n_samples_train = np.shape(x_train)[0]
n_samples_dev = np.shape(x_dev)[0]
n_samples_test = np.shape(x_test)[0]

print('Number of samples to train = ', n_samples_train)
print('Number of samples to validate = ', n_samples_dev)
print('Number of samples to test = ', n_samples_test)

batch_size = 64
num_epochs = 5
learning_rate = 0.0001

train_dataset = AudioDataset(x_train,y_train, n_samples_train)
dev_dataset = AudioDataset(x_dev, y_dev, n_samples_dev)
test_dataset = AudioDataset(x_test,y_test,n_samples_test)

train_loader = get_dataloader(train_dataset, batch_size, shuffle=True)
dev_loader = get_dataloader(dev_dataset, batch_size)


model = StutterNet(batch_size).to(device)
print(model)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

epochs=num_epochs
for epoch in range(1,epochs+1):
    print('EPOCH {epoch} ...')
    train(model, train_loader, optimizer, criterion)
    evaluate(model, dev_loader, criterion)