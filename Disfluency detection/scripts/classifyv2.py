import torch
import torch.nn as nn
import torchaudio
import os
import csv  # Import the csv module
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_device() :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print('Device available is', device)
    return device


class StutterNet(nn.Module):
    def init(self, batch_size):
        super(StutterNet, self).init()
        # input shape = (batch_size, 1, 149,768)
        # in_channels is batch size
        self.layer1 = nn.Sequential(
            torch.nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=0.5)
        )
        self.layer1_bn = nn.BatchNorm2d(8)
        # input size = (batch_size, 8, 74, 384)
        self.layer2 = nn.Sequential(
            torch.nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=1, stride=2),
            torch.nn.Dropout(p=0.5)
        )
        self.layer2_bn = nn.BatchNorm2d(16)
        # input size = (batch_size, 16, 37, 192)
        self.flatten = torch.nn.Flatten()
        self.fc1 = nn.Linear(16* 37* 256,500, bias=True)
        self.fc1_bn = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500,250, bias=True)
        self.fc2_bn = nn.BatchNorm1d(250)
        self.fc3 = nn.Linear(250,100, bias=True)
        self.fc3_bn = nn.BatchNorm1d(100)
        self.fc4 = nn.Linear(100,10, bias=True)
        self.fc4_bn = nn.BatchNorm1d(10)
        self.fc5 = nn.Linear(10,2, bias=True)

        self.relu = nn.LeakyReLU()
        self.sm = nn.Softmax()
    
    def forward(self, x):
        #print('Before Layer1',np.shape(x))
        out = self.layer1(x)
        # out = self.layer1_bn(out)
        # print('After layer 1',np.shape(out))
        out = self.layer2(out)
        # out = self.layer2_bn(out)
        # print('After layer 2',np.shape(out))

        # Print shape before flattening
        print("Shape before flatten:", out.shape)  # Debugging line
        
        out  = self.flatten(out)

        out = self.fc1(out)
        out = self.relu(out)
        # out = self.fc1_bn(out)

        out = self.fc2(out)
        out = self.relu(out)
        # out = self.fc2_bn(out)

        out = self.fc3(out)
        out = self.relu(out)
        # out = self.fc3_bn(out)

        out = self.fc4(out)
        out = self.relu(out)
        # out = self.fc4_bn(out)

        out = self.fc5(out)
        out = self.sm(out)
        #print('After final ',np.shape(out))

        # log_probs = torch.nn.functional.log_softmax(out, dim=1)

        return out


device  = get_device()

# wav2vec2.0
bundle = torchaudio.pipelines.WAV2VEC2_XLSR53
print("Sample Rate of model:", bundle.sample_rate)
print("Audio backends:", torchaudio.list_audio_backends())

model_wav2vec = bundle.get_model().to(device)
## Convert audio to numpy to wav2vec feature encodings
def conv_audio_data(filename, device, bundle):
    audio_format = 'wav'
    waveform, sample_rate = torchaudio.load(filename, format=audio_format)

    # Convert stereo to mono if necessary
    if waveform.size(0) == 2:  # Check if the audio is stereo
        waveform = torch.mean(waveform, dim=0, keepdim=True)  # Averaging both channels
        
    waveform = waveform.to(device)
    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
    emission, _ = model_wav2vec(waveform)
    emission = emission.cpu().detach().numpy()
    return emission

def process_directory(all_audio_files, model, device, bundle, true_labels):
    results = []
    predictions = []

    for file_path in all_audio_files:
        filename = os.path.basename(file_path)
        print(f"Processing {file_path}")

        # Convert audio to feature encodings
        audio_features = conv_audio_data(file_path, device, bundle)

        # Make predictions
        with torch.no_grad():
            print("predicting")
            outputs = model(torch.tensor(audio_features).unsqueeze(0).to(device))
            _, predicted = torch.max(outputs.data, 1)
            print(f"Predicted for {filename}: {outputs}, {predicted}")
            results.append([filename, outputs.tolist(), predicted.item()])

        predictions.append(predicted.item())

    # Write results to CSV
    with open('prediction_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'Outputs', 'Predicted'])
        writer.writerows(results)

    print(f"Number of predictions: {len(predictions)}")
    print(f"Number of true labels: {len(true_labels)}") 

    # Check if lengths match
    if len(predictions) != len(true_labels):
        print("Error: Mismatch in the number of samples between predictions and true labels.")
        return

    # Verify the content of predictions and true_labels
    print(f"Sample predictions: {predictions[:10]}")
    print(f"Sample true labels: {true_labels[:10]}")

    # Generate the confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    print("Confusion Matrix:\n", cm)

    # Calculating the metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)

    # Print the metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Visualize the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='g', cmap='viridis')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Save the plot to a file
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as 'confusion_matrix.png'.")

def get_true_labels(audio_directory):
    all_audio_files = []
    true_labels = []

    for root, dirs, files in os.walk(audio_directory):
        for file in files:
            if file.lower().endswith('.wav'):
                file_path = os.path.join(root, file)
                all_audio_files.append(file_path)
                # Label as 1 if in "Block" subdirectory, else 0
                true_labels.append(1 if "/WordRep/" in file_path else 0)

    return all_audio_files, true_labels

model_path = "/content/drive/MyDrive/Ulima/Data/saves/DisfluencyNet_train_data_wp_quart.pth"

print("loading model")
model = torch.load(model_path)
print("model loaded")    
model.eval()

# Directory containing your audio files
audio_directory = "/content/drive/MyDrive/Ulima/Data/test_data"
df = pd.read_csv('/content/drive/MyDrive/Ulima/Data/stuttering_test.csv')

# Generate true labels
all_audio_files, true_labels = get_true_labels(audio_directory)

# Debugging: Print out the first few labels and the length of the labels list
print("First few true labels:", true_labels[:10])
print("Number of true labels:", len(true_labels))

# Debugging: Print out all true labels
print("All true labels:", true_labels)

# Continue with process_directory if true_labels is not empty
if len(true_labels) > 0:
    process_directory(all_audio_files, model, device, bundle, true_labels)
else:
    print("Error: True labels list is empty.")