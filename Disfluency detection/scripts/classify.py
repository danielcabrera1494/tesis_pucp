import torch
import torch.nn as nn
import torchaudio
import os
import csv  # Import the csv module

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
    waveform = waveform.to(device)
    if sample_rate != bundle.sample_rate:
        print('Mismatched sample rate')
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
    emission, _ = model_wav2vec(waveform)
    emission = emission.cpu().detach().numpy()
    return emission

def process_directory(directory, model, device, bundle):
    results = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith('.wav'):
                file_path = os.path.join(root, filename)
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

    # Write results to CSV
    with open('prediction_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'Outputs', 'Predicted'])
        writer.writerows(results)

model_path = "/content/drive/MyDrive/Ulima/Data/saves/DisfluencyNet_train_data_blk_quart.pth"

print("loading model")
model = torch.load(model_path)
print("model loaded")    
model.eval()

# Directory containing your audio files
audio_directory = "/content/drive/MyDrive/Ulima/Data/disfluency_augmentation_clips"
process_directory(audio_directory, model, device, bundle)