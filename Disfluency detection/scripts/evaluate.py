import torch
import numpy as np
import pandas as pd
from model import StutterNet
from dataset import get_dataloader, AudioDataset, load_dataset_from_path
from utils import __get_device__, Wav2VecRepresentation
from sklearn.metrics import f1_score

def load_model(model_path, device):
    model = StutterNet(batch_size=1)  # Ensure this matches your model's architecture
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def test_model(model, test_loader, device, output_csv_path):
    model.eval()
    total = correct = predicted_stutter = labels_stutter = correct_stutter = 0
    results = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            probabilities = torch.sigmoid(outputs)[:, 0]  # Apply sigmoid
            predicted = (probabilities > 0.5).long()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            results.extend(zip(labels.cpu().numpy(), predicted.cpu().numpy(), probabilities.cpu().numpy()))

    # Metrics and CSV
    final_labels, final_predictions, probabilities = zip(*results)
    results_df = pd.DataFrame({
        "Actual Label": np.array(final_labels),
        "Predicted Label": np.array(final_predictions),
        "Probability": np.array(probabilities)
    })
    results_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")

if __name__ == '__main__':
    device = __get_device__()
    model_path = 'ckp_stutternet_SoundRep.pt'
    model = load_model(model_path, device)

    # Setup the data - assuming you have functions to create/load your dataset
    wav2vec_rep = Wav2VecRepresentation(device)

    subset = "test"
    disfluency = "SoundRep"
    
    stutter_test_path = '/content/drive/MyDrive/Ulima/Data/test_data/SoundRep'
    fluent_test_path = '/content/drive/MyDrive/Ulima/Data/test_data/NoStutteredWords'
    x_test, y_test = load_dataset_from_path(stutter_test_path, fluent_test_path, wav2vec_rep, balance=False)
    test_dataset = AudioDataset(x_test, y_test, len(x_test))
    test_loader = get_dataloader(test_dataset, batch_size=32, shuffle=True)

    output_csv_path = 'test_predictions.csv'
    if not os.path.exists(output_csv_path):
        print("Creating a new CSV file for predictions.")
    else:
        print("CSV file already exists and will be overwritten.")
    
    test_model(model, test_loader, device, output_csv_path)
