uimport torch
import numpy as np
import pandas as pd
from model import StutterNet
from dataset import get_dataloader, AudioDataset, load_dataset_from_path
from utils import __get_device__, Wav2VecRepresentation
from sklearn.metrics import f1_score

import os

def load_model(model_path, device):
    model = StutterNet(batch_size=32)  # Ensure this matches your model's architecture
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def test_model(model, test_loader, device, output_csv_path):
    model.eval()
    total = correct = predicted_stutter = labels_stutter = correct_stutter = 0
    results = []
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            probabilities = torch.sigmoid(outputs)[:, 0]  # Apply sigmoid to the output
            predicted = (probabilities > 0.5).long()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            results.extend(zip(labels.cpu().numpy(), predicted.cpu().numpy(), probabilities.cpu().numpy()))
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate metrics
    final_labels, final_predictions, probabilities = zip(*results)
    final_labels = np.array(final_labels)
    final_predictions = np.array(final_predictions)
    probabilities = np.array(probabilities)

    # Generating confusion matrix
    cm = confusion_matrix(final_labels, final_predictions)
    print("Confusion Matrix:\n", cm)

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
    print(f'Accuracy: {100 * correct / total:.2f}%')

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
    test_dataset = AudioDataset(x_test, y_test, np.shape(x_test)[0])
    test_loader = get_dataloader(test_dataset, batch_size=32, shuffle=True)

    output_csv_path = 'test_predictions.csv'
    if not os.path.exists(output_csv_path):
        print("Creating a new CSV file for predictions.")
    else:
        print("CSV file already exists and will be overwritten.")
    
    test_model(model, test_loader, device, output_csv_path)