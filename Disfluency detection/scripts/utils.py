import os
import torch
import random
import numpy as np
import torchaudio

def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def __get_device__() :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print('Device available is', device)
    return device


class Wav2VecRepresentation:

    def __init__(self, device):
        self._device = device

        self._bundle = torchaudio.pipelines.WAV2VEC2_XLSR53
        print("Sample Rate of model:", self._bundle.sample_rate)

        self._model_wav2vec = self._bundle.get_model().to(device)

    def get_representation(self, audio_file):
        audio_format = 'wav'
        waveform, sample_rate = torchaudio.load(audio_file, format = audio_format)
        waveform = waveform.to(self._device)
        if sample_rate != self._bundle.sample_rate:
            #print('Mismatched sample rate')
            waveform = torchaudio.functional.resample(waveform, sample_rate,
                                                        self._bundle.sample_rate)
        emission, _ = self._model_wav2vec(waveform)
        emission = emission.cpu().detach().numpy()
        return emission
