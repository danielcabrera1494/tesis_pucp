import torch
import augment
import torchaudio
import numpy as np

# Generate a random shift applied to the speaker's pitch
def random_pitch_shift():
    return np.random.randint(-300, 300)

# Generate a random size of the room
def random_room_size():
    return np.random.randint(0, 100)

def _pitch_reverb(audio_waveform, sr):
        combination = augment.EffectChain() \
            .pitch("-q", random_pitch_shift).rate(sr) \
            .reverb(50, 50, random_room_size).channels(1) 
        y = combination.apply(audio_waveform, src_info={'rate': sr}, target_info={'rate': sr})
        return y


def _pitch_add_reverb(audio_waveform, sr):
    noise_generator = lambda: torch.zeros_like(audio_waveform).uniform_()
    combination = augment.EffectChain() \
        .pitch("-q", random_pitch_shift).rate(sr) \
        .additive_noise(noise_generator, snr=5) \
        .reverb(50, 50, random_room_size).channels(1) 
    y = combination.apply(audio_waveform, src_info={'rate': sr}, target_info={'rate': sr})
    return y



class NoiseAugmenter:
    
    def __init__(self):
        print("init")

    def transform_1(self, audio_path, output_path):
        signal, sr = torchaudio.load(audio_path)
        ## resample all the datasets to 16k Hz
        signal = torchaudio.functional.resample(signal, orig_freq=sr, new_freq=16000)

        '''
        ## Handle mismatched audio samples
        if (np.shape(signal)[1] !=48000) :
            print('Signals are not of the same length at :', audio_path)
            if (np.shape(signal)[1] > 48000):
                signal = torch.reshape(signal, (1, 48000))
        '''
        
        # Call Augmentation 1 : Random Pitch Shift + Reverberation
        aug_1 = _pitch_reverb(signal, 16000)

        torchaudio.save(output_path, aug_1, sample_rate=16000)


    def transform_2(self, audio_path, output_path):
        signal, sr = torchaudio.load(audio_path)
        ## resample all the datasets to 16k Hz
        signal = torchaudio.functional.resample(signal, orig_freq=sr, new_freq=16000) 

        """
        ## Handle mismatched audio samples
        if (np.shape(signal)[1] !=48000) :
            print('Signals are not of the same length at :', audio_path)
            if (np.shape(signal)[1] > 48000):
                signal = torch.reshape(signal, (1, 48000))
        """

        # Call Augmentation 2 : Random Pitch Shift + Noise + Reverberation 
        aug_2 = _pitch_add_reverb(signal, 16000)

        torchaudio.save(output_path, aug_2, sample_rate=16000)