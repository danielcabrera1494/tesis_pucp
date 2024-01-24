import os
import glob
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader


def get_processed_data_from_folder(path, label, audio_data_conversor):
    x = []
    y = []

    discarded = 0
    included = 0

    for filename in glob.glob(os.path.join(path, '*.wav')):
        np_data = audio_data_conversor.get_representation(filename)
        #print(np_data.shape)
        # fluent_np --> (1, 149, 1024)
        if ((np.shape(np_data)[0] != 1) |
                (np.shape(np_data)[1] != 149) | 
                (np.shape(np_data)[2] != 1024)):
            discarded += 1
        else:
            included += 1
            x.append(np_data)
            y.append(label)

    print(f'Number of discarded instances: {discarded}')
    print(f'Number of included instances: {included}')
    return x, y

def load_dataset_from_path(stutter_path, fluent_path, wav2vec_rep, balance=False):
    random.seed(42)
    stutter_x, stutter_y = get_processed_data_from_folder(stutter_path, 1, wav2vec_rep)
    fluent_x, fluent_y = get_processed_data_from_folder(fluent_path, 0, wav2vec_rep)

    if balance:
        random.shuffle(fluent_x)
        fluent_x = fluent_x[:len(stutter_x)]
        fluent_y = fluent_y[:len(stutter_x)]

    return stutter_x + fluent_x, stutter_y + fluent_y


class AudioDataset(Dataset) :
    def __init__(self,x,y, n_samples) :
        # data loading
        self.x = x
        self.y = y 
        self.n_samples = n_samples
        
        
    def __getitem__(self,index) :
        return self.x[index], self.y[index]

    def __len__(self) :    
        return self.n_samples

def get_dataloader(dataset, batch_size, shuffle=False):
    return DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=1)
