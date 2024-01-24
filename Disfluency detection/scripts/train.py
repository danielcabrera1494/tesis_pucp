import numpy as np
from dataset import load_dataset_from_path
from utils import set_seed, __get_device__
from utils import Wav2VecRepresentation

from sklearn.model_selection import train_test_split

from utils import AudioDataset, get_dataloader
from model import StutterNet

set_seed(42)
wav2vec_rep = Wav2VecRepresentation(__get_device__())

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
print('Number of samples to validate = ', n_samples_valid)
print('Number of samples to test = ', n_samples_test)

batch_size = 64

train_dataset = AudioDataset(x_train,y_train, n_samples_train)
dev_dataset = AudioDataset(x_dev, y_dev, n_samples_dev)
test_dataset = AudioDataset(x_test,y_test,n_samples_test)

train_loader = get_dataloader(train_dataset, batch_size, shuffle=True)
dev_loader = get_dataloader(dev_dataset, batch_size)



