import os
from utils import read_tokenize_file
from components.data_augmentation.augment import AugmentationTool
from components.data_augmentation.noising.noise_text_augment import DisfluencyAugmenter


texts_path = "Information/Texts/"

sentences_per_file = {}
for fname in os.listdir(texts_path):
    sentences_per_file[fname.lower().replace(".txt", "").replace(" ", "_")] = read_tokenize_file(os.path.join(texts_path, fname))

output_dir = "disfluency_augmentation"
speaker_audio_folder = "Information/Clips"

disfluencyAugmenter = DisfluencyAugmenter(speakers_folder=speaker_audio_folder,
                                          output_dir=output_dir)


for key in sentences_per_file:
    sentences = sentences_per_file[key]
    for idx, sentence in enumerate(sentences):
        disfluencyAugmenter.generate(sentence,
                                     output_name=key + "_" + str(idx) + ".wav")
