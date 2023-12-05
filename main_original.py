from utils import read_tokenize_file
from components.data_augmentation.augment import AugmentationTool

sentences = read_tokenize_file("sample.txt")

output_dir = "cloning_test"
speaker_audio_path = "Information/Clips/WhatsApp Audio 2020-05-27 at 3.55.03 PM_0006.wav"

augmentation_tool = AugmentationTool(output_dir)

for idx, snt in enumerate(sentences):

    augmentation_tool.augment(speaker_audio_path=speaker_audio_path,
                              text=snt, output_name="sample_" + str(idx) +".wav")







