import os
import random
from .noising.noise_audio_augment import NoiseAugmenter

class AugmentationTool:

    def __init__(self, output_dir):
        self._output_dir = output_dir
    
        # Create the output directory if it doesn't exist
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)

        self._noise_augmenter = NoiseAugmenter()

    def augment(self, speaker_audio_path, output_name):
        aleatorio = random.random()
        # Randomly choose a noise transformation
        if aleatorio >= 0.5:
            # Use transform_1
            self._noise_augmenter.transform_1(
                speaker_audio_path,
                os.path.join(self._output_dir, "aug1_" + output_name)
            )
        else:
            # Use transform_2
            self._noise_augmenter.transform_2(
                speaker_audio_path,
                os.path.join(self._output_dir, "aug2_" + output_name)
            )
