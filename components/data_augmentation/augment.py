import os
from .cloning.voice_cloning import VoiceCloner
from .noising.noise_augment import NoiseAugmenter


class AugmentationTool:

    def __init__(self, output_dir, gpu=False):
        self._lang = "es"
        self._output_dir = output_dir
    
        # Create the "speech_generated" directory if it doesn't exist
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)

        self._voice_cloner = VoiceCloner(gpu=gpu)

        self._noise_augmenter = NoiseAugmenter()


    def augment(self, speaker_audio_path, text, output_name):

        # generate "cleaned" speech
        self._voice_cloner.generate_speech(speaker_audio_path, text,
                                           os.path.join(self._output_dir, output_name),
                                           language=self._lang)

        # introduce dysfluencies
        self._noise_augmenter.transform_1(os.path.join(self._output_dir, output_name),
                                          os.path.join(self._output_dir, "sample_aug1.wav"))
        
        self._noise_augmenter.transform_2(os.path.join(self._output_dir, output_name),
                                          os.path.join(self._output_dir, "sample_aug2.wav"))
