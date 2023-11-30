import os
from .cloning.voice_cloning import generate_speech


class AugmentationTool:

    def __init__(self, output_dir):
        self._lang = "es"
        self._output_dir = output_dir
    
        # Create the "speech_generated" directory if it doesn't exist
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)


    def augment(self, speaker_audio_path, text, output_name):

        # generate "cleaned" speech
        generate_speech(speaker_audio_path, text, 
                        os.path.join(self._output_dir, output_name),
                        language=self._lang)

        # introduce dysfluencies