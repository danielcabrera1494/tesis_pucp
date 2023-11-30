# -*- coding: utf-8 -*-
"""
@author: Daniel
"""

import os
import argparse
from TTS.api import TTS

#TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v1"
TTS_MODEL = "xtts"

def generate_speech(speaker_audio_path, text, output_path, language, gpu=False):
    tts = TTS(TTS_MODEL, gpu=gpu)
    
    # Ask the user for the input text
    #input_text = input("Introduce el texto que quieres convertir a voz: ")

    # generate speech by cloning a voice using default settings
    tts.tts_to_file(text=text,
                    file_path=output_path,
                    speaker_wav=speaker_audio_path,
                    language=language,
    )

relative_path = 'test_speaker/audio_to_clone.wav'
absolute_path = os.path.abspath(relative_path)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Generate speech from input text")
  parser.add_argument("--text", type=str, default="Este texto es para generar audio pasando un audio de entrada",
                      help="Text to convert to speech")
  parser.add_argument("--audio_file", type=str, default=absolute_path,
                      help="Audio file used for cloning")

  args = parser.parse_args()

  generate_speech(file_path=args.audio_file, text=args.text, 
                  output_path="output_sample.wav", 
                  language="es", gpu=False)
