# -*- coding: utf-8 -*-
"""
@author: Daniel
"""

import os
import argparse
from TTS.api import TTS

def generate_speech(file_path, language, text, gpu=True):
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v1", gpu=gpu)
    
    # Ask the user for the input text
    #input_text = input("Introduce el texto que quieres convertir a voz: ")

    # Create the "speech_generated" directory if it doesn't exist
    output_directory = "speech_generated"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Set the output file path to "output.wav" within the "speech_generated" directory
    output_path = os.path.join(output_directory, "output.wav")

    # generate speech by cloning a voice using default settings
    tts.tts_to_file(text=text,
                    file_path=output_path,
                    speaker_wav=file_path,
                    language=language,
    )

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Generate speech from input text")
  parser.add_argument("--text", type=str, default="Este texto es para generar audio pasando un audio de entrada",
                      help="Text to convert to speech")

  args = parser.parse_args()

relative_path = '../scripts/audio/audio_to_clone.wav'
absolute_path = os.path.abspath(relative_path)

generate_speech(file_path=absolute_path, language="es", text=args.text, gpu=False)