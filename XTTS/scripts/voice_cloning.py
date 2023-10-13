# -*- coding: utf-8 -*-
"""
@author: Daniel
"""

import os
from TTS.api import TTS

def generate_speech(file_path, language, gpu=True):
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v1", gpu=gpu)
    
    # Ask the user for the input text
    input_text = input("Introduce el texto que quieres convertir a voz: ")
    
    relative_path = '../scripts/speech_generated/speech_cloned.wav'
    absolute_path = os.path.abspath(relative_path)
    
    # generate speech by cloning a voice using default settings
    tts.tts_to_file(text=input_text,
                    file_path=file_path,
                    speaker_wav=absolute_path,
                    language=language,
    )
    

relative_path = '../scripts/audio/audio_to_clone.wav'
absolute_path = os.path.abspath(relative_path)

generate_speech(file_path="output.wav", language="en", gpu=True)