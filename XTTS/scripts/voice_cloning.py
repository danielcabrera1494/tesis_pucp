# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 14:39:15 2023

@author: Daniel
"""

from TTS.api import TTS

def generate_speech(file_path, language, gpu=True):
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v1", gpu=gpu)
    
    # generate speech by cloning a voice using default settings
    tts.tts_to_file(text="It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                    file_path=file_path,
                    speaker_wav="/path/to/target/speaker.wav",
                    language=language,
    )