import os
import torch
import augment
import torchaudio
import numpy as np
import random
from pydub import AudioSegment
from syltippy import syllabize
from nltk.tokenize import word_tokenize
from sacremoses import MosesDetokenizer
from components.data_augmentation.cloning.voice_cloning import VoiceCloner

PROLONGATION_CONSONANTS = ["l", "m", "n", "s"]

def add_prolongation(tokens):

    idxs = random.sample([i for i in range(len(tokens))], 2)

    for idx in idxs:
        if len(tokens[idx]) > 1:
            for consonant in PROLONGATION_CONSONANTS:
                try:
                    pos = tokens[idx].index(consonant)
                    content = tokens[idx]
                    tokens[idx] = content[:pos] + consonant * random.choice(range(6,14)) + content[pos:]
                    break
                except:
                    pass
    return tokens

def add_word_repetition(tokens):
    idxs = random.sample([i for i in range(len(tokens))], 2)

    new_tokens = []
    for idx, token in enumerate(tokens):
        if idx in idxs and len(token) > 1:
            new_tokens += [token] * 2
        else:
            new_tokens.append(token)
    return new_tokens

def add_sound_repetition(tokens):
    idxs = random.sample([i for i in range(len(tokens))], 2)

    new_tokens = []
    for idx, token in enumerate(tokens):
        syllables, _ = syllabize(token)
        if idx in idxs and len(syllables) > 1:
            #selecting sylable that will be modified
            syl_idx = random.choice(range(len(syllables)))
            if syl_idx == len(syllables) - 1:
                continue

            rep_syl_idx = random.choice(range(2,4))
            syllables[syl_idx] = "-".join([syllables[syl_idx]] * rep_syl_idx)
            new_tokens.append("".join(syllables))
        else:
            new_tokens.append(token)
    return new_tokens

def add_interjection(tokens):
    print("here we define a prolongation")

def add_blocking(input_path, num_blocking):

    audio_in_file = input_path
    audio = AudioSegment.from_wav(audio_in_file)

    for _ in range(num_blocking):
        # create silence segment
        silence_sec = random.uniform(1,2.01)
        silence_segment = AudioSegment.silent(duration=silence_sec * 1000)
        
        stop_sec = random.uniform(0,audio.duration_seconds)
        audio = audio[:stop_sec*1000] + silence_segment + audio[stop_sec*1000:]
    
    return audio



class DisfluencyAugmenter:
    
    def __init__(self, speakers_folder, 
                 output_dir,
                 random_state=42, 
                 gpu=False):
        self._lang = "es"

        random.seed(random_state)
        
        self._speakers_folder = speakers_folder
        self._speaker_paths = os.listdir(speakers_folder)
        self._output_dir = output_dir
    
        # Create the "speech_generated" directory if it doesn't exist
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)

        self._voice_cloner = VoiceCloner(gpu=gpu)

        self._detokenizer = MosesDetokenizer(lang='es')        

    def generate(self, text, output_name):
        
        chosen_speaker_idx = random.choice(range(len(self._speaker_paths)))
        chosen_speaker_path = os.path.join(self._speakers_folder,
                                           self._speaker_paths[chosen_speaker_idx])

        tokens = word_tokenize(text)
        tokens = add_prolongation(tokens)
        tokens = add_word_repetition(tokens)
        tokens = add_sound_repetition(tokens)

        disfluency_text = self._detokenizer.detokenize(tokens)
        
        self._voice_cloner.generate_speech(chosen_speaker_path, 
                                           disfluency_text,
                                           os.path.join(self._output_dir, 
                                                        output_name),
                                           language=self._lang)

        num_blocking = random.choice(range(3))
        if num_blocking > 0:
            audio = add_blocking(os.path.join(self._output_dir, output_name),
                                 num_blocking=num_blocking)
            
            #Either save modified audio
            audio.export(os.path.join(self._output_dir, output_name), 
                         format="wav")