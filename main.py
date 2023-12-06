import os
from components.data_augmentation.augment import AugmentationTool

import random
#x1 = 3 | x2 = 4 | x3 = 5 | x4 = 6
random.seed(3)

# Base directory for original train data and directory for augmented files
base_dir = '/content/drive/MyDrive/Ulima/Data/train_data'
augmented_base_dir = '/content/drive/MyDrive/Ulima/Data/augment_x1_train_data'
stutter_categories = ['WordRep','Block','SoundRep','Prolongation']

# Ensure augmented directories exist
for category in stutter_categories:
    augmented_category_dir = os.path.join(augmented_base_dir, category)
    if not os.path.exists(augmented_category_dir):
        os.makedirs(augmented_category_dir)

# Process each stutter category
for category in stutter_categories:
    category_path = os.path.join(base_dir, category)
    print(f"Processing category: {category}")

    # Initialize AugmentationTool with specific output directory for the category
    augmentation_tool = AugmentationTool(os.path.join(augmented_base_dir, category))

    # Loop through each audio file in the category
    for idx, filename in enumerate(os.listdir(category_path)):
        if filename.endswith('.wav'):
            speaker_audio_path = os.path.join(category_path, filename)
            output_name = f"{category}_sample_{idx}.wav"
            augmentation_tool.augment(speaker_audio_path=speaker_audio_path, output_name=output_name)