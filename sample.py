import os
import numpy as np
import librosa
import torch
import soundfile as sf
from scipy.signal import convolve
from audiomentations import Compose, AddBackgroundNoise

class CONFIG:
    SR = 25600
    N_MEL = 128
    DURATION = 0.1
    NUM_AUGMENTATIONS = 10
    NOISE_DIR = {'train': 'background_noises/train', 'val': 'background_noises/val', 'test': 'background_noises/test'}
    RESULT_DIR = '.'
    TARGET_SIZE = (128, 128)
    DATA_DIR = 'data'

def add_reverb(signal, sr, reverb_amount=0.01, start_time=0.05):
    impulse_response = np.concatenate([np.zeros(int(sr * 0.01)), np.ones(int(sr * 0.1))])
    reverb_signal = convolve(signal, impulse_response, mode='full')
    reverb_signal = reverb_signal[:len(signal)]
    start_sample = int(sr * start_time)
    padded_reverb_signal = np.concatenate([np.zeros(start_sample), reverb_signal])
    padded_reverb_signal = padded_reverb_signal[:len(signal)]
    reverb_signal = signal + reverb_amount * padded_reverb_signal
    return reverb_signal

class AudioProcessor:
    def __init__(self, data_dir, transform, split='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        self.label_map = {'Caution': 0, 'Fault': 1, 'Normal': 2}
        self.inverse_label_map = {v: k for k, v in self.label_map.items()}
        self.file_paths, self.labels = self._load_file_paths_and_labels()
        self.result_wav_dir = os.path.join(CONFIG.RESULT_DIR, split)
        os.makedirs(self.result_wav_dir, exist_ok=True)

    def _load_file_paths_and_labels(self):
        file_paths = []
        labels = []
        for label, idx in self.label_map.items():
            label_dir = os.path.join(self.data_dir, label)
            if not os.path.exists(label_dir):
                print(f"Directory {label_dir} does not exist.")
                continue
            wav_files = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.wav')]
            if wav_files:
                file_paths.extend(wav_files)
                labels.extend([idx] * len(wav_files))
            else:
                print(f"No .wav files found in {label_dir}.")
        return file_paths, labels

    def process_files(self):
        for file_path, label in zip(self.file_paths, self.labels):
            audio, sr = librosa.load(file_path, sr=CONFIG.SR, duration=CONFIG.DURATION)
            for aug_idx in range(CONFIG.NUM_AUGMENTATIONS):
                augmented_audio = self.apply_augmentation(audio, sr, aug_idx)
                self.save_audio(augmented_audio, file_path, label, aug_idx)

    def apply_augmentation(self, audio, sr, aug_idx):
        audio = add_reverb(audio, sr)
        audio = np.array(audio, dtype=np.float32)
        target_length = int(CONFIG.SR * CONFIG.DURATION)
        if len(audio) != target_length:
            audio = librosa.util.fix_length(audio, size=target_length)
        if self.transform:
            audio = self.transform(samples=audio, sample_rate=CONFIG.SR)
        return audio

    def save_audio(self, audio, file_path, label, aug_idx):
        base_filename = os.path.basename(file_path)
        label_name = self.inverse_label_map[label]
        result_wav_filename = f"{label_name}_{base_filename[:-4]}_{aug_idx}.wav"
        result_wav_path = os.path.join(self.result_wav_dir, result_wav_filename)
        sf.write(result_wav_path, audio, CONFIG.SR)
        print(f"Saving file: {result_wav_path}")

# Define audio augmentations
augmentations = {
   
    'train': Compose([AddBackgroundNoise(sounds_path=CONFIG.NOISE_DIR['train'], min_snr_in_db=20, max_snr_in_db=35, p=1)]),
    'val': Compose([AddBackgroundNoise(sounds_path=CONFIG.NOISE_DIR['val'], min_snr_in_db=20, max_snr_in_db=35, p=1)]),
    'test': Compose([AddBackgroundNoise(sounds_path=CONFIG.NOISE_DIR['test'], min_snr_in_db=20, max_snr_in_db=35, p=1)])
}

# Process and save augmented audio for validation and test sets

train_processor = AudioProcessor('train1', transform=augmentations['train'], split='train')
val_processor = AudioProcessor('val1', transform=augmentations['val'], split='val')
test_processor = AudioProcessor('test1', transform=augmentations['test'], split='test')
train_processor.process_files()
val_processor.process_files()
test_processor.process_files()
