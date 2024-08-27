import os
import numpy as np
import librosa
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from audiomentations import (
    Compose, AddBackgroundNoise, PitchShift, Gain, PolarityInversion
)
import matplotlib.pyplot as plt
import requests
import tarfile
import io
import soundfile as sf
from sklearn.model_selection import train_test_split
from scipy.signal import convolve

# Configuration class
class CONFIG:
    SR = 25600
    N_MEL = 128
    DURATION = 0.1
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    NUM_AUGMENTATIONS = 10
    NOISE_DIR = 'background_noises'
    RESULT_DIR = 'result'
    TARGET_SIZE = (128, 128)  # Target size for Mel spectrograms
    DATA_DIR = 'data'  # Root directory containing Caution, Fault, Normal folders

# Resample audio files to the target sample rate
def resample_audio_files(input_dir, output_dir, target_sr=25600):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.wav'):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_dir, os.path.relpath(input_path, input_dir))

                if not os.path.exists(os.path.dirname(output_path)):
                    os.makedirs(os.path.dirname(output_path))

                audio, sr = librosa.load(input_path, sr=None)
                if sr != target_sr:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                sf.write(output_path, audio, target_sr)
                print(f"Resampled {input_path} to {output_path}")

# Download and extract noise data
def download_and_extract_noise_data():
    url = "[datasource]"
    print("Downloading noise data...")
    response = requests.get(url)
    
    print("Extracting noise data...")
    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
        tar.extractall(path=CONFIG.NOISE_DIR)
    
    print("Noise data downloaded and extracted successfully.")

# Ensure noise directory exists and download data if needed
if not os.path.exists(CONFIG.NOISE_DIR) or not os.listdir(CONFIG.NOISE_DIR):
    os.makedirs(CONFIG.NOISE_DIR, exist_ok=True)
    download_and_extract_noise_data()
    input_directory = 'inyourdata'
    output_directory = 'inyourdata'
    resample_audio_files(input_directory, output_directory, target_sr=CONFIG.SR)

# Ensure result directory exists
os.makedirs(CONFIG.RESULT_DIR, exist_ok=True)

# Add reverb effect to audio
def add_reverb(signal, sr, reverb_amount=0.01, start_time=0.05):
    # Create an impulse response
    impulse_response = np.concatenate([np.zeros(int(sr * 0.01)), np.ones(int(sr * 0.1))])
    
    # Apply convolution to create the reverb signal
    reverb_signal = convolve(signal, impulse_response, mode='full')
    
    # Ensure the reverb signal has the same length as the original signal
    reverb_signal = reverb_signal[:len(signal)]
    
    # Calculate the start sample for the reverb
    start_sample = int(sr * start_time)
    
    # Pad the beginning of the reverb signal to start at start_sample
    padded_reverb_signal = np.concatenate([np.zeros(start_sample), reverb_signal])
    
    # Ensure the padded reverb signal has the same length as the original signal
    padded_reverb_signal = padded_reverb_signal[:len(signal)]
    
    # Mix the original signal with the reverb signal
    reverb_signal = signal + reverb_amount * padded_reverb_signal
    
    return reverb_signal


# Pad Mel spectrogram to the target size
def pad_to_target_size(mel_spec, target_size=(128, 128)):
    c, h, w = mel_spec.shape
    target_h, target_w = target_size

    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    mel_spec_padded = F.pad(mel_spec, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    
    return mel_spec_padded

class AudioDataset(Dataset):
    def __init__(self, data_dir, transform=None, split='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        self.file_paths, self.labels = self._load_file_paths_and_labels()
        self.file_paths, self.labels = self._split_dataset(self.file_paths, self.labels)

        # Ensure result_wav directory exists
        self.result_wav_dir = CONFIG.RESULT_DIR+"_wav"
        os.makedirs(self.result_wav_dir, exist_ok=True)

    def _load_file_paths_and_labels(self):
        file_paths = []
        labels = []

        label_map = {'Caution': 0, 'Fault': 1, 'Normal': 2}
        for label, idx in label_map.items():
            label_dir = os.path.join(self.data_dir, label)
            if not os.path.exists(label_dir):
                continue
            wav_files = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.wav')]
            if wav_files:
                file_paths.extend(wav_files)
                labels.extend([idx] * len(wav_files))

        return file_paths, labels

    def _split_dataset(self, file_paths, labels):
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            file_paths, labels, test_size=0.3, stratify=labels
        )
    
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=1/3, stratify=temp_labels
        )
    
        if self.split == 'train':
            return train_paths, train_labels
        elif self.split == 'val':
            return val_paths, val_labels
        elif self.split == 'test':
            return test_paths, test_labels

    def __len__(self):
        return len(self.file_paths) * CONFIG.NUM_AUGMENTATIONS

    def __getitem__(self, idx):
        file_idx = idx // CONFIG.NUM_AUGMENTATIONS
        file_path = self.file_paths[file_idx]
        label = self.labels[file_idx]
        
        audio, sr = librosa.load(file_path, sr=CONFIG.SR, duration=CONFIG.DURATION)
        
        # Apply reverb effect
        audio = add_reverb(audio, sr)
        audio = np.array(audio, dtype=np.float32)
        
        # Save the audio to the result_wav directory
        result_wav_path = os.path.join(self.result_wav_dir, f"{os.path.basename(file_path).replace('.wav', f'_{file_idx}.wav')}")
      
        if self.transform:
            audio = self.transform(samples=audio, sample_rate=CONFIG.SR)

        # Ensure consistent audio length
        target_length = int(CONFIG.SR * CONFIG.DURATION)
        if len(audio) != target_length:
            audio = librosa.util.fix_length(audio, size=target_length)
        sf.write(result_wav_path, audio, sr)
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=CONFIG.SR, 
            n_mels=CONFIG.N_MEL,
            n_fft=2048,
            hop_length=32
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        mel_spec_db = torch.tensor(mel_spec_db).unsqueeze(0).float()
        
        # Pad the Mel spectrogram to the target size
        mel_spec_db = pad_to_target_size(mel_spec_db, target_size=CONFIG.TARGET_SIZE)

        return mel_spec_db, label


# Define audio augmentations
augment = Compose([
    AddBackgroundNoise(sounds_path=CONFIG.NOISE_DIR, min_snr_in_db=20, max_snr_in_db=35, p=1)
])

# Create dataset and dataloader for each split
train_dataset = AudioDataset(CONFIG.DATA_DIR, transform=augment, split='train')
val_dataset = AudioDataset(CONFIG.DATA_DIR, transform=augment, split='val')
test_dataset = AudioDataset(CONFIG.DATA_DIR, transform=augment, split='test')

train_dataloader = DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True, num_workers=CONFIG.NUM_WORKERS)
val_dataloader = DataLoader(val_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False, num_workers=CONFIG.NUM_WORKERS)
test_dataloader = DataLoader(test_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False, num_workers=CONFIG.NUM_WORKERS)

# Save Mel spectrogram images
def save_spectrogram_images(batch, batch_idx):
    images, labels = batch
    batch_size = len(images)

    for i in range(batch_size):
        image = images[i].numpy()[0]
        label = labels[i].item()
        plt.figure(figsize=(10, 4))
        plt.imshow(image, aspect='auto', cmap='viridis')
        plt.title(f"Label: {label}")
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        file_name = f"label_{label}_{i}.png"
        file_path = os.path.join(CONFIG.RESULT_DIR, file_name)
        plt.savefig(file_path)
        plt.close()

# Example of saving a batch of spectrogram images
for batch_idx, batch in enumerate(train_dataloader):
    save_spectrogram_images(batch, batch_idx)
    break

print(f"Spectrogram shape: {batch[0][0].shape}")
print(f"Total number of mel spectrograms generated: {len(train_dataset)}")
