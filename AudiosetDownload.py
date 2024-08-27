from audioset_download import Downloader
import os
import shutil
from sklearn.model_selection import train_test_split
import librosa
import soundfile as sf
def download_and_split_audioset(root_path, labels, n_jobs=2, train_size=0.7, val_size=0.2, test_size=0.1):
    # Create directories for train, val, and test
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(root_path, split), exist_ok=True)

    # Download balanced_train data
    d_train = Downloader(root_path=root_path, labels=labels, n_jobs=n_jobs, download_type='balanced_train', copy_and_replicate=False)
    d_train.download(format='wav')

    # Download eval data
    d_eval = Downloader(root_path=root_path, labels=labels, n_jobs=n_jobs, download_type='eval', copy_and_replicate=False)
    d_eval.download(format='wav')
    # d_eval = Downloader(root_path=root_path, labels=labels, n_jobs=n_jobs, download_type='unbalanced_train', copy_and_replicate=False)
    # d_eval.download(format='wav')
   
    all_files = []
    for root, _, files in os.walk(root_path):
        for file in files:
            if file.endswith('.wav'):
                all_files.append(os.path.join(root, file))

    # Split files into train, val, and test sets
    train_files, test_files = train_test_split(all_files, test_size=test_size + val_size, random_state=42)
    val_files, test_files = train_test_split(test_files, test_size=test_size/(test_size + val_size), random_state=42)

    # Move files to their respective directories
    for files, split in zip([train_files, val_files, test_files], ['train', 'val', 'test']):
        for file in files:
            dest = os.path.join(root_path, split, os.path.basename(file))
            shutil.move(file, dest)

    print(f"Train set: {len(train_files)} files")
    print(f"Validation set: {len(val_files)} files")
    print(f"Test set: {len(test_files)} files")


def delete_small_wav_files(directory, size_threshold=100):
    # Function to delete empty directories
    def delete_empty_dirs(path):
        for root, dirs, files in os.walk(path, topdown=False):
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                if not os.listdir(dir_path):
                    print(f"Deleting empty directory {dir_path}")
                    os.rmdir(dir_path)

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                if os.path.getsize(file_path) < size_threshold:
                    print(f'Deleting {file_path} (Size: {os.path.getsize(file_path)} bytes)')
                    os.remove(file_path)

    # Delete empty directories
    delete_empty_dirs(directory)
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
# Usage
root_path = 'background_noises'
labels = ['Vibration', 'Mechanisms', 'Mechanical fan', 'Tools']

download_and_split_audioset(root_path, labels)

# # Delete small files in all subdirectories
# delete_small_wav_files(root_path)

# input_directory = 'background_noises'
# output_directory = 'background_noises'
# resample_audio_files(input_directory, output_directory)

# delete_small_wav_files(root_path)
