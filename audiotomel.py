import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def pad_to_target_size(mel_spec, target_size=(128, 128)):
    mel_spec = torch.tensor(mel_spec)  # 넘파이 배열을 파이토치 텐서로 변환
    h, w = mel_spec.shape
    target_h, target_w = target_size

    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    mel_spec_padded = F.pad(mel_spec, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    
    return mel_spec_padded.numpy()  # 파이토치 텐서를 다시 넘파이 배열로 변환

def wav_to_log_mel_spectrogram_image(wav_path, output_image_path):
    # 오디오 파일 로드
    y, sr = librosa.load(wav_path, sr=25600)
    
    # Mel 스펙트로그램 계산
    S = librosa.feature.melspectrogram(
            y=y, 
            sr=25600, 
            n_mels=128,
            n_fft=2048,
            hop_length=32
        )
    # Mel 스펙트로그램을 로그 스케일로 변환
    S_db = librosa.power_to_db(S, ref=np.max)
    #S_db = pad_to_target_size(S_db)

    # Mel 스펙트로그램을 이미지로 저장
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-Mel Spectrogram')
    plt.tight_layout()
    plt.savefig(output_image_path)
    plt.close()

def convert_all_wavs_to_images(wav_dir):
    for filename in os.listdir(wav_dir):
        if filename.endswith('.wav'):
            wav_path = os.path.join(wav_dir, filename)
            output_image_path = os.path.join(wav_dir, f"{filename[:-4]}.png")
            wav_to_log_mel_spectrogram_image(wav_path, output_image_path)
            print(f"Saved spectrogram image for {filename} as {output_image_path}")

# pictureimage 디렉토리의 모든 wav 파일들을 로그 멜 스펙트로그램 이미지로 변환
wav_dir = 'pictureimage'
convert_all_wavs_to_images(wav_dir)
