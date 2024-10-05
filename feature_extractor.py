import librosa
import numpy as np

def extract_features(file_name):
    y, sr = librosa.load(file_name)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    chroma_mean = np.mean(chroma.T, axis=0)
    mel_mean = np.mean(mel.T, axis=0)
    return np.hstack((mfcc_mean, chroma_mean, mel_mean))  # Combine all features

if __name__ == "__main__":
    file_name = 'user_voice.wav'
    features = extract_features(file_name)
    print(f"Extracted Features: {features}")
