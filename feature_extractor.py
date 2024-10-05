import librosa
import numpy as np

def extract_features(file_name):
    y, sr = librosa.load(file_name)
    y = librosa.util.normalize(y)  # Normalize the audio signal
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  # Keep it as 40
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)

    # Get mean and std dev of MFCC, Chroma, and Mel features
    mfcc_mean = np.mean(mfcc.T, axis=0)
    mfcc_std = np.std(mfcc.T, axis=0)
    chroma_mean = np.mean(chroma.T, axis=0)
    chroma_std = np.std(chroma.T, axis=0)
    mel_mean = np.mean(mel.T, axis=0)
    mel_std = np.std(mel.T, axis=0)

    # Combine mean and std dev features
    return np.hstack((mfcc_mean, mfcc_std, chroma_mean, chroma_std, mel_mean, mel_std))

if __name__ == "__main__":
    file_name = 'user_voice.wav'  # Example file to extract features from
    features = extract_features(file_name)
    print(f"Extracted Features: {features}")
