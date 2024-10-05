import sounddevice as sd
from scipy.io.wavfile import write

def record_voice(file_name, duration=5, fs=44100):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    write(file_name, fs, recording)
    print(f"Recording saved to {file_name}")

if __name__ == "__main__":
    file_name = input("Enter file name to save recording (e.g., './voice_files/new_voice.wav'): ")
    duration = int(input("Enter recording duration in seconds (default is 5): ") or 5)
    record_voice(file_name, duration)
