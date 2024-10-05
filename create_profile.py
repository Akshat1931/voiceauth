import json
from feature_extractor import extract_features

def save_profile(username, features):
    profile = {
        "username": username,
        "voiceprint": features.tolist()
    }
    with open(f'{username}_profile.json', 'w') as f:
        json.dump(profile, f)
    print(f"Profile for {username} saved.")

def create_user_profile(username, file_name):
    features = extract_features(file_name)
    save_profile(username, features)

if __name__ == "__main__":
    username = input("Enter username: ")
    file_name = input("Enter path to the voice recording (e.g., './voice_files/user_voice.wav'): ")
    create_user_profile(username, file_name)
