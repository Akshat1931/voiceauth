import json
import numpy as np
from voice_recorder import record_voice
from feature_extractor import extract_features
from sklearn.metrics.pairwise import cosine_similarity

def load_profile(username):
    try:
        with open(f'{username}_profile.json', 'r') as f:
            profile = json.load(f)
        return np.array(profile['voiceprint'])
    except FileNotFoundError:
        print(f"Profile for {username} not found.")
        return None

def authenticate_user(username):
    # Load the saved profile for the username
    profile_voiceprint = load_profile(username)
    if profile_voiceprint is None:
        return False
    
    # Record a new voice sample for verification
    file_name = 'login_voice.wav'
    print("Please speak to authenticate...")
    record_voice(file_name, duration=5)
    
    # Extract features from the recorded voice
    input_voiceprint = extract_features(file_name)
    
    # Compare the input voiceprint with the stored profile using cosine similarity
    similarity = cosine_similarity([profile_voiceprint], [input_voiceprint])
    print(f"Similarity Score: {similarity[0][0]}")
    
    # Authentication based on similarity score threshold
    if similarity[0][0] > 0.87:
        print("Authentication successful!")
        return True
    else:
        print("Authentication failed!")
        return False

if __name__ == "__main__":
    username = input("Enter username to authenticate: ")
    authenticate_user(username)
