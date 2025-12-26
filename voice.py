import os
import time
import torch
import torchaudio
import sounddevice as sd
import soundfile as sf  # Backup loader
from scipy.io.wavfile import write

# --- MONKEY PATCH: Fix torchaudio version issues ---
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = getattr(torchaudio, 'list_backends', lambda: [])

from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy

class VoiceAuthSystem:
    def __init__(self):
        print("\n--- Loading AI Voice Engine (ECAPA-TDNN) ---")
        # LocalStrategy.COPY avoids the Symlink/Permission error on Windows
        self.model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            savedir="pretrained_models",
            local_strategy=LocalStrategy.COPY
        )
        self.sample_rate = 16000 
        self.db_path = "voice_db"
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)

    def get_embedding(self, audio_path):
        """Loads audio safely and extracts the voiceprint."""
        try:
            # Try loading with torchaudio first
            signal, fs = torchaudio.load(audio_path)
        except Exception:
            # Backup: Load with soundfile if torchaudio backend fails
            print("Note: Using backup audio loader...")
            data, fs = sf.read(audio_path)
            signal = torch.tensor(data).float().unsqueeze(0)

        # Ensure correct sample rate for the AI
        if fs != self.sample_rate:
            resampler = torchaudio.transforms.Resample(fs, self.sample_rate)
            signal = resampler(signal)
            
        with torch.no_grad():
            embeddings = self.model.encode_batch(signal)
        return embeddings

    def record_audio(self, filename, duration=5):
        print(f"\n[RECORDING] Speak now for {duration} seconds...")
        recording = sd.rec(int(duration * self.sample_rate), 
                          samplerate=self.sample_rate, channels=1)
        for i in range(duration, 0, -1):
            print(f"{i}...", end=" ", flush=True)
            time.sleep(1)
        sd.wait()
        write(filename, self.sample_rate, recording)
        print("\n[DONE] Audio captured.")

    def enroll(self, name):
        path = "temp_enroll.wav"
        self.record_audio(path)
        embedding = self.get_embedding(path)
        torch.save(embedding, f"{self.db_path}/{name}.pt")
        print(f"--- Voiceprint for '{name}' saved! ---")

    def verify(self, name):
        user_file = f"{self.db_path}/{name}.pt"
        if not os.path.exists(user_file):
            print("Error: User not found.")
            return
        
        # 1. Capture new audio
        path = "temp_verify.wav"
        self.record_audio(path)
        
        # 2. Get embeddings (The current voice and the stored one)
        stored_emb = torch.load(user_file)
        new_emb = self.get_embedding(path)
        
        # 3. MANUAL MATH: Calculate Cosine Similarity
        # We flatten the vectors and use the formula: (A . B) / (||A|| * ||B||)
        cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        score = cos(stored_emb.flatten(), new_emb.flatten()).item()
        
        if os.path.exists(path): os.remove(path)

        print(f"\nSimilarity: {score:.4f} (Threshold: 0.25)")
        
        if score > 0.25:
            print(">>> [SUCCESS] Voice Match! Access Granted.")
        else:
            print(">>> [FAILED] Unauthorized! Voice does not match.")
            
if __name__ == "__main__":
    auth = VoiceAuthSystem()
    while True:
        print("\n1. Register Voice\n2. Login\n3. Exit")
        choice = input("Choice: ")
        if choice == '1': auth.enroll(input("Enter Name: "))
        elif choice == '2': auth.verify(input("Enter Name: "))
        elif choice == '3': break