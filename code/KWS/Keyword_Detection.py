import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import torch
import torchaudio
import torch.nn as nn

# ==============================================================
# 1. Labels (same as training)
# ==============================================================
labels = [
    'backward','bed','bird','cat','dog','down','eight','five','follow','forward',
    'four','go','happy','house','learn','left','marvin','nine','no','off','on',
    'one','right','seven','sheila','six','stop','three','tree','two','up',
    'visual','wow','yes','zero'
]

# ==============================================================
# 2. Model definition
# ==============================================================
class CNN_Transformer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(64,128,3,padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128,128,3,padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2)
        )

        enc = nn.TransformerEncoderLayer(128,4,256,0.2,batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=4)

        self.fc = nn.Sequential(
            nn.Linear(128,256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self,x):
        x = self.cnn(x)
        x = x.transpose(1,2)
        x = self.transformer(x)
        x = x.mean(1)
        return self.fc(x)

# ==============================================================
# 3. Load pretrained model
# ==============================================================
model = CNN_Transformer(len(labels))
model.load_state_dict(torch.load(
    r"C:\Users\karula qiu\OneDrive - The University of Sydney (Students)\桌面\ELEC5305-Project\project\kws_cnn_transformer.pth",
    map_location="cpu"
))
model.eval()

# ==============================================================
# 4. Feature
# ==============================================================
mel = torchaudio.transforms.MelSpectrogram(16000, n_mels=64)
amp = torchaudio.transforms.AmplitudeToDB()

def extract_feature(w):
    # w: [N] 或 [1, N]
    if w.dim() == 1:
        w = w.unsqueeze(0)          # [1, N]

    x = mel(w)                      # [1, 64, T]
    x = amp(x)
    x = (x - x.mean())/(x.std()+1e-9)
    x = x.squeeze(0)                # -> [64, T]  和训练时一致
    return x


# ==============================================================
# 5. Record audio by pressing ENTER
# ==============================================================
def record_audio():
    SAMPLE_RATE = 16000
    print("\nPress ENTER to start recording…")
    input()
    print("🎤 Recording... Press ENTER again to stop.")

    audio = []
    sd.default.samplerate = SAMPLE_RATE
    sd.default.channels = 1

    recording = True

    def callback(indata, frames, time, status):
        if recording:
            audio.append(indata.copy())

    stream = sd.InputStream(callback=callback)
    stream.start()

    input()  # wait for second ENTER
    recording = False
    stream.stop()

    audio = np.concatenate(audio, axis=0)
    print(f"📁 Recorded {audio.shape[0]/SAMPLE_RATE:.2f} seconds.")
    return SAMPLE_RATE, audio.squeeze()

# ==============================================================
# 6. Sliding window keyword detection
# ==============================================================
def detect_keywords(wav, sr):
    if sr != 16000:
        wav = torchaudio.functional.resample(torch.tensor(wav), sr, 16000)
        sr = 16000
    else:
        wav = torch.tensor(wav)

    wav = wav.unsqueeze(0)  # [1, N]

    window = 16000
    hop = 4000
    detections = []

    for s in range(0, wav.shape[1] - window, hop):

        seg = wav[:, s:s+window]        # [1, window]
        feat = extract_feature(seg)     # 🔹 现在是 [64, T]

        # ✅ 这里加 batch 维度，变成 [1, 64, T]
        feat = feat.unsqueeze(0)

        with torch.no_grad():
            out = model(feat)
            prob = torch.softmax(out,1)
            conf, idx = prob.max(1)

        if conf.item() > 0.85:
            time_sec = s / sr
            detections.append((time_sec, labels[idx.item()], conf.item()))

    return detections



# ==============================================================
# 7. MAIN
# ==============================================================
if __name__ == "__main__":
    sr, audio = record_audio()

    # save the recorded file (optional)
    save_path = "recorded.wav"
    write(save_path, sr, audio)
    print(f"\nSaved recording to {save_path}")

    detections = detect_keywords(audio, sr)

    if detections:
        print("\n🔍 Keyword detections:")
        for t, w, c in detections:
            print(f"[{t:.2f}s] {w} ({c:.2f})")
    else:
        print("\n❌ No keywords detected.")

