# Keyword Spotting with CNN + Transformer Encoder
import torch, torchaudio, numpy as np, pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt, seaborn as sns

# ===================== Device ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.set_num_threads(4)
torchaudio.set_audio_backend("soundfile")

# ===================== 1. Load dataset ======================
root = r"C:\Users\karula qiu\PycharmProjects\JupyterProject\data"
train_set = torchaudio.datasets.SPEECHCOMMANDS(root=root, download=True, subset="training")
test_set  = torchaudio.datasets.SPEECHCOMMANDS(root=root, download=True, subset="testing")

labels = sorted(list(set(dat[2] for dat in train_set)))
label2id = {lb:i for i,lb in enumerate(labels)}
id2label = {i:lb for lb,i in label2id.items()}
print(f"Classes={len(labels)}, Train={len(train_set)}, Test={len(test_set)}")

# ===================== 2. Feature extraction ======================
mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=64).to(device)
amp_to_db = torchaudio.transforms.AmplitudeToDB().to(device)
time_mask = torchaudio.transforms.TimeMasking(20).to(device)
freq_mask = torchaudio.transforms.FrequencyMasking(8).to(device)

def extract_feature(waveform, augment=False):
    waveform = waveform.to(device)
    x = mel(waveform)
    x = amp_to_db(x)

    if augment:      # ✔ 只在训练中使用
        x = time_mask(x)
        x = freq_mask(x)

    x = (x - x.mean()) / (x.std() + 1e-9)
    return x.squeeze(0).cpu()


# ===================== Dataset with padding ======================
class KWS_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, max_len=160, augment=False):
        self.dataset = dataset
        self.max_len = max_len
        self.augment = augment

    def __getitem__(self, idx):
        waveform, sr, label, *_ = self.dataset[idx]
        feat = extract_feature(waveform, augment=self.augment)

        T = feat.shape[1]
        if T < self.max_len:
            pad = torch.zeros((64, self.max_len - T))
            feat = torch.cat([feat, pad], dim=1)
        else:
            feat = feat[:, :self.max_len]

        return feat, label2id[label]

    def __len__(self):
        return len(self.dataset)


train_loader = DataLoader(KWS_Dataset(train_set, augment=True), batch_size=64, shuffle=True)
test_loader  = DataLoader(KWS_Dataset(test_set, augment=False), batch_size=64, shuffle=False)


# ===================== 3. Model ======================
class CNN_Transformer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(64,128,3,padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128,128,3,padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256, dropout=0.2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.fc = nn.Sequential(
            nn.Linear(128,256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256,num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.transpose(1,2)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)

# ===================== 4. Training ======================
model = CNN_Transformer(num_classes=len(labels)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

for epoch in range(20):
    total_loss = 0
    model.train()
    for feats, y in train_loader:
        feats = feats.to(device)
        y = y.to(device)

        out = model(feats)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()   # ✅ 每个 epoch 调整一次学习率

    print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}")


# ===================== 5. Evaluation ======================
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for feats, y in test_loader:
        feats = feats.to(device)
        preds = model(feats).argmax(1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(y.numpy())

# save model
torch.save(model.state_dict(), "kws_cnn_transformer.pth")
print("\nSaved model to kws_cnn_transformer.pth")

# accuracy
acc = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"\nAccuracy = {acc * 100:.2f}%")

# classification table
report = classification_report(all_labels, all_preds, target_names=labels)
print(report)


# 7️⃣ 混淆矩阵可视化 --------------------------------------------------
cm = pd.crosstab(pd.Series(all_labels, name="True"), pd.Series(all_preds, name="Pred"))
plt.figure(figsize=(10,8))
sns.heatmap(cm, cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix (CNN+Transformer)")
plt.show()
