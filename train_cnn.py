"""CNN training on recording-level splits (refactored from the original Colab notebook)."""
import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAMPLE_RATE = 32000
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

def load_audio_segment(path, start_time, duration, sr=32000):
    y, _ = librosa.load(
        path,
        sr=sr,
        mono=True,
        offset=float(start_time),
        duration=float(duration)
    )

    target_length = int(sr * duration)

    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))

    if len(y) > target_length:
        y = y[:target_length]

    return y


def audio_to_mel_spectrogram(y, sr=32000, n_mels=128, n_fft=2048, hop_length=512):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def spec_augment(mel, max_time_mask=30, max_freq_mask=15):
    mel = mel.copy()

    n_mels, n_steps = mel.shape

    freq_mask = np.random.randint(0, max_freq_mask + 1)
    if freq_mask > 0:
        f0 = np.random.randint(0, max(1, n_mels - freq_mask))
        mel[f0:f0 + freq_mask, :] = mel.mean()

    time_mask = np.random.randint(0, max_time_mask + 1)
    if time_mask > 0:
        t0 = np.random.randint(0, max(1, n_steps - time_mask))
        mel[:, t0:t0 + time_mask] = mel.mean()

    return mel


class BirdAudioDataset(Dataset):
    def __init__(self, df, augment=False):
        self.df = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        y = load_audio_segment(
            path=row["local_path"],
            start_time=row["start_time"],
            duration=row["duration"],
            sr=SAMPLE_RATE
        )

        mel = audio_to_mel_spectrogram(
            y,
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )

        if self.augment:
            mel = spec_augment(mel)

        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        target = torch.tensor(row["target"], dtype=torch.long)

        return mel, target

class BirdCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    all_preds = []
    all_targets = []

    for x, y in tqdm(loader, desc="Training", leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

        preds = logits.argmax(dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(y.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro")

    return avg_loss, acc, f1


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating", leave=False):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)

            preds = logits.argmax(dim=1)

            all_preds.extend(preds.detach().cpu().numpy())
            all_targets.extend(y.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro")

    return avg_loss, acc, f1, all_targets, all_preds

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--segments-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--no-augment', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    frames = [pd.read_csv(args.segments_dir / f'{split}_segments.csv') for split in ('train','val','test')]
    labels = sorted(frames[0].label.unique())
    mapping = {label:i for i,label in enumerate(labels)}
    ids = [set(df.recording_id) for df in frames]
    if any(ids[i] & ids[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError('Recording IDs overlap across splits')
    for df in frames:
        if df.empty or set(df.label) != set(labels): raise ValueError('Empty split or missing classes')
        df['target'] = df.label.map(mapping)
    loaders = [DataLoader(BirdAudioDataset(df,augment=(i==0 and not args.no_augment)),batch_size=args.batch_size,shuffle=i==0,num_workers=0) for i,df in enumerate(frames)]
    model = BirdCNN(len(labels)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history=[]; best=-1; stale=0; ckpt=args.out/'best.pt'
    for epoch in range(1,args.epochs+1):
        tr=train_one_epoch(model,loaders[0],optimizer,criterion,device)
        va=evaluate(model,loaders[1],criterion,device)[:3]
        row=dict(zip(('train_loss','train_acc','train_f1','val_loss','val_acc','val_f1'),(*tr,*va)))
        history.append({'epoch':epoch,**row}); print(history[-1])
        if va[2]>best:
            best=va[2]; stale=0; torch.save(model.state_dict(),ckpt)
        else:
            stale+=1
            if stale>=3: break
    model.load_state_dict(torch.load(ckpt,map_location=device,weights_only=True))
    loss,acc,f1,y,pred=evaluate(model,loaders[2],criterion,device)
    pd.DataFrame(history).to_csv(args.out/'training_history.csv',index=False)
    report=classification_report(y,pred,labels=list(range(len(labels))),target_names=labels,output_dict=True,zero_division=0)
    pd.DataFrame(report).T.to_csv(args.out/'classification_report.csv')
    (args.out/'label_to_idx.json').write_text(json.dumps(mapping,indent=2))
    (args.out/'metrics.json').write_text(json.dumps({'test_loss':loss,'accuracy':acc,'macro_f1':f1,'test_segments':len(y),'best_val_f1':best},indent=2))
    cm=confusion_matrix(y,pred,labels=list(range(len(labels))))
    fig,ax=plt.subplots(figsize=(10,8)); im=ax.imshow(cm);fig.colorbar(im,ax=ax)
    ax.set_xticks(range(len(labels)),labels,rotation=90);ax.set_yticks(range(len(labels)),labels)
    ax.set_xlabel('Predicted');ax.set_ylabel('True');fig.tight_layout();fig.savefig(args.out/'confusion_matrix.png');plt.close(fig)
    print((args.out/'metrics.json').read_text())

if __name__ == '__main__': main()
