import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
from timm import create_model
from timm.optim import AdamP
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, update_bn



# =============================================
# 1. 설정 & seed 고정
# =============================================
CFG = {
    'MODEL_NAMES' : [
        'coatnet_2_rw_224',
     ],
    'IMG_SIZES':   [224],
    'BATCH_SIZES': [16],  # 예시; VRAM에 맞게 조정
    'EPOCHS_PHASE':[60],
    'SEED' : 42,
    'FOLDS' : 3,
    'LEARNING_RATE': 1e-3
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG['SEED'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# =============================================
# 2. Dataset 정의
# =============================================
class CustomImageDataset(Dataset):
    def __init__(self, samples, transform=None, is_test=False):
        self.samples = samples
        self.transform = transform
        self.is_test = is_test
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, *lbl = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        if self.is_test:
            return img
        return img, lbl[0]

# =============================================
# 3. MixUp & CutMix 함수
# =============================================
def mixup_cutmix(x, y, alpha=1.0):
    if random.random() < 0.5:
        lam = np.random.beta(alpha, alpha)
        idx = torch.randperm(x.size(0)).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[idx]
        return mixed_x, y, y[idx], lam
    lam = np.random.beta(alpha, alpha)
    B, C, H, W = x.size()
    idx = torch.randperm(B).to(x.device)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bw, bh = int(W * np.sqrt(1 - lam)), int(H * np.sqrt(1 - lam))
    x1, y1 = np.clip(cx - bw//2, 0, W), np.clip(cy - bh//2, 0, H)
    x2, y2 = np.clip(cx + bw//2, 0, W), np.clip(cy + bh//2, 0, H)
    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam_eff = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    return x, y, y[idx], lam_eff

# =============================================
# 4. 데이터 & CV Split 준비
# =============================================
train_root = '/home/kim/Desktop/hecto/open/train'
test_root  = '/home/kim/Desktop/hecto/open/test'
classes    = sorted(os.listdir(train_root))
print(len(classes))
samples = [(os.path.join(train_root, cls, f), classes.index(cls))
           for cls in classes for f in os.listdir(os.path.join(train_root, cls))]
targets = [s[1] for s in samples]
skf = StratifiedKFold(n_splits=CFG['FOLDS'], shuffle=True, random_state=CFG['SEED'])

# =============================================
# 5. 모델별 학습 함수
# =============================================
def train_model(model_idx, fold, train_idx, val_idx):
    name        = CFG['MODEL_NAMES'][model_idx]
    size        = CFG['IMG_SIZES'][model_idx]
    batch_size  = CFG['BATCH_SIZES'][model_idx]
    epochs      = CFG['EPOCHS_PHASE'][model_idx]
    freeze_epochs = 5   # 헤드만 학습할 에폭 수

    # 1) DataLoader 준비
    transform_train = T.Compose([
        T.Resize((size, size)),
        T.RandAugment(),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        T.RandomErasing(p=0.2),
    ])
    transform_val = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    trn = DataLoader(
        CustomImageDataset([samples[i] for i in train_idx], transform_train),
        batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True
    )
    val = DataLoader(
        CustomImageDataset([samples[i] for i in val_idx], transform_val),
        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # 2) 모델 생성 & 파라미터 분리
    use_pretrained = True
    model = create_model(name, pretrained=use_pretrained, num_classes=len(classes)).to(device)
    head = model.get_classifier()

    # head 파라미터
    head_params = list(head.parameters())
    head_ids    = {id(p) for p in head_params}
    # backbone 파라미터
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]

    # 처음에는 backbone freeze
    for p in backbone_params:
        p.requires_grad = False
    for p in head_params:
        p.requires_grad = True

    # 3) Optimizer & OneCycleLR
    optimizer = AdamP([
        {'params': head_params,      'lr': 1e-3},
        {'params': backbone_params,  'lr': 1e-5},
    ], weight_decay=1e-4)

    steps_per_epoch = len(trn)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[1e-3, 1e-5],
        total_steps=epochs * steps_per_epoch
    )

    scaler    = GradScaler()
    swa       = AveragedModel(model)
    best_loss = float('inf')
    swa_start = int(epochs * 0.75)

    # 4) Epoch Loop
    for ep in range(epochs):
        # ep == freeze_epochs 시 backbone unfreeze
        if ep == freeze_epochs:
            for p in backbone_params:
                p.requires_grad = True

        # — Train —
        model.train()
        running_loss = 0.0
        for x, y in trn:
            x, y = x.to(device), y.to(device)
            mx, y1, y2, lam = mixup_cutmix(x, y, alpha=0.4)

            optimizer.zero_grad()
            with autocast():
                out  = model(mx)
                loss = lam * F.cross_entropy(out, y1) + (1 - lam) * F.cross_entropy(out, y2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()   # OneCycleLR: 배치마다 호출

            running_loss += loss.item()

        avg_train_loss = running_loss / len(trn)

        # — SWA 업데이트 —
        if ep >= swa_start:
            swa.update_parameters(model)
        if ep == epochs - 1:
            update_bn(trn, swa, device=device)
            model = swa.module.to(device)

        # — Validation —
        model.eval()
        all_preds, all_gts = [], []
        with torch.no_grad():
            for x, y in val:
                x = x.to(device)
                preds = F.softmax(model(x), dim=1).cpu().numpy()
                all_preds.append(preds)
                all_gts.append(y.numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_gts   = np.concatenate(all_gts,   axis=0)
        vl = log_loss(all_gts, all_preds)

        # — Model Save —
        if vl < best_loss:
            best_loss = vl
            sd = swa.module.state_dict() if ep >= swa_start else model.state_dict()
            torch.save(sd, f'best_{name}_fold{fold}.pth')

        print(f"[{name}] Fold{fold} Ep{ep+1}/{epochs} "
              f"TrainLoss: {avg_train_loss:.4f} ValLogLoss: {vl:.4f} "
              f"LR: {optimizer.param_groups[0]['lr']:.1e}")



# =============================================
# 6. 학습 실행
# =============================================
for model_idx in range(len(CFG['MODEL_NAMES'])):
    for fold, (tr, va) in enumerate(skf.split(samples, targets)):
        train_model(model_idx, fold, tr, va)

# =============================================
# 7. 앙상블 추론 + TTA
# =============================================
# ----------------------------
#  Config
# ----------------------------
MODEL_NAME    = 'coatnet_2_rw_224'
NUM_FOLDS     = 3
IMG_SIZE      = 224
BATCH_SIZE    = 16
TEST_ROOT     = '/home/kim/Desktop/hecto/open/test'
SUBMIT_PATH   = '/home/kim/Desktop/hecto/sample_submission.csv'
OUTPUT_PATH   = '/home/kim/Desktop/hecto/coatnet2_3fold_ensemble.csv'

# ----------------------------
#  Device
# ----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# ----------------------------
#  Dataset
# ----------------------------
class TestDataset(Dataset):
    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        img  = Image.open(path).convert('RGB')
        return self.transform(img)

# ----------------------------
#  Load submission + classes + IDs
# ----------------------------
sub_df     = pd.read_csv(SUBMIT_PATH, encoding='utf-8-sig')
id_col     = sub_df.columns[0]
ids        = sub_df[id_col].astype(str).tolist()
class_cols = sub_df.columns[1:].tolist()

# ----------------------------
#  Prepare test file list in the same order as IDs
# ----------------------------
# if your IDs lack extension, append '.jpg'; else adjust as needed
test_paths = [os.path.join(TEST_ROOT, f"{img_id}.jpg") for img_id in ids]

# ----------------------------
#  TTA Transforms
# ----------------------------
normalize = T.Normalize([0.485,0.456,0.406],
                        [0.229,0.224,0.225])
base = [T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), normalize]

tta_transforms = [
    T.Compose(base),
    T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)),
               T.RandomHorizontalFlip(1.0)] + base[1:]),
    T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)),
               T.ColorJitter(0.1,0.1)]     + base[1:])
]

# ----------------------------
#  Load 3-fold models
# ----------------------------
models = []
for fold in range(NUM_FOLDS):
    m = create_model(MODEL_NAME, pretrained=True, num_classes=len(class_cols))
    m.load_state_dict(torch.load(f'best_{MODEL_NAME}_fold{fold}.pth', map_location=device))
    m.eval().to(device)
    models.append(m)

# ----------------------------
#  Inference with TTA + fold ensemble
# ----------------------------
all_fold_preds = []

for fold_idx, model in enumerate(models, start=1):
    print(f'Inference — fold {fold_idx}')
    tta_preds = []
    for tta_idx, tt in enumerate(tta_transforms, start=1):
        ds     = TestDataset(test_paths, transform=tt)
        loader = DataLoader(ds,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)

        preds = []
        with torch.no_grad():
            for imgs in tqdm(loader, desc=f'Fold{fold_idx}-TTA{tta_idx}', leave=False):
                imgs = imgs.to(device)
                out  = model(imgs)
                p    = F.softmax(out, dim=1).cpu().numpy()
                preds.append(p)

        preds = np.concatenate(preds, axis=0)
        tta_preds.append(preds)

    # TTA 평균
    fold_pred = np.stack(tta_preds, axis=0).mean(axis=0)  # shape (N_test, num_classes)
    all_fold_preds.append(fold_pred)

# Fold 앙상블 평균
ensemble_pred = np.stack(all_fold_preds, axis=0).mean(axis=0)  # shape (N_test, num_classes)

# ----------------------------
#  Save submission (ID 순서 보장)
# ----------------------------
sub_df[class_cols] = ensemble_pred
sub_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
print(f'Ensembled predictions saved to {OUTPUT_PATH}')
