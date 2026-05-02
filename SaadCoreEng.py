import os, io, json, math, random, warnings, logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import timm
from sklearn.metrics import f1_score, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

@dataclass
class GlobalConfig:
    BASE_DIR: str = "./data"
    SAVE_DIR: str = "./results"
    EPOCHS: int = 30
    BATCH_SIZE: int = 32
    IMG_SIZE: int = 384
    SEED: int = 42
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    PSO_PARTICLES: int = 24
    PSO_ITERS: int = 50
    GAMMA: float = 0.8

L_NAMES = ["N","D","G","C","A","H","M","O"]
NC = len(L_NAMES)

def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def _g_gamma(img, g=0.8):
    ig = 1.0 / g
    t = np.array([((i/255.0)**ig)*255 for i in np.arange(0,256)]).astype("uint8")
    return cv2.LUT(img, t)

def _f_crop(img, p=0.05):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, m = cv2.threshold(g, 10, 255, cv2.THRESH_BINARY)
    cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnt: return img
    c = max(cnt, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    ph, pw = int(h*p), int(w*p)
    return img[max(0,y-ph):min(img.shape[0],y+h+ph), max(0,x-pw):min(img.shape[1],x+w+pw)]

def build_retfound(nc, isz, ckpt=None):
    m = timm.create_model("vit_base_patch16_384", pretrained=True, img_size=isz, num_classes=0)
    f = nn.Sequential(m, nn.Linear(m.num_features, nc))
    if ckpt and os.path.isfile(ckpt):
        try: f.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
        except: pass
    return f

def pso_engine(stk, y_ref, n_p=24, n_i=50):
    M, N, C = stk.shape
    def _obj(p):
        w = np.exp(p[:M]) / np.sum(np.exp(p[:M]))
        t = 1.0 / (1.0 + np.exp(-p[M:]))
        l = (stk * w[:, None, None]).sum(0)
        pr = 1.0 / (1.0 + np.exp(-l))
        pd = (pr >= t[None, :]).astype(int)
        return f1_score(y_ref, pd, average="macro", zero_division=0)

    rn = np.random.RandomState(42)
    pos = rn.normal(0, 1.0, (n_p, M+C)); vel = np.zeros_like(pos)
    pb_p = pos.copy(); pb_s = np.full(n_p, -1e9)
    gb_p, gb_s = None, -1e9

    for _ in range(n_i):
        for i in range(n_p):
            s = _obj(pos[i])
            if s > pb_s[i]: pb_s[i] = s; pb_p[i] = pos[i].copy()
            if s > gb_s: gb_s = s; gb_p = pos[i].copy()
        r1, r2 = rn.rand(n_p, M+C), rn.rand(n_p, M+C)
        vel = (0.729*vel + 1.494*r1*(pb_p - pos) + 1.494*r2*(gb_p - pos))
        pos += vel
    return gb_p

class ODIRLoader(Dataset):
    def __init__(self, df, isz, train=True):
        self.df = df.reset_index(drop=True)
        t = [transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]
        if train: t = [transforms.RandomHorizontalFlip()] + t
        self.tf = transforms.Compose(t)
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        x = self.tf(Image.open(r["pp_path"]).convert("RGB"))
        y = torch.tensor([r[k] for k in L_NAMES], dtype=torch.float32)
        return x, y

if __name__ == "__main__":
    cfg = GlobalConfig()
    set_seed(cfg.SEED)
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    print(f"System logic initialized. Target: Multi-label ODIR-5K pipeline.")