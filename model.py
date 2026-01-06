import math
import os
import random
import time
import argparse
from collections import defaultdict
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torchvision import datasets, transforms

try:
    from SPSD import SPSD2D, PoolType
    _HAVE_SPSD = True
except Exception:
    _HAVE_SPSD = False


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.scale


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.size(0),) + (1,) * (x.ndim - 1)
        rand = x.new_empty(shape).bernoulli_(keep)
        return x.div(keep) * rand


# =====================
# Patch Embedding
# =====================
class ImgEmbeddings(nn.Module):
    def __init__(self, img_size=256, patch_size=32, in_chans=3, embed_dim=192, dropout=0.1):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.patch_hw = img_size // patch_size
        self.num_patches = self.patch_hw * self.patch_hw
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1,2)  # (B,N,C)
        if x.size(1) != self.pos.size(1):
            target = x.size(1)
            posv = self.pos.transpose(1,2)  # (1,C,N)
            posv = F.interpolate(posv, size=target, mode='nearest')
            pos = posv.transpose(1,2)
        else:
            pos = self.pos
        x = x + pos
        return self.drop(x)


# =====================
# Fractal Adapter (Local SPSD)
# =====================
class FractalAdapter(nn.Module):

    def __init__(self, in_chans, frac_dim, patch_num, N2=9, offset=3, poolType=None):
        super().__init__()
        self.kind = 'spsd'
        self.patch_num = patch_num
        self.frac_dim = frac_dim
        self.N2 = int(N2)
        self.offset = offset

        if poolType is None:
            try:
                from SPSD import PoolType as _PT
                poolType = _PT.MAX
            except Exception:
                poolType = None
        self.poolType = poolType

        if not _HAVE_SPSD:
            raise RuntimeError("SPSD module not found. Please ensure SPSD.py is available.")
        self.spsd = SPSD2D(minAlpha=0, maxAlpha=3, N=self.N2, offset=self.offset, poolType=self.poolType)
        self.linear_proj = None
        self.aux_head = nn.Sequential(
            nn.Linear(frac_dim, max(1, frac_dim//2)),
            nn.GELU(),
            nn.Linear(max(1, frac_dim//2), 1)
        )

    def _init_proj(self, in_dim, out_dim):
        if (self.linear_proj is None) or (self.linear_proj.in_features != in_dim) or (self.linear_proj.out_features != out_dim):
            self.linear_proj = nn.Linear(in_dim, out_dim)

    def forward(self, img):

        B, C, H, W = img.shape
        Ph = Pw = self.patch_num
        ps_h = H // Ph
        ps_w = W // Pw

        patches = F.unfold(img, kernel_size=(ps_h, ps_w), stride=(ps_h, ps_w))
        patches = patches.transpose(1, 2).reshape(B * Ph * Pw, C, ps_h, ps_w)

        hist_local, _ = self.spsd(patches)

        hist_local = F.softplus(hist_local)
        hist_local = torch.log1p(hist_local)
        mu = hist_local.mean(dim=(1, 2), keepdim=True)
        sd = hist_local.std(dim=(1, 2), keepdim=True).clamp_min(1e-4)
        hist_local = (hist_local - mu) / sd

        hist_local = hist_local.reshape(B, Ph * Pw, -1)
        aux = hist_local.max(-1).values 

        self._init_proj(hist_local.size(-1), self.frac_dim)
        self.linear_proj = self.linear_proj.to(hist_local.device)
        frac_tok = self.linear_proj(hist_local)
        frac_tok = F.layer_norm(frac_tok, frac_tok.shape[-1:])
        return frac_tok, aux


# =====================
# Attention + Graph
# =====================
class Attention(nn.Module):
    def __init__(self, dim, heads, drop=0.1):
        super().__init__()
        assert dim % heads == 0
        self.h = heads
        self.d = dim // heads
        self.q = Linear(dim, dim)
        self.k = Linear(dim, dim)
        self.v = Linear(dim, dim)
        self.o = Linear(dim, dim)
        self.drop_attn = Dropout(drop)
        self.drop_proj = Dropout(drop)
    def _split(self, x):
        B,N,C = x.shape
        return x.view(B,N,self.h,self.d).permute(0,2,1,3).contiguous()
    def forward(self, x, tau=1.0):
        Q = self._split(self.q(x))
        K = self._split(self.k(x))
        V = self._split(self.v(x))
        logits = (Q @ K.transpose(-1,-2)) / (math.sqrt(self.d) * max(1e-4, float(tau)))
        prob = logits.softmax(dim=-1)
        prob = self.drop_attn(prob)
        out = (prob @ V).permute(0,2,1,3).contiguous().view(x.size())
        out = self.drop_proj(self.o(out))
        return out, logits


@torch.no_grad()
def _row_topk_edges(sim_bn, K, undirected=True, add_self_loops=False, edge_drop=0.0):

    B, N, _ = sim_bn.shape
    device = sim_bn.device
    K = int(K)
    maxK = max(1, N - 1)
    K = min(maxK, max(1, K))
    vals, idx = torch.topk(sim_bn, k=K, dim=-1)  # (B,N,K)
    if edge_drop > 0.0:
        m = (torch.rand_like(vals) > edge_drop).float()
        vals = vals * m + (1 - m) * (-1e9)
    row = torch.arange(N, device=device).view(1,N,1).expand(B,N,K)
    offset = (torch.arange(B, device=device) * N).view(B,1,1)
    r = (row + offset).reshape(-1).long()
    c = (idx + offset).reshape(-1).long()
    w = vals.reshape(-1).float()
    if undirected:
        r0, c0, w0 = r, c, w
        r = torch.cat([r0, c0], 0)
        c = torch.cat([c0, r0], 0)
        w = torch.cat([w0, w0], 0)
    if add_self_loops:
        sl = torch.arange(B*N, device=device)
        r = torch.cat([r, sl], 0)
        c = torch.cat([c, sl], 0)
        w = torch.cat([w, torch.ones_like(sl, dtype=w.dtype)], 0)
    valid_mask = (r >= 0) & (r < B*N) & (c >= 0) & (c < B*N)
    if (~valid_mask).any():
        r = r[valid_mask]; c = c[valid_mask]; w = w[valid_mask]
    if r.numel() == 0:
        sl = torch.arange(B*N, device=device)
        r = sl; c = sl; w = torch.ones_like(sl, dtype=torch.float32)
        return torch.stack([r,c],0), w
    return torch.stack([r,c],0), w


class FractalDiffusionConv(nn.Module):
    def __init__(self, dim, K=3, p_init=1.0, share_theta=True, dropout=0.0):
        super().__init__()
        self.K = int(K)
        self.log_p = nn.Parameter(torch.log(torch.tensor(float(p_init))))
        self.share = bool(share_theta)
        if self.share:
            self.theta = nn.Linear(dim, dim, bias=False)
        else:
            self.theta = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(self.K+1)])
        self.theta2 = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)
    def _power_weights(self, device):
        p = self.log_p.exp().clamp(0.1,6.0)
        ks = torch.arange(self.K+1, device=device, dtype=torch.float32)
        ks[0] = 1.0
        raw = torch.where(ks>0, ks.pow(-p), torch.ones_like(ks))
        return raw / (raw.sum() + 1e-6)
    def forward(self, x_flat, edge_index, edge_weight, num_nodes):
        row, col = edge_index[0], edge_index[1]
        feats = []
        h = x_flat
        feats.append(h)
        for _ in range(1, self.K+1):
            ew = edge_weight
            if ew.numel() != row.numel():
                ew = ew.new_ones(row.numel())
            msg = h[col] * ew.unsqueeze(-1)
            y = h.new_zeros((num_nodes, h.size(1)))
            y = y.index_add(0, row, msg)
            h = y
            feats.append(h)
        w = self._power_weights(x_flat.device)
        out = 0.0
        if self.share:
            for k in range(self.K+1):
                out = out + w[k] * self.theta(feats[k])
        else:
            for k in range(self.K+1):
                out = out + w[k] * self.theta[k](feats[k])
        out = self.theta2(F.gelu(out))
        return self.drop(out)


# =====================
# GHA Encoder
# =====================
class GHABlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, k_top, drop=0.1, K_diff=3, p_init=1.0, share_theta=True,
                 tau_init=1.0, droppath=0.1, layerscale_init=1e-3):
        super().__init__()
        self.n1 = RMSNorm(dim)
        self.n2 = RMSNorm(dim)
        self.attn = Attention(dim, heads, drop)
        self.tau = nn.Parameter(torch.tensor(float(tau_init)))
        self.k_top = int(k_top)
        self.diff = FractalDiffusionConv(dim, K=K_diff, p_init=p_init, share_theta=share_theta, dropout=drop)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(mlp_dim, dim), nn.Dropout(drop))
        self.gamma_g = nn.Parameter(torch.ones(1) * layerscale_init)
        self.gamma_m = nn.Parameter(torch.ones(1) * layerscale_init)
        self.drop = DropPath(droppath)
        self.register_buffer("_eye", torch.empty(0, dtype=torch.bool), persistent=False)
    def _mask_diag(self, A):
        B,N,_ = A.shape
        if self._eye.numel()==0 or self._eye.size(0)!=N:
            self._eye = torch.eye(N, dtype=torch.bool, device=A.device)
        return A.masked_fill(self._eye.unsqueeze(0), 0.0)
    def forward(self, x, frac_sim=None, mix_lambda=0.0):
        B,N,C = x.shape
        y, logits = self.attn(self.n1(x), tau=float(self.tau.clamp(0.5,2.0).item()))
        attn_sim = logits.softmax(dim=-1).mean(1)
        attn_sim = self._mask_diag(attn_sim)
        sim = attn_sim if frac_sim is None else (1.0 - mix_lambda) * attn_sim + mix_lambda * frac_sim
        k_use = min(max(1, self.k_top), N - 1 if N>1 else 1)
        ei, ew = _row_topk_edges(sim, k_use, undirected=True)
        if ew is not None and ei is not None:
            row = ei[0]
            if row.numel() > 0:
                sum_per_row = ew.new_zeros(B*N).index_add(0, row, ew)
                denom = (sum_per_row[row] + 1e-6)
                ew = ew / denom
            else:
                ew = ew.new_ones(ei.size(1))
        y_flat = y.reshape(B*N, C)
        num_nodes = B*N
        g_out_flat = self.diff(y_flat, ei, ew if ew is not None else torch.ones(ei.size(1), device=ei.device), num_nodes)
        g_out = g_out_flat.view(B,N,C)
        x = x + self.drop(self.gamma_g * g_out)
        x = x + self.drop(self.gamma_m * self.mlp(self.n2(x)))
        return x, attn_sim


class GHAEncoder(nn.Module):
    def __init__(self, dim, heads, layers, mlp_dim, k_list, drop=0.1, droppath=0.1, K_diff=3, p_init=1.0, share_theta=True):
        super().__init__()
        if isinstance(k_list, int):
            k_list = [k_list] * layers
        assert len(k_list) == layers
        dps = torch.linspace(0, droppath, steps=layers).tolist()
        self.blocks = nn.ModuleList([
            GHABlock(dim, heads, mlp_dim, k_list[i], drop, K_diff=K_diff, p_init=p_init, share_theta=share_theta, droppath=dps[i])
            for i in range(layers)
        ])
        self.norm = RMSNorm(dim)
    def forward(self, x, frac_sim=None, inject_layer=0, mix_lambda=0.0):
        attn_sims = []
        for i, blk in enumerate(self.blocks):
            sim_for_blk = frac_sim if i >= inject_layer else None
            x, attn_sim = blk(x, frac_sim=sim_for_blk, mix_lambda=mix_lambda)
            attn_sims.append(attn_sim)
        return self.norm(x), attn_sims



class FD_ViG(nn.Module):
    def __init__(self,
                 num_classes=21,
                 img_size=256,
                 patch_size=32,
                 in_chans=3,
                 embed_dim=192,
                 num_heads=6,
                 num_layers=4,
                 mlp_dim=768,
                 k_list=6,
                 dropout=0.1,
                 use_cls=False,
                 frac_dim=None,
                 frac_N2=9,
                 frac_offset=3,
                 inject_layer=None,
                 mix_lambda_init=0.05,
                 aux_init_w=0.25):
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_num = img_size // patch_size
        self.embed = ImgEmbeddings(img_size, patch_size, in_chans, embed_dim, dropout)
        frac_dim = frac_dim or embed_dim
        self.frac_adapter = FractalAdapter(in_chans, frac_dim, self.patch_num, N2=frac_N2, offset=frac_offset)
        self.frac_proj = nn.Linear(frac_dim, embed_dim) if frac_dim != embed_dim else nn.Identity()
        self.gate_linear = nn.Linear(embed_dim*2, embed_dim)
        self.alpha_param = nn.Parameter(torch.tensor(-2.0))
        self.spec_sharp = nn.Parameter(torch.tensor(0.03))
        # encoder
        self.encoder = GHAEncoder(embed_dim, num_heads, num_layers, mlp_dim, k_list, drop=dropout, droppath=0.1)
        self.inject_layer = inject_layer if inject_layer is not None else max(0, num_layers-2)
        self.mix_param = nn.Parameter(torch.logit(torch.tensor(float(mix_lambda_init)), eps=1e-6))
        self.aux_init_w = aux_init_w
        self.use_cls = bool(use_cls)
        if self.use_cls:
            self.cls = nn.Parameter(torch.zeros(1,1,embed_dim))
            nn.init.trunc_normal_(self.cls, std=0.02)
        self.head = Linear(embed_dim, num_classes)
        self.aux_head = nn.Linear(embed_dim, 1)
        self.aux_loss_fn = nn.SmoothL1Loss()

    def forward(self, img, return_aux=False, align_mode='min'):

        B = img.size(0)
        patch_tok = self.embed(img)  # (B,Np,C)
        frac_tok, aux_target = self.frac_adapter(img)  # (B,Nf,frac_dim), (B,Nf)
        frac_tok = self.frac_proj(frac_tok)  # (B,Nf,C)

        frac_tok = frac_tok + self.spec_sharp.tanh() * (patch_tok - frac_tok)

        frac_norm = F.normalize(frac_tok, dim=-1)
        cos = torch.einsum('bnc,bmc->bnm', frac_norm, frac_norm).clamp(-1, 1)
        temp = 0.85
        frac_sim = (cos / temp).softmax(dim=-1)

        gate_in = torch.cat([patch_tok, frac_tok], dim=-1)
        gate = torch.sigmoid(self.gate_linear(gate_in))
        x = gate * patch_tok + (1.0 - gate) * frac_tok

        alpha = torch.sigmoid(self.alpha_param)
        x = x + alpha * (patch_tok - x) * 0.05

        if self.use_cls:
            cls = self.cls.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)

        mix_lambda = torch.sigmoid(self.mix_param)
        x, attn_sims = self.encoder(x, frac_sim=frac_sim, inject_layer=self.inject_layer, mix_lambda=mix_lambda)
        feat = x[:,0] if self.use_cls else x.mean(1)
        logits = self.head(feat)
        if return_aux:
            aux_pred = self.aux_head(frac_tok).squeeze(-1)
            return logits, aux_pred, aux_target
        return logits


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_device(pref=None):
    if pref: return torch.device(pref)
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

def stratified_indices(labels: List[int], val_ratio=0.15, test_ratio=0.15, seed=42):
    rs = np.random.RandomState(seed)
    buckets = defaultdict(list)
    for i, y in enumerate(labels):
        buckets[int(y)].append(i)
    train_idx, val_idx, test_idx = [], [], []
    for _, idxs in buckets.items():
        rs.shuffle(idxs)
        n = len(idxs)
        nv = int(round(n*val_ratio)); nt = int(round(n*test_ratio)); tr = n-nv-nt
        train_idx += idxs[:tr]; val_idx += idxs[tr:tr+nv]; test_idx += idxs[tr+nv:tr+nv+nt]
    rs.shuffle(train_idx); rs.shuffle(val_idx); rs.shuffle(test_idx)
    return train_idx, val_idx, test_idx

def topk_acc(logits, y, k=1):
    topk = logits.topk(k, dim=1).indices
    eq = topk.eq(y.view(-1,1).expand_as(topk))
    return eq.any(dim=1).float().mean().item()

class MixupCollate:
    def __init__(self, alpha=0.2, p=1.0):
        self.alpha = alpha; self.p = p
    def __call__(self, batch):
        imgs, labels = zip(*batch)
        x = torch.stack(imgs, dim=0)
        y = torch.tensor(labels, dtype=torch.long)
        if self.alpha <= 0 or random.random() > self.p:
            return x, y, None
        lam = np.random.beta(self.alpha, self.alpha)
        idx = torch.randperm(x.size(0))
        x = lam * x + (1 - lam) * x[idx]
        y2 = y[idx]
        return x, (y, y2, lam), "mixup"

def mixup_criterion(criterion, pred, target):
    if isinstance(target, tuple):
        y1,y2,lam = target
        return lam * criterion(pred, y1) + (1-lam) * criterion(pred, y2)
    else:
        return criterion(pred, target)

def build_loaders(data_root, img_size=256, batch_size=32, workers=4, val_ratio=0.15, test_ratio=0.15, seed=42, mixup_alpha=0.3):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    tf_train = [
        transforms.RandomResizedCrop(img_size, scale=(0.7,1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    try:
        from torchvision.transforms import RandAugment
        tf_train.append(RandAugment(num_ops=2, magnitude=9))
    except Exception:
        pass
    tf_train += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    try:
        tf_train.append(transforms.RandomErasing(p=0.25, scale=(0.02,0.2), ratio=(0.3,3.3)))
    except Exception:
        pass
    train_tf = transforms.Compose(tf_train)
    eval_tf = transforms.Compose([transforms.Resize((img_size,img_size)), transforms.ToTensor(), transforms.Normalize(mean, std)])
    base = datasets.ImageFolder(data_root, transform=transforms.ToTensor())
    labels = base.targets if hasattr(base, "targets") else [s[1] for s in base.samples]
    tr_idx, va_idx, te_idx = stratified_indices(labels, val_ratio, test_ratio, seed)
    train_ds = datasets.ImageFolder(data_root, transform=train_tf)
    val_ds = datasets.ImageFolder(data_root, transform=eval_tf)
    test_ds = datasets.ImageFolder(data_root, transform=eval_tf)
    mixup_collate = MixupCollate(alpha=mixup_alpha, p=1.0 if mixup_alpha>0 else 0.0)
    common = dict(num_workers=workers, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_ds, tr_idx), batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=mixup_collate, **common)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(val_ds, va_idx), batch_size=batch_size, shuffle=False, **common)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(test_ds, te_idx), batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader, base.classes

class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k,v in model.state_dict().items() if v.dtype.is_floating_point}
    @torch.no_grad()
    def update(self, model, decay=None):
        if decay is None: decay = self.decay
        for k, v in model.state_dict().items():
            if k in self.shadow and v.dtype.is_floating_point:
                self.shadow[k].mul_(decay).add_(v.detach(), alpha=1.0-decay)
    def apply(self, model):
        self.backup = {}
        st = model.state_dict()
        for k in self.shadow:
            self.backup[k] = st[k].clone()
            st[k].copy_(self.shadow[k])
    def restore(self, model):
        st = model.state_dict()
        for k in self.backup:
            st[k].copy_(self.backup[k])
        self.backup = {}

def build_scheduler(optimizer, epochs, min_lr_ratio=1e-3):
    def lr_lambda(e):
        return min_lr_ratio + 0.5*(1+math.cos(math.pi*e/epochs))*(1 - min_lr_ratio)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_one_epoch(model, loader, optimizer, device, scaler, amp, ema, epoch, total_epochs, aux_init_w=0.25, aux_min_w=0.02):
    model.train()
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    mse = nn.SmoothL1Loss()
    loss_sum = acc_sum = n = 0
    frac_w = max(aux_min_w, aux_init_w * (0.96 ** epoch))
    for x, target, tag in loader:
        x = x.to(device, non_blocking=True)
        if tag == "mixup":
            y = (target[0].to(device), target[1].to(device), target[2])
        else:
            y = target.to(device)
        optimizer.zero_grad(set_to_none=True)
        if amp and device.type=="cuda":
            with torch.cuda.amp.autocast():
                logits, aux_pred, aux_target = model(x, return_aux=True)
                loss_main = mixup_criterion(ce, logits, y)
                loss_aux = mse(aux_pred, aux_target.to(device))
                loss = loss_main + frac_w * loss_aux
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            logits, aux_pred, aux_target = model(x, return_aux=True)
            loss_main = mixup_criterion(ce, logits, y)
            loss_aux = mse(aux_pred, aux_target.to(device))
            loss = loss_main + frac_w * loss_aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        # EMA
        if ema is not None:
            d = 0.99985 if epoch > (total_epochs*0.6) else 0.998
            ema.update(model, decay=d)
        b = x.size(0)
        y_eval = target[0] if tag=="mixup" else target
        acc = topk_acc(logits.detach(), y_eval.to(device), k=1)
        loss_sum += loss.item() * b
        acc_sum += acc * b
        n += b
    return loss_sum/max(n,1), acc_sum/max(n,1)

@torch.no_grad()
def evaluate(model, loader, device, amp=False, ema=None, topk=(1,3)):
    if ema is not None: ema.apply(model)
    model.eval()
    ce = nn.CrossEntropyLoss()
    loss_sum = n = 0
    topk_sums = [0.0]*len(topk)
    all_preds, all_labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device)
        if amp and device.type=="cuda":
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = ce(logits, y)
        else:
            logits = model(x)
            loss = ce(logits, y)
        b = x.size(0)
        loss_sum += loss.item() * b
        for i,k in enumerate(topk):
            topk_sums[i] += topk_acc(logits, y, k=k) * b
        n += b
        all_preds.append(logits.argmax(1).cpu())
        all_labels.append(y.cpu())
    if ema is not None: ema.restore(model)
    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)
    num_classes = int(labels.max().item())+1 if labels.numel()>0 else 0
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for p,l in zip(preds, labels):
        cm[int(l), int(p)] += 1
    metrics = {
        'loss': loss_sum/max(n,1),
        'top1': topk_sums[0]/max(n,1),
        'top3': topk_sums[1]/max(n,1) if len(topk)>1 else None,
        'confusion_matrix': cm
    }
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./UCMerced_LandUse/Images")
    ap.add_argument("--save_dir", type=str, default="./ckpt_FD_ViG")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--patch_size", type=int, default=32)
    ap.add_argument("--embed_dim", type=int, default=192)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--mlp_dim", type=int, default=768)
    ap.add_argument("--num_edges", type=str, default="6,8,8,10")
    ap.add_argument("--lr", type=float, default=4e-4)           # 调高一点
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--mixup_alpha", type=float, default=0.3)   # 更强一点
    ap.add_argument("--aux_init_w", type=float, default=0.25)   # 温和的辅助权重
    ap.add_argument("--aux_min_w", type=float, default=0.02)
    ap.add_argument("--frac_N2", type=int, default=9)
    ap.add_argument("--frac_offset", type=int, default=3)
    ap.add_argument("--mix_lambda_init", type=float, default=0.05)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    train_loader, val_loader, test_loader, classes = build_loaders(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        workers=args.workers,
        val_ratio=0.15, test_ratio=0.15,
        seed=args.seed,
        mixup_alpha=args.mixup_alpha
    )
    print("Classes:", classes)

    try:
        if "," in args.num_edges:
            _k = [int(x) for x in args.num_edges.split(",")]
        else:
            _k = int(args.num_edges)
    except:
        _k = [6,8,8,10]
    k_list = _k if isinstance(_k, list) else [int(_k)]*args.layers

    model = FD_ViG(
        num_classes=len(classes),
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=3,
        embed_dim=args.embed_dim,
        num_heads=args.heads,
        num_layers=args.layers,
        mlp_dim=args.mlp_dim,
        k_list=k_list,
        dropout=0.1,
        use_cls=False,
        frac_dim=None,
        frac_N2=args.frac_N2,
        frac_offset=args.frac_offset,
        inject_layer=None,
        mix_lambda_init=args.mix_lambda_init,
        aux_init_w=args.aux_init_w
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = build_scheduler(optimizer, args.epochs, min_lr_ratio=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type=="cuda"))
    ema = EMA(model, decay=0.9999)

    best_val = 0.0
    best_path = os.path.join(args.save_dir, "best_v3.pt")
    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        # warmup
        if epoch <= 5:
            lr_scale = epoch/5.0
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr * lr_scale

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, device, scaler, args.amp, ema,
            epoch, args.epochs,
            aux_init_w=args.aux_init_w, aux_min_w=args.aux_min_w
        )
        scheduler.step()
        va_metrics = evaluate(model, val_loader, device, amp=args.amp, ema=ema, topk=(1,3))
        dt = time.time() - t0
        print(f"[{epoch:03d}/{args.epochs}] {dt:.1f}s | lr={optimizer.param_groups[0]['lr']:.2e} | "
              f"train {tr_loss:.4f}/{tr_acc*100:.2f}% | "
              f"val {va_metrics['loss']:.4f}/{va_metrics['top1']*100:.2f}% top3 {va_metrics['top3']*100:.2f}%")

        if va_metrics['top1'] > best_val:
            best_val = va_metrics['top1']
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "val_acc": va_metrics['top1'], "cfg": vars(args)}, best_path)
            print(f"  -> saved best {va_metrics['top1']*100:.2f}% to {best_path}")

    if os.path.isfile(best_path):
        ck = torch.load(best_path, map_location=device)
        model.load_state_dict(ck["state_dict"])
        print("Loaded best:", ck["epoch"], ck["val_acc"])
    te_metrics = evaluate(model, test_loader, device, amp=args.amp, ema=ema, topk=(1,3))
    print(f"[TEST] loss={te_metrics['loss']:.4f} top1={te_metrics['top1']*100:.2f}% top3={te_metrics['top3']*100:.2f}%")
    print("Confusion matrix:\n", te_metrics['confusion_matrix'])


if __name__ == "__main__":
    main()
