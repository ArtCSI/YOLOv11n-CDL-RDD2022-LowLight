
# common.py — shared modules and utilities for the YOLOv11n-CDL reconstruction project.
# Import this from every notebook / training script so ConvSmart, DSA, the augmentation
# pipelines, and the dark-set generator are identical everywhere. Source of truth for the
# module math: YOLOv11n-CDL_reconstruction_spec.md, Sections 4 and 10.

import re
import inspect
import random
import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# 4.1 ConvSmart (spec Sec 4.1, Eqs 1-6, Fig 3)
# ----------------------------------------------------------------------------
class ConvSmart(nn.Module):
    # F1 = Conv3x3(X)              # ceil(c2/2) channels, standard conv, no bias
    # F2 = Conv5x5(X)               # floor(c2/2) channels, standard conv, no bias
    # F_merge = concat([F1, F2])    # c2 channels
    # F_id = Conv1x1(X)             # c2 channels (residual path)
    # F_sum = F_merge + F_id
    # F_pool = MaxPool2d(k=2, s=2)(F_sum)  if s > 1 else F_sum
    # F_out = SiLU(Norm(F_pool))           # Norm = BN, or IBN if ibn=True
    #
    # Constructor mirrors Ultralytics' Conv(c1, c2, k, s) call convention so YAML args
    # [c2, 3, 2] work through parse_model unmodified. `k` is accepted for YAML compatibility
    # but ignored - kernel sizes are fixed at 3x3/5x5/1x1 per the paper. [DECISION, spec 4.1]
    # odd c2 gives the extra channel to the 3x3 branch. [DECISION, spec 10.2] `ibn=True`
    # replaces the single BN with an Instance/Batch split-norm (IBN-Net style): the first
    # half of channels get InstanceNorm2d(affine=True), the rest get BatchNorm2d.
    def __init__(self, c1, c2, k=3, s=1, ibn=False):
        super().__init__()
        c_3x3 = -(-c2 // 2)   # ceil
        c_5x5 = c2 // 2       # floor
        assert c_3x3 + c_5x5 == c2
        self.conv3 = nn.Conv2d(c1, c_3x3, 3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(c1, c_5x5, 5, 1, 2, bias=False)
        self.conv1 = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if s and s > 1 else nn.Identity()
        self.ibn = bool(ibn)
        if self.ibn:
            self.in_ch = c2 // 2
            self.bn_ch = c2 - self.in_ch
            self.inorm = nn.InstanceNorm2d(self.in_ch, affine=True)
            self.bnorm = nn.BatchNorm2d(self.bn_ch)
        else:
            self.norm = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.c1, self.c2, self.s = c1, c2, s

    def forward(self, x):
        f1 = self.conv3(x)
        f2 = self.conv5(x)
        f_merge = torch.cat([f1, f2], dim=1)
        f_id = self.conv1(x)
        f_sum = f_merge + f_id
        f_pool = self.pool(f_sum)
        if self.ibn:
            a, b = torch.split(f_pool, [self.in_ch, self.bn_ch], dim=1)
            f_norm = torch.cat([self.inorm(a), self.bnorm(b)], dim=1)
        else:
            f_norm = self.norm(f_pool)
        return self.act(f_norm)


# ----------------------------------------------------------------------------
# 4.2 DSA — Double-Stage Attention (spec Sec 4.2, Eqs 7-10, Fig 4)
# ----------------------------------------------------------------------------
class SEBlock(nn.Module):
    # Standard Squeeze-and-Excitation: GAP -> FC-ReLU-FC-Sigmoid (r=16, min 4 hidden
    # units) -> reweight input. [DECISION, spec 4.2] output is the reweighted feature map,
    # not just the gate.
    def __init__(self, c, r=16):
        super().__init__()
        hidden = max(c // r, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, c, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.sigmoid(self.fc2(self.relu(self.fc1(self.pool(x)))))
        return x * w


class SpatialAttention(nn.Module):
    # CBAM-style: channel-wise mean+max -> concat (2ch) -> 7x7 conv -> sigmoid ->
    # reweight input. [DECISION, spec 4.2] - paper cites Mnih 2014 / Xue 2021 without
    # giving an explicit form.
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, 1, k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        w = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * w


class DSA(nn.Module):
    # Channel-preserving: (B,C,H,W) -> (B,C,H,W).
    # Stage 1 (parallel):  X_SE1 = SE(X)              X_SP1 = SpatialAttention(X)
    # Stage 2 (fuse):      X1 = Conv1x1(concat[X_SE1, X_SP1])          # 2C -> C
    # Stage 3 (cross-fed): X_SE2 = SE(X1 + X_SP1)     X_SP2 = SpatialAttention(X1 + X_SE1)
    # Final fusion:        X_out = Conv1x1(concat[X_SE2, X_SP2])       # 2C -> C
    # [DECISION, spec 4.2] the two 1x1 fusion convs are Conv+BN+SiLU (unspecified in the
    # paper). YAML arg (e.g. 1024) is accepted but ignored: DSA is channel-preserving and
    # c1 always equals c2 for this module (see parse_model registration notes, spec Sec 6).
    def __init__(self, c1, c2=None, *args):
        super().__init__()
        c = c1
        self.se1, self.sp1 = SEBlock(c), SpatialAttention()
        self.fuse1 = nn.Sequential(nn.Conv2d(2 * c, c, 1, bias=False), nn.BatchNorm2d(c), nn.SiLU())
        self.se2, self.sp2 = SEBlock(c), SpatialAttention()
        self.fuse2 = nn.Sequential(nn.Conv2d(2 * c, c, 1, bias=False), nn.BatchNorm2d(c), nn.SiLU())

    def forward(self, x):
        x_se1, x_sp1 = self.se1(x), self.sp1(x)
        x1 = self.fuse1(torch.cat([x_se1, x_sp1], dim=1))
        x_se2 = self.se2(x1 + x_sp1)
        x_sp2 = self.sp2(x1 + x_se1)
        return self.fuse2(torch.cat([x_se2, x_sp2], dim=1))


# ----------------------------------------------------------------------------
# Registration into ultralytics.nn.tasks.parse_model (spec Section 6)
# ----------------------------------------------------------------------------
def register_custom_modules():
    # Idempotent: safe to call any number of times per process (and after importlib.reload).
    #
    # Ultralytics builds modules inside parse_model(); how a module is constructed
    # (which args it receives, how channels propagate) depends on membership in a LOCAL
    # frozenset `base_modules` - not a public registry. We recompile parse_model from its
    # own source with ConvSmart AND DSA added to base_modules, so both receive
    # (c1, c2, *args) with width/max_channels scaling applied:
    #   ConvSmart [c2, 3, s(, ibn)] -> ConvSmart(c1, c2_scaled, 3, s(, ibn))
    #   DSA       [1024]            -> DSA(c1=256, c2=256)   (n scale; channel-preserving)
    # (Leaving DSA out of base_modules would build it as DSA(1024) with the unscaled YAML
    # arg while its input has 256 channels -> shape crash at model build.)
    # Verified against the parse_model source of ultralytics==8.3.152 (pinned in cell 2).
    import ultralytics.nn.tasks as tasks

    tasks.ConvSmart = ConvSmart
    tasks.DSA = DSA
    globals()['ConvSmart'] = ConvSmart
    globals()['DSA'] = DSA

    ns = getattr(tasks, '_cdl_patched_ns', None)
    if ns is not None:
        # Already patched in this process: only refresh the class references the patched
        # function resolves (covers importlib.reload(common) creating new class objects,
        # which would otherwise break pickling of saved checkpoints).
        ns['ConvSmart'] = ConvSmart
        ns['DSA'] = DSA
        return

    src = inspect.getsource(tasks.parse_model)
    patched, n = re.subn(r'(base_modules\s*=\s*frozenset\(\s*\{)',
                         r'\1ConvSmart, DSA, ', src, count=1)
    if n == 0:
        raise RuntimeError(
            "Could not find `base_modules = frozenset({...})` in this Ultralytics version's "
            "parse_model. Print inspect.getsource(ultralytics.nn.tasks.parse_model), add "
            "ConvSmart and DSA to the channel-changing module set by hand, and re-check the "
            "pinned version in cell 2."
        )
    ns = dict(vars(tasks))
    exec(compile(patched, '<parse_model_patched_for_ConvSmart_DSA>', 'exec'), ns)
    tasks.parse_model = ns['parse_model']
    tasks._cdl_patched_ns = ns
    print("Registered ConvSmart and DSA into ultralytics.nn.tasks.parse_model "
          f"(ultralytics {__import__('ultralytics').__version__}).")


# ----------------------------------------------------------------------------
# 10.1 Pipeline A — low-light TRAINING augmentation (R3, R4 only)
# ----------------------------------------------------------------------------
# Photometric only (boxes unchanged). Applied with probability ~0.3 per training image.
# Deliberately different code AND different parameter ranges from pipeline B (spec 10.1,
# 10.3) so the training augmentation and the test corruption are not literally the same
# transform.
import albumentations as A

def build_pipeline_a(p=0.3):
    # Gamma darkening + brightness scale + additive Gaussian/ISO noise + optional mild
    # colour-temperature shift. Used only on TRAIN images, only for R3/R4.
    return A.Compose([
        A.OneOf([
            A.RandomGamma(gamma_limit=(150, 300), p=1.0),          # gamma ~1.5-3.0
            A.RandomBrightnessContrast(brightness_limit=(-0.8, -0.4), contrast_limit=0.1, p=1.0),  # scale ~0.2-0.6
        ], p=1.0),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.2, 0.5), p=1.0),
        ], p=0.8),
        A.RGBShift(r_shift_limit=8, g_shift_limit=4, b_shift_limit=-8, p=0.3),  # mild colour-temp shift
    ], p=p)


def install_pipeline_a(trainer_or_model=None):
    # Hook pipeline A into Ultralytics' training-time Albumentations transform by
    # monkey-patching ultralytics.data.augment.Albumentations.__init__. The ORIGINAL
    # __init__ is stashed on the class (survives importlib.reload) so
    # uninstall_pipeline_a() can restore stock behaviour for no-aug runs.
    # NOTE: while installed, Ultralytics' default light Albumentations (Blur/MedianBlur/
    # ToGray/CLAHE at p=0.01) are replaced by pipeline A - note this in the report.
    import ultralytics.data.augment as aug_mod

    if not hasattr(aug_mod, 'Albumentations'):
        raise RuntimeError(
            "This Ultralytics version has no `Albumentations` class in "
            "ultralytics.data.augment - adjust this hook for the pinned version."
        )
    cls = aug_mod.Albumentations
    if not hasattr(cls, '_orig_init_cdl'):
        cls._orig_init_cdl = cls.__init__
    orig_init = cls._orig_init_cdl
    pipeline_a = build_pipeline_a(p=0.3)

    def patched_init(self, p=1.0):
        orig_init(self, p=p)
        self.contains_spatial = False
        self.transform = pipeline_a
        print("pipeline A (low-light training augmentation) installed, p=0.3")

    cls.__init__ = patched_init


def uninstall_pipeline_a():
    # Restore Ultralytics' stock Albumentations __init__ (needed before any no-aug run
    # that follows an aug run in the same session).
    import ultralytics.data.augment as aug_mod
    cls = aug_mod.Albumentations
    if hasattr(cls, '_orig_init_cdl'):
        cls.__init__ = cls._orig_init_cdl
        print("pipeline A uninstalled - stock Ultralytics augmentation restored")
    else:
        print("pipeline A was never installed - nothing to restore")


# ----------------------------------------------------------------------------
# 10.3 Pipeline B — TEST dark-set generator (test_dark_mild / test_dark_severe)
# ----------------------------------------------------------------------------
# Physically motivated, deliberately different code/params from pipeline A. Generated
# ONCE from the frozen test split, fixed seed, saved to disk, never regenerated.
def _srgb_to_linear(img01):
    a = 0.055
    return np.where(img01 <= 0.04045, img01 / 12.92, ((img01 + a) / (1 + a)) ** 2.4)

def _linear_to_srgb(lin):
    a = 0.055
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, lin * 12.92, (1 + a) * np.power(lin, 1 / 2.4) - a)

def darken_pipeline_b(img_uint8, exposure_scale, poisson_peak=255.0, read_noise_sigma=0.01,
                       rng=None):
    # img_uint8: HxWx3 uint8 RGB. exposure_scale: mild ~0.25, severe ~0.08 (spec 10.3).
    # poisson_peak=255: at 30 the shot noise swamps the signal (mid-gray mild: mean 58,
    # std 30; SNR ~1 for severe), so the test would measure noise more than darkness.
    rng = rng or np.random.default_rng(0)
    img01 = img_uint8.astype(np.float64) / 255.0
    lin = _srgb_to_linear(img01)
    lin_dark = lin * exposure_scale
    # Poisson shot noise (scale-dependent) + Gaussian read noise, then convert back.
    photons = np.clip(lin_dark, 0, 1) * poisson_peak
    noisy_photons = rng.poisson(photons).astype(np.float64)
    noisy_lin = noisy_photons / poisson_peak
    noisy_lin += rng.normal(0, read_noise_sigma, size=noisy_lin.shape)
    noisy_lin = np.clip(noisy_lin, 0, 1)
    out01 = _linear_to_srgb(noisy_lin)
    return np.clip(out01 * 255.0, 0, 255).astype(np.uint8)
