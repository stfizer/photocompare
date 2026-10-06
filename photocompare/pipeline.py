# -*- coding: utf-8 -*-
"""
修图参数管线：定义 12 个可调参数 + 分阶段处理函数。
图像统一表示为 float32 ndarray, shape (H, W, 3), RGB 顺序, 范围 [0, 1]。

参数与常见调色软件（NVIDIA Freestyle / Facebook 风格滤镜）对应：
  曝光/对比度/高光/阴影/伽马、色温/色调/自然饱和度、
  清晰度(局部对比度)/锐化/忽略胶片颗粒/泛光。
"""
from __future__ import annotations
import numpy as np
import cv2

# (key, 中文名, min, max, default, 格式, 说明)
PARAM_SPECS = [
    ("exposure",     "曝光",       -2.0, 2.0, 0.0, "{:+.2f} EV", "整体亮度，单位 EV"),
    ("contrast",     "对比度",     -1.0, 1.0, 0.0, "{:+.2f}",    "中间调反差"),
    ("highlights",   "高光",       -1.0, 1.0, 0.0, "{:+.2f}",    "高光区提亮/压暗"),
    ("shadows",      "阴影",       -1.0, 1.0, 0.0, "{:+.2f}",    "暗部提亮/压暗"),
    ("gamma",        "伽马",       -1.0, 1.0, 0.0, "{:+.2f}",    "编码值 v，实际 γ=2^-v，v>0 提亮中间调"),
    ("temperature",  "色温",       -1.0, 1.0, 0.0, "{:+.2f}",    "负=偏冷(蓝)，正=偏暖(黄)"),
    ("tint",         "色调",       -1.0, 1.0, 0.0, "{:+.2f}",    "负=偏绿，正=偏洋红"),
    ("vibrance",     "自然饱和度", -1.0, 1.0, 0.0, "{:+.2f}",    "低饱和像素增强更多"),
    ("clarity",      "清晰度",     -1.0, 1.0, 0.0, "{:+.2f}",    "局部对比度(中间调大半径)"),
    ("sharpen",      "锐化",        0.0, 1.0, 0.0, "{:+.2f}",    "小半径细节锐化"),
    ("grain_ignore", "忽略胶片颗粒", 0.0, 1.0, 0.0, "{:+.2f}",    "锐化时抑制小幅度噪点"),
    ("bloom",        "泛光",        0.0, 1.0, 0.0, "{:+.2f}",    "高光溢出/柔光"),
]
PARAM_KEYS = [p[0] for p in PARAM_SPECS]
PARAM_NAMES = {p[0]: p[1] for p in PARAM_SPECS}
PARAM_RANGE = {p[0]: (p[2], p[3]) for p in PARAM_SPECS}
PARAM_FMT = {p[0]: p[5] for p in PARAM_SPECS}
PARAM_DESC = {p[0]: p[6] for p in PARAM_SPECS}
DEFAULT_PARAMS = {p[0]: p[4] for p in PARAM_SPECS}

# 阶段划分：坐标下降按此分组（同阶段参数彼此耦合）
STAGE_KEYS = [
    ["exposure"],
    ["contrast"],
    ["highlights", "shadows"],
    ["gamma"],
    ["temperature", "tint"],
    ["vibrance"],
    ["clarity"],
    ["sharpen", "grain_ignore"],
    ["bloom"],
]


def luma(img: np.ndarray) -> np.ndarray:
    """Rec.709 亮度, (H, W)"""
    return img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _clip01(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0.0, 1.0, out=img)


# ---------------- 各阶段实现 ----------------

def stage_exposure(img, sub, unit):
    return np.clip(img * (2.0 ** sub["exposure"]), 0.0, 1.0)


def stage_contrast(img, sub, unit):
    c = sub["contrast"]
    return np.clip((img - 0.5) * (1.0 + c) + 0.5, 0.0, 1.0)


def stage_tone_shadows(img, sub, unit):
    """高光/阴影：按亮度 mask 施加非线性增益"""
    L = luma(img)
    hm = _smoothstep(0.4, 1.0, L)          # 高光权重
    sm = 1.0 - _smoothstep(0.0, 0.6, L)    # 阴影权重
    gain = 1.0 + (sub["highlights"] * 0.6 * hm + sub["shadows"] * 0.6 * sm)
    return np.clip(img * gain[..., None], 0.0, 1.0)


def stage_gamma(img, sub, unit):
    g = sub["gamma"]
    if abs(g) < 1e-4:
        return img
    # 编码值 v>0 提亮: 指数 = 2^-v < 1
    return np.clip(img ** (2.0 ** (-g)), 0.0, 1.0)


def stage_color(img, sub, unit):
    """色温/色调"""
    t, ti = sub["temperature"], sub["tint"]
    out = img.copy()
    out[..., 0] *= 1.0 + 0.25 * t + 0.05 * ti   # R
    out[..., 1] *= 1.0 + 0.03 * t - 0.20 * ti   # G
    out[..., 2] *= 1.0 - 0.25 * t + 0.05 * ti   # B
    return _clip01(out)


def stage_vibrance(img, sub, unit):
    v = sub["vibrance"]
    if abs(v) < 1e-4:
        return img
    mx = img.max(axis=-1)
    mn = img.min(axis=-1)
    sat = (mx - mn) / (mx + mn + 1e-6)          # [0,1]
    scale = 1.0 + v * (1.0 - sat)               # 低饱和增强更多
    gray = luma(img)[..., None]
    return _clip01(gray + (img - gray) * scale[..., None])


def stage_clarity(img, sub, unit):
    """清晰度 = 中间调加权的局部对比度（大半径 unsharp）"""
    c = sub["clarity"]
    if abs(c) < 1e-4:
        return img
    L = luma(img)
    sigma = max(10.0 * unit, 0.4)
    blur = cv2.GaussianBlur(L, (0, 0), sigma)
    detail = L - blur
    mid = np.clip(1.0 - np.abs(L - 0.5) * 1.2, 0.15, 1.0)  # 中间调权重
    dL = c * 1.5 * detail * mid
    return _clip01(img + dL[..., None])


def stage_sharpen(img, sub, unit):
    """锐化 + 忽略胶片颗粒（软阈值抑制小幅度差异）"""
    s, gi = sub["sharpen"], sub["grain_ignore"]
    if s < 1e-4:
        return img
    L = luma(img)
    sigma = max(1.2 * unit, 0.15)
    blur = cv2.GaussianBlur(L, (0, 0), sigma)
    d = L - blur
    if gi > 1e-4:
        thr = gi * 0.03
        mag = np.abs(d)
        d = np.sign(d) * np.clip(mag - thr, 0.0, None)     # 软阈值
    dL = s * 1.2 * d
    return _clip01(img + dL[..., None])


def stage_bloom(img, sub, unit):
    b = sub["bloom"]
    if b < 1e-4:
        return img
    L = luma(img)
    hl = np.clip((L - 0.72) / 0.28, 0.0, 1.0)   # 高光提取
    glow_src = img * hl[..., None]
    sigma = max(18.0 * unit, 0.5)
    glow = cv2.GaussianBlur(glow_src, (0, 0), sigma)
    # screen 混合
    out = 1.0 - (1.0 - img) * (1.0 - glow * b * 1.5)
    return _clip01(out)


STAGE_FUNCS = [
    stage_exposure, stage_contrast, stage_tone_shadows, stage_gamma,
    stage_color, stage_vibrance, stage_clarity, stage_sharpen, stage_bloom,
]


def apply_stage(img: np.ndarray, stage_idx: int, sub: dict, unit: float) -> np.ndarray:
    return STAGE_FUNCS[stage_idx](img, sub, unit)


def run_from(img: np.ndarray, stage_idx: int, params: dict, unit: float) -> np.ndarray:
    """从第 stage_idx 阶段开始用 params 跑完整管线"""
    out = img
    for j in range(stage_idx, len(STAGE_FUNCS)):
        out = apply_stage(out, j, {k: params[k] for k in STAGE_KEYS[j]}, unit)
    return out


def apply_pipeline(img: np.ndarray, params: dict) -> np.ndarray:
    """完整管线。img: float32 [0,1] RGB；params 缺省项用 0"""
    full = dict(DEFAULT_PARAMS)
    full.update({k: float(v) for k, v in params.items() if k in full})
    unit = max(img.shape[:2]) / 1000.0
    return run_from(np.ascontiguousarray(img, dtype=np.float32), 0, full, unit)


def params_nonzero(params: dict, eps: float = 0.02) -> dict:
    """提取非零（有效）参数"""
    return {k: v for k, v in params.items() if abs(v) > eps}


# ---------------- 残差色调曲线（tone LUT） ----------------
# 结构: {"xs": np.ndarray(npts), "R": ys, "G": ys, "B": ys}，ys 为各控制点输出值。
# 12 个滑块参数表达不了修图 App"曲线工具"的任意形状（如中间调大提亮+高光肩部压缩），
# 参数反推后叠加每通道单调曲线，把剩余的全局色调差异补掉。

_LUT_TABLE = 2049


def apply_tone_lut(img: np.ndarray, lut: dict) -> np.ndarray:
    """对 [0,1] RGB 图应用 tone LUT（全局、逐通道、单调）"""
    if not lut:
        return img
    xs = lut["xs"]
    tbl = np.interp(np.linspace(0.0, 1.0, _LUT_TABLE), xs, lut["R"])
    out = np.empty_like(img)
    for c, key in enumerate(("R", "G", "B")):
        t = np.interp(np.linspace(0.0, 1.0, _LUT_TABLE), xs, lut[key])
        out[..., c] = t[np.clip(img[..., c] * (_LUT_TABLE - 1), 0,
                               _LUT_TABLE - 1).astype(np.int32)]
    del tbl
    return out


def lut_is_identity(lut: dict, eps: float = 0.01) -> bool:
    """曲线是否接近恒等（不值得启用）"""
    if not lut:
        return True
    xs = lut["xs"]
    return all(float(np.max(np.abs(lut[k] - xs))) < eps for k in ("R", "G", "B"))


def describe_tone_lut(lut: dict) -> str:
    """从曲线形状生成自然语言（与 12 参数互补的描述）"""
    if not lut or lut_is_identity(lut):
        return ""
    xs = lut["xs"]
    parts = []
    segs = [("暗部", 0.0, 0.25), ("中间调", 0.35, 0.65), ("高光", 0.75, 1.0)]
    for name, lo, hi in segs:
        sel = (xs >= lo) & (xs <= hi)
        d = float(np.mean(lut["G"][sel] - xs[sel]))
        if d > 0.02:
            parts.append(f"{name}提升")
        elif d < -0.02:
            parts.append(f"{name}压暗")
    # 通道差异（色偏）
    mid = (xs >= 0.25) & (xs <= 0.75)
    rd = float(np.mean(lut["R"][mid] - xs[mid]))
    bd = float(np.mean(lut["B"][mid] - xs[mid]))
    if rd - bd > 0.03:
        parts.append("整体偏暖")
    elif bd - rd > 0.03:
        parts.append("整体偏冷")
    return "、".join(parts) + "的色调曲线调整" if parts else "整体色调曲线微调"
