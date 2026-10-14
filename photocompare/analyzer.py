# -*- coding: utf-8 -*-
"""
参数反推分析：给定原图 A 与修改图 B，估算 pipeline 参数使 pipeline(A, P) ≈ B。

方法：
1. 双方降采样到工作尺寸（默认长边 256）加速；
2. 曝光用亮度对数比做解析初值；
3. 分块联合网格搜索：强耦合参数放同一块（曝光×对比度 / 高光×阴影×伽马 /
   色温×色调×自然饱和度）联合粗网格 + MSE 粗筛 + ΔE00 复评选优——
   逐参数坐标下降会陷入局部极小，且 MSE 最优点 ≠ 感知(ΔE)最优点；
4. 误差度量排除"双饱和"像素（双方都裁切到 0/1 的点不携带信息）；
5. 稀疏剪枝：删掉置零后 ΔE 恶化 <0.8% 的补偿性小参数，再做 ΔE 口径局部细扫；
6. 空间类参数（清晰度/锐化/颗粒抑制/泛光）在 768px 高分辨率上精化——
   这些参数的效果依赖噪声与细节尺度，256px 降采样会抹掉。
"""
from __future__ import annotations
import itertools
import numpy as np
import cv2

from .pipeline import (PARAM_KEYS, PARAM_RANGE, DEFAULT_PARAMS,
                       STAGE_KEYS, run_from, apply_stage,
                       apply_tone_lut, lut_is_identity)
from . import metrics

try:
    from PIL import Image
except ImportError:
    Image = None


def resize_long(img: np.ndarray, long_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = long_side / max(h, w)
    if scale >= 1.0:
        return img
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def _masked_mse(out, ref, region=None):
    """排除双饱和像素；region 给定时只统计重叠区"""
    diff = out - ref
    sq = diff * diff
    sat_hi = (out >= 0.995) & (ref >= 0.995)
    sat_lo = (out <= 0.005) & (ref <= 0.005)
    keep = ~(sat_hi | sat_lo)
    if region is not None:
        keep = keep & region[..., None]   # region 是 2D，饱和标记是 3D
    if int(keep.sum()) < 32:
        return float(sq.mean())
    return float(sq[keep].mean())


def _w1_hist(a, b, bins: int = 128, region=None) -> float:
    """每通道直方图 1-Wasserstein 距离（归一化 0~1）。
    对局部/细节类修改（清晰度/泛光/锐化）比 MSE 敏感。"""
    w = 0.0
    n = a.shape[0] * a.shape[1]
    for c in range(3):
        av = a[..., c].ravel()
        bv = b[..., c].ravel()
        if region is not None:
            rm = region.ravel()
            av, bv = av[rm], bv[rm]
        if av.size < 64:
            continue
        ha = np.histogram(av, bins, (0.0, 1.0))[0].astype(np.float64)
        hb = np.histogram(bv, bins, (0.0, 1.0))[0].astype(np.float64)
        sa, sb = ha.sum(), hb.sum()
        if sa < 1 or sb < 1:
            continue
        w += float(np.abs(np.cumsum(ha / sa) - np.cumsum(hb / sb)).mean())
    return w / 3.0


def _error(out, ref, region=None) -> float:
    """综合误差：掩码 MSE + 直方图距离"""
    return _masked_mse(out, ref, region) + 0.05 * _w1_hist(out, ref, region=region)


def _candidates(key: str, cur: float, rnd: int, n: int = 13):
    lo, hi = PARAM_RANGE[key]
    if rnd <= 1:
        return list(np.linspace(lo, hi, n))
    step = (hi - lo) / (n * 2)
    return list(np.clip(np.linspace(cur - step * (n // 2), cur + step * (n // 2), n), lo, hi))


# 容易成为"补偿项"的参数（与对比度/曝光本质耦合），引入时要求更大改善
_STRICT_ACCEPT = {"highlights": 0.995, "shadows": 0.995, "gamma": 0.995}


def _accept_for(key: str, cur: float, v: float) -> float:
    """返回该次改动的接受阈值（误差需相对改善到此比例）"""
    if key == "clarity" and v < -0.02 and abs(cur) < 0.02:
        return 0.99   # 负清晰度(柔化)易顶替"颗粒抑制"，要求 1% 改善才引入
    if key in ("clarity", "sharpen", "grain_ignore", "bloom"):
        return 0.9995  # 空间/细节类参数对全局误差影响小，放宽接受
    return _STRICT_ACCEPT.get(key, 0.999)


# ---- 分块联合搜索 ----
# (起始stage, 结束stage, 参数列表, 各参数粗网格点数)
# 强耦合参数必须同块联合搜索：逐参数扫会陷入局部极小
# （实测：对比度下降+高光压缩的组合逐参数扫永远检不出）
# gamma 与 exposure/contrast 同为 tone 形状参数，必须同一块跨 stage 联合，
# 否则块1会先用 exposure 近似 gamma 的提亮、被后续收缩锁死（case6 教训）；
# highlights/shadows 在 E/C/G 定型后做块内联合，轮间全域重锚协调
_BLOCKS = [
    (0, 4, ["exposure", "contrast", "gamma"], {"exposure": 13, "contrast": 7, "gamma": 7}),
    (2, 4, ["highlights", "shadows"],         {"highlights": 9, "shadows": 9}),
    # 跨块重定位：exposure/gamma/shadows 构成等价类族（暗部提亮可由三者多种组合实现），
    # H/S 定型后必须联合重平衡，否则锁死在等价类的另一端（case3 教训）
    (0, 4, ["exposure", "gamma", "shadows"],  {"exposure": 9, "gamma": 5, "shadows": 7}),
    (4, 6, ["temperature", "tint", "vibrance"], {"temperature": 9, "tint": 9, "vibrance": 7}),
    (6, 7, ["clarity"],                       {"clarity": 9}),
    (7, 8, ["sharpen", "grain_ignore"],       {"sharpen": 7, "grain_ignore": 5}),
    (8, 9, ["bloom"],                         {"bloom": 7}),
]


_GLOBAL_KEYS = ["exposure", "contrast", "highlights", "shadows",
                "gamma", "temperature", "tint", "vibrance"]


def _de_of(out: np.ndarray, ref_lab: np.ndarray, region=None) -> float:
    """ΔE00 均值（感知误差）；region 给定时只统计重叠区"""
    de = metrics.ciede2000_map(metrics.srgb_to_lab(out), ref_lab)
    if region is not None:
        if de[region].size < 100:
            return float(de.mean())
        return float(de[region].mean())
    return float(de.mean())


def _block_search(cache, params, unit, B, ref_lab, rnd, progress_cb, done,
                  region=None):
    """对每个参数块做联合网格搜索。
    MSE 粗筛全部组合 → top-K 用 ΔE00 复评（MSE 最优≠感知最优，实测错位严重）。
    返回更新后的 done 计数。"""
    for si0, si1, keys, grid_n in _BLOCKS:
        base = cache[si0]
        # 网格：前两轮全域（轮1全域重锚：协调块间依赖，防收缩锁死），
        # 之后在当前值附近收缩
        grids = {}
        for k in keys:
            lo, hi = PARAM_RANGE[k]
            n = grid_n[k]
            if rnd <= 1:
                grids[k] = list(np.linspace(lo, hi, n))
            else:
                step = (hi - lo) / (n * 1.5) / (rnd - 1)
                grids[k] = list(np.clip(
                    np.linspace(params[k] - step, params[k] + step, max(n // 2, 5)), lo, hi))
        scored = []
        for vals in itertools.product(*[grids[k] for k in keys]):
            trial = dict(params)
            for k, v in zip(keys, vals):
                trial[k] = float(v)
            out = run_from(base, si0, trial, unit)
            scored.append((_masked_mse(out, B, region), vals))
        # 当前值组合强制参与（全域重锚时网格可能不含当前解，防止好解被换掉）
        cur_vals = tuple(float(params[k]) for k in keys)
        cur_out = run_from(base, si0, params, unit)
        scored.append((_masked_mse(cur_out, B, region), cur_vals))
        scored.sort(key=lambda t: t[0])
        top = scored[:max(8, len(scored) * 4 // 100)]
        cands = [(_de_of(cur_out, ref_lab, region), cur_vals)]   # 当前值同样保底 dE 复评
        for mse, vals in top:
            trial = dict(params)
            for k, v in zip(keys, vals):
                trial[k] = float(v)
            cands.append((_de_of(run_from(base, si0, trial, unit), ref_lab, region), vals))
        best_d = min(c[0] for c in cands)
        # 近似等价谷底（dE 差 <0.5%）内选参数绝对值和最小的组合——
        # 仅在"几乎完全等价"时保稀疏，避免压制真实样图的表达力
        pool = [c for c in cands if c[0] < best_d * 1.005]
        best_d, best_vals = min(pool, key=lambda t: (t[0], sum(abs(float(v)) for v in t[1])))
        for k, v in zip(keys, best_vals):
            params[k] = float(v)
        for si in range(si0, si1):
            cache[si + 1] = apply_stage(cache[si], si,
                                        {k: params[k] for k in STAGE_KEYS[si]}, unit)
        done[0] += 1
        if progress_cb:
            progress_cb(done[0] / done[1],
                        f"联合搜索[{','.join(_ZH.get(k, k) for k in keys)}] "
                        f"({done[0]}/{done[1]})")
    return done


def _run_until(img, params, unit, until_stage):
    """跑 stage[0, until_stage) 之前的阶段"""
    out = img
    for j in range(until_stage):
        out = apply_stage(out, j, {k: params[k] for k in STAGE_KEYS[j]}, unit)
    return out


def _refine_spatial(A_full, B_full, params, progress_cb=None, refine_size=768,
                    region_full=None):
    """
    空间类参数（清晰度/锐化/颗粒抑制/泛光）在较高分辨率上精化——
    这些参数的效果依赖噪声与细节尺度，256px 降采样会抹掉，导致估计失真。
    """
    A2 = resize_long(np.ascontiguousarray(A_full, np.float32), refine_size)
    B2 = resize_long(np.ascontiguousarray(B_full, np.float32), refine_size)
    if B2.shape[:2] != A2.shape[:2]:
        B2 = cv2.resize(B2, (A2.shape[1], A2.shape[0]), interpolation=cv2.INTER_AREA)
    region2 = None
    if region_full is not None:
        region2 = cv2.resize(region_full.astype(np.uint8),
                             (A2.shape[1], A2.shape[0]),
                             interpolation=cv2.INTER_NEAREST) > 0
    unit2 = max(A2.shape[:2]) / 1000.0

    # 空间阶段之前的输出只算一次
    pre = _run_until(A2, params, unit2, 6)

    spatial_stages = [6, 7, 8]
    total = sum(len(STAGE_KEYS[s]) for s in spatial_stages) * 2
    done = [0]

    def report():
        done[0] += 1
        if progress_cb:
            progress_cb(min(1.0, 0.85 + 0.15 * done[0] / max(total, 1)), "高分辨率精化空间参数")

    cur = dict(params)
    for rnd in range(2):
        base = pre.copy()   # 每轮重置，避免 stage 6-8 被重复应用
        for si in spatial_stages:
            for key in STAGE_KEYS[si]:
                best_v = cur[key]
                best_e = _error(run_from(base, si, cur, unit2), B2, region2)
                lo, hi = PARAM_RANGE[key]
                for v in _candidates(key, cur[key], rnd, n=9):
                    if abs(v - cur[key]) < 1e-9:
                        continue
                    trial = dict(cur)
                    trial[key] = float(v)
                    e = _error(run_from(base, si, trial, unit2), B2, region2)
                    if e < best_e * _accept_for(key, cur[key], v):
                        best_e = e
                        best_v = float(v)
                cur[key] = best_v
                report()
            base = apply_stage(base, si, {k: cur[k] for k in STAGE_KEYS[si]}, unit2)

    params.update(cur)
    fitted_full = base
    return params, fitted_full


def _initial_guess(A: np.ndarray, B: np.ndarray) -> dict:
    """曝光初值：亮度对数比（联合估计实测会引入参数间污染，保持简单可靠）"""
    p = dict(DEFAULT_PARAMS)
    la = A[..., 0] * 0.2126 + A[..., 1] * 0.7152 + A[..., 2] * 0.0722
    lb = B[..., 0] * 0.2126 + B[..., 1] * 0.7152 + B[..., 2] * 0.0722
    ma, mb = float(la.mean()), float(lb.mean())
    if ma > 1e-4:
        p["exposure"] = float(np.clip(np.log2(mb / ma), -2.0, 2.0))
    return p


def _masked_mse_ch(out, ref, region=None):
    """单通道掩码 MSE（LUT 拟合用）"""
    sat = ((out >= 0.995) & (ref >= 0.995)) | ((out <= 0.005) & (ref <= 0.005))
    d = (out - ref) ** 2
    keep = ~sat
    if region is not None:
        keep = keep & region
    if int(keep.sum()) < 32:
        return float(d.mean())
    return float(d[keep].mean())


def _fit_tone_lut_iter(src: np.ndarray, ref: np.ndarray, npts: int = 9,
                       rounds: int = 3, smooth_w: float = 2e-4, region=None,
                       init: dict | None = None):
    """策略一：迭代版 tone 曲线（控制点坐标下降）。
    init: 初始曲线（如分位数匹配结果）——好初值下迭代收敛到更优解。
    MSE 粗筛候选 + ΔE00 复评（MSE 最优≠感知最优，同主搜索的教训）。"""
    xs = np.linspace(0.0, 1.0, npts)
    lab_ref = metrics.srgb_to_lab(ref)
    src_c = np.clip(src, 0, 1)

    cur = {"xs": xs, "R": xs.copy(), "G": xs.copy(), "B": xs.copy()}
    if init is not None:
        for k in "RGB":
            cur[k] = np.interp(xs, init["xs"], init[k]).astype(np.float64)

    def de_with(key, y):
        trial = dict(cur)
        trial[key] = y
        out = apply_tone_lut(src_c, trial)
        return _de_of(out, lab_ref, region)

    for c, key in enumerate(("R", "G", "B")):
        s = src_c[..., c]
        r = ref[..., c]
        idx = np.clip(s * 255.0, 0, 255).astype(np.int32)
        y = cur[key].copy()
        for rnd in range(rounds):
            span = 0.25 / (rnd + 1)
            for i in range(npts):
                lo = 0.0 if i == 0 else y[i - 1]
                hi = 1.0 if i == npts - 1 else y[i + 1]
                cand = np.linspace(max(lo, y[i] - span), min(hi, y[i] + span), 9)
                scored = []
                for v in cand:
                    yt = y.copy()
                    yt[i] = float(v)
                    reg = 0.0
                    if 0 < i < npts - 1:
                        reg = smooth_w * (yt[i - 1] - 2 * yt[i] + yt[i + 1]) ** 2
                    tbl = np.interp(np.linspace(0, 1, 256), xs, yt)
                    scored.append((_masked_mse_ch(tbl[idx], r, region) + reg, float(v)))
                scored.sort(key=lambda t: t[0])
                # top-2 + 当前值 用 dE 复评（感知口径）
                trials = [y[i]] + [v for _, v in scored[:2]]
                best_v, best_d = y[i], None
                for v in trials:
                    yt = y.copy()
                    yt[i] = float(v)
                    d = de_with(key, yt)
                    if best_d is None or d < best_d:
                        best_d, best_v = d, float(v)
                y[i] = best_v
        cur[key] = y

    return None if lut_is_identity(cur) else cur


def _quantile_tone_lut(src: np.ndarray, ref: np.ndarray, npts: int, region=None):
    """策略二：直方图匹配曲线（免迭代）：src 分位数 -> ref 分位数，
    采样到均匀网格。对全局色调差异拟合快而准，但会被空间性残差带偏——
    与迭代版互为补充，最终由 dE 收益门槛择优。"""
    xs = np.linspace(0.0, 1.0, npts)
    qs = np.linspace(0.0, 1.0, 257)
    lut = {"xs": xs}
    rm = region.ravel() if region is not None else None
    for c, key in enumerate("RGB"):
        s = np.clip(src[..., c], 0, 1).ravel()
        r = np.clip(ref[..., c], 0, 1).ravel()
        if rm is not None:
            s, r = s[rm], r[rm]
        s = np.sort(s)
        r = np.sort(r)
        fine_x = np.concatenate(([0.0], np.quantile(s, qs), [1.0]))
        fine_y = np.concatenate(([float(np.quantile(r, 0.0))], np.quantile(r, qs),
                                 [float(np.quantile(r, 1.0))]))
        lut[key] = np.interp(xs, fine_x, fine_y)
    return None if lut_is_identity(lut, eps=0.005) else lut


def _fit_tone_lut(src: np.ndarray, ref: np.ndarray, region=None):
    """残差 tone 曲线总调度：多策略候选池（迭代9点 / 分位初值+迭代17/33 /
    分位数多档），统一过收益门槛（应用后 dE 须改善 ≥2%），选 dE 最低者。"""
    src_c = np.clip(src, 0, 1)
    ref_c = np.clip(ref, 0, 1)
    lab_ref = metrics.srgb_to_lab(ref_c)
    base_de = _de_of(src_c, lab_ref, region)
    if base_de < 0.15:
        return None   # 已经极接近，无需曲线

    best_lut, best_de = None, base_de * 0.98   # 收益门槛
    cands = []
    lut_i = _fit_tone_lut_iter(src_c, ref_c, region=region)
    if lut_i:
        cands.append(lut_i)
    # 分位数初值 + 迭代精修：好初值让迭代收敛到更优解（实测显著优于恒等初值）
    for npts, rounds in ((17, 3), (33, 2)):
        lq = _quantile_tone_lut(src_c, ref_c, npts, region=region)
        if lq:
            cands.append(lq)
            lqi = _fit_tone_lut_iter(src_c, ref_c, npts=npts, rounds=rounds,
                                     region=region, init=lq)
            if lqi:
                cands.append(lqi)
    for npts in (9, 17, 25, 33):
        lut_q = _quantile_tone_lut(src_c, ref_c, npts, region=region)
        if lut_q:
            cands.append(lut_q)
    for lut in cands:
        out = apply_tone_lut(src_c, lut)
        d = _de_of(out, lab_ref, region)
        if d < best_de:
            best_de, best_lut = d, lut
    return best_lut


def analyze(img_a: np.ndarray, img_b: np.ndarray,
            progress_cb=None, work_size: int = 256, rounds: int = 3,
            region_mask=None):
    """
    返回 (params: dict, fitted_img: ndarray, final_mse: float, used_size: int,
          tone_lut: dict | None)
    tone_lut: 12 参数之外的残差色调曲线（修图 App 曲线工具类修改），
    预览/固化时需在管线输出后用 pipeline.apply_tone_lut 叠加。
    region_mask: bool (H,W)，自动对齐后的重叠区——所有误差/指标只在该区内计算。
    progress_cb(frac: float 0~1, message: str)
    """
    A = resize_long(np.ascontiguousarray(img_a, np.float32), work_size)
    B = resize_long(np.ascontiguousarray(img_b, np.float32), work_size)
    if B.shape[:2] != A.shape[:2]:
        B = cv2.resize(B, (A.shape[1], A.shape[0]), interpolation=cv2.INTER_AREA)
    region = None
    if region_mask is not None:
        region = cv2.resize(region_mask.astype(np.uint8), (A.shape[1], A.shape[0]),
                            interpolation=cv2.INTER_NEAREST) > 0
        if region.mean() < 0.05:      # 重叠区过小，退回全图
            region = None

    unit = max(A.shape[:2]) / 1000.0
    params = _initial_guess(A, B)
    ref_lab = metrics.srgb_to_lab(B)

    n_blocks = len(_BLOCKS)
    total = rounds * n_blocks + 12 * 2 + 4     # 块搜索 + 剪枝/细扫 + 精化
    done = [0, total]

    # ---- 分块联合网格搜索 ----
    cache = [A] + [None] * len(STAGE_KEYS)
    for rnd in range(rounds):
        _block_search(cache, params, unit, B, ref_lab, rnd, progress_cb, done,
                      region=region)

    fitted = cache[-1]

    # ---- 稀疏剪枝（ΔE 口径：删掉后感知误差恶化 <0.8% 的参数没多大用）----
    cur_d = _de_of(fitted, ref_lab, region)
    for _ in range(3):
        changed = False
        for key in PARAM_KEYS:
            if abs(params[key]) < 0.02:
                continue
            done[0] += 1
            if progress_cb:
                progress_cb(done[0] / total, f"剪枝[{_zh(key)}]")
            old = params[key]
            params[key] = 0.0
            trial = run_from(A, 0, params, unit)
            d0 = _de_of(trial, ref_lab, region)
            if d0 < cur_d * 1.008:
                cur_d = d0
                fitted = trial
                changed = True
            else:
                params[key] = old
        if not changed:
            break

    # ---- 局部细扫（ΔE 口径，剪枝后精修；仅全局参数——空间参数留给 768px 精化）----
    _SPATIAL = {"clarity", "sharpen", "grain_ignore", "bloom"}
    for _round in range(1):
        for key in PARAM_KEYS:
            if abs(params[key]) < 1e-6 or key in _SPATIAL:
                continue
            done[0] += 1
            if progress_cb:
                progress_cb(done[0] / total, f"细扫[{_zh(key)}]")
            lo, hi = PARAM_RANGE[key]
            best_v, best_d = params[key], cur_d
            for v in np.linspace(params[key] - 0.05, params[key] + 0.05, 7):
                v = float(np.clip(v, lo, hi))
                if abs(v - params[key]) < 1e-6:
                    continue
                trial = dict(params)
                trial[key] = v
                d = _de_of(run_from(A, 0, trial, unit), ref_lab, region)
                if d < best_d:
                    best_d, best_v = d, v
            if best_v != params[key]:
                params[key] = float(best_v)
                cur_d = best_d
                fitted = run_from(A, 0, params, unit)

    # ---- 高分辨率精化空间类参数（清晰度/锐化/颗粒抑制/泛光）----
    params, fitted = _refine_spatial(img_a, img_b, params, progress_cb=progress_cb,
                                     region_full=region_mask)

    # ---- 残差色调曲线（修图 App 曲线工具类修改，12 参数表达不了的形状）----
    tone_lut = None
    if progress_cb:
        progress_cb(0.97, "拟合色调曲线")
    fitted_w = resize_long(np.clip(fitted, 0, 1), work_size)
    if fitted_w.shape[:2] != B.shape[:2]:
        fitted_w = cv2.resize(fitted_w, (B.shape[1], B.shape[0]),
                              interpolation=cv2.INTER_AREA)
    tone_lut = _fit_tone_lut(fitted_w, B, region=region)
    if tone_lut is not None:
        fitted = np.clip(apply_tone_lut(np.clip(fitted, 0, 1), tone_lut), 0, 1)

    # ---- 终局精化：768px 全局参数细扫 + 二级曲线级联 ----
    # 256px 学到的参数在全尺寸上有系统性偏差（下采样改变像素分布），
    # 768px 上以 ΔE00 口径细扫修正（实测 5 组平均 +6.4pt）；
    # 二级曲线吸收一级应用后的剩余全局残差（对部分样图 +11~14pt）。
    A768 = resize_long(np.ascontiguousarray(img_a, np.float32), 768)
    B768 = resize_long(np.ascontiguousarray(img_b, np.float32), 768)
    if B768.shape[:2] != A768.shape[:2]:
        B768 = cv2.resize(B768, (A768.shape[1], A768.shape[0]),
                          interpolation=cv2.INTER_AREA)
    region768 = None
    if region_mask is not None:
        region768 = cv2.resize(region_mask.astype(np.uint8),
                               (A768.shape[1], A768.shape[0]),
                               interpolation=cv2.INTER_NEAREST) > 0
        if region768.mean() < 0.05:
            region768 = None
    lab_B768 = metrics.srgb_to_lab(B768)
    unit768 = max(A768.shape[:2]) / 1000.0

    def _de768(p):
        out = run_from(A768, 0, p, unit768)
        if tone_lut:
            out = apply_tone_lut(out, tone_lut)
        return _de_of(out, lab_B768, region768)

    cur768 = _de768(params)
    if progress_cb:
        progress_cb(0.985, "高分辨率全局精扫")
    for key in _GLOBAL_KEYS:
        if abs(params[key]) < 0.02:
            continue   # 保持稀疏：不引入新参数，只精修已检出参数
        lo, hi = PARAM_RANGE[key]
        best_v, best_d = params[key], cur768
        for v in np.linspace(params[key] - 0.03, params[key] + 0.03, 5):
            v = float(np.clip(v, lo, hi))
            if abs(v - params[key]) < 1e-6:
                continue
            t = dict(params)
            t[key] = v
            d = _de768(t)
            if d < best_d:
                best_d, best_v = d, v
        if best_v != params[key]:
            params[key] = float(best_v)
            cur768 = best_d

    fitted768 = run_from(A768, 0, params, unit768)
    if tone_lut:
        fitted768 = np.clip(apply_tone_lut(np.clip(fitted768, 0, 1), tone_lut), 0, 1)

    # 二级曲线：一级应用后的剩余全局残差（9 点迭代，门槛 ≥1% 才启用）
    if progress_cb:
        progress_cb(0.995, "二级色调曲线")
    fw2 = resize_long(fitted768, work_size)
    if fw2.shape[:2] != B.shape[:2]:
        fw2 = cv2.resize(fw2, (B.shape[1], B.shape[0]), interpolation=cv2.INTER_AREA)
    lut2 = _fit_tone_lut_iter(fw2, B, npts=9, rounds=2, region=region)
    if lut2 is not None:
        de_with2 = _de_of(apply_tone_lut(fitted768, lut2), lab_B768, region768)
        if de_with2 < cur768 * 0.99:
            tone_lut = _compose_luts(lut2, tone_lut) if tone_lut else lut2
            fitted768 = np.clip(apply_tone_lut(fitted768, lut2), 0, 1)

    fitted = fitted768
    final_mse = _masked_mse(resize_long(fitted, work_size), B, region)
    if progress_cb:
        progress_cb(1.0, "完成")
    return params, fitted, final_mse, max(A.shape[:2]), tone_lut


def _compose_luts(outer: dict, inner: dict) -> dict:
    """复合两条 tone 曲线：outer(inner(x))，采样到统一网格"""
    xs = np.linspace(0.0, 1.0, 513)
    lut = {"xs": xs}
    for k in "RGB":
        yi = np.interp(xs, inner["xs"], inner[k])
        lut[k] = np.interp(yi, outer["xs"], outer[k])
    return lut


_ZH = {
    "exposure": "曝光", "contrast": "对比度", "highlights": "高光", "shadows": "阴影",
    "gamma": "伽马", "temperature": "色温", "tint": "色调", "vibrance": "自然饱和度",
    "clarity": "清晰度", "sharpen": "锐化", "grain_ignore": "忽略胶片颗粒", "bloom": "泛光",
}


def _zh(key):
    return _ZH.get(key, key)


def describe_params(params: dict) -> str:
    """生成自然语言结论"""
    parts = []
    e = params.get("exposure", 0)
    if abs(e) > 0.03:
        parts.append(f"整体{'提亮' if e > 0 else '压暗'}约 {abs(e):.2f} EV")
    c = params.get("contrast", 0)
    if abs(c) > 0.05:
        parts.append(f"对比度{'增强' if c > 0 else '降低'} {abs(c)*100:.0f}%")
    h, s = params.get("highlights", 0), params.get("shadows", 0)
    if abs(h) > 0.08:
        parts.append(f"高光{'提亮' if h > 0 else '压暗(恢复高光)'}")
    if abs(s) > 0.08:
        parts.append(f"阴影{'提亮' if s > 0 else '压暗'}")
    g = params.get("gamma", 0)
    if abs(g) > 0.06:
        parts.append(f"伽马{'提升(中间调变亮)' if g > 0 else '下降(中间调变暗)'}")
    t = params.get("temperature", 0)
    if abs(t) > 0.05:
        parts.append(f"色温偏{'暖(黄)' if t > 0 else '冷(蓝)'} {abs(t)*100:.0f}%")
    ti = params.get("tint", 0)
    if abs(ti) > 0.05:
        parts.append(f"色调偏{'洋红' if ti > 0 else '绿'}")
    v = params.get("vibrance", 0)
    if abs(v) > 0.05:
        parts.append(f"自然饱和度{'提升' if v > 0 else '降低'} {abs(v)*100:.0f}%")
    cl = params.get("clarity", 0)
    if abs(cl) > 0.08:
        parts.append(f"清晰度(局部对比度){'增强' if cl > 0 else '柔化'}")
    sh = params.get("sharpen", 0)
    if sh > 0.08:
        lv = "轻度" if sh < 0.35 else ("中等" if sh < 0.7 else "强烈")
        parts.append(f"{lv}锐化")
    gi = params.get("grain_ignore", 0)
    if gi > 0.2 and sh > 0.08:
        parts.append("锐化时抑制了噪点(胶片颗粒)")
    bl = params.get("bloom", 0)
    if bl > 0.1:
        parts.append(f"高光泛光 {bl*100:.0f}%")
    if not parts:
        return "未检测到明显修改（两图基本一致）"
    return "、".join(parts) + "。"


def align_pair(img_a: np.ndarray, img_b: np.ndarray, feature_size: int = 1200):
    """自动对齐：SIFT 粗对齐 + ECC 像素级精化，把 B 映射到 A 的坐标系。

    返回 (b_aligned, mask, info)：
      b_aligned: 与 A 同尺寸的对齐后 B（重叠区外为 -1）
      mask: bool (H,W)，True=两图都有效的重叠区（边缘已收缩）
      info: 人类可读的对齐说明；对齐失败时 b_aligned/mask 为 None。
    """
    a8 = cv2.cvtColor((np.clip(img_a, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    b8 = cv2.cvtColor((np.clip(img_b, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    ha, wa = a8.shape
    hb, wb = b8.shape

    # 大图缩到 feature_size 提速（H 结果按缩放比例换算回去）
    sc = feature_size / max(ha, wa, hb, wb)
    use_small = sc < 1.0
    if use_small:
        sa, sb = feature_size / max(ha, wa), feature_size / max(hb, wb)
        a_s = cv2.resize(a8, (int(round(wa * sa)), int(round(ha * sa))),
                         interpolation=cv2.INTER_AREA)
        b_s = cv2.resize(b8, (int(round(wb * sb)), int(round(hb * sb))),
                         interpolation=cv2.INTER_AREA)
    else:
        sa = sb = 1.0
        a_s, b_s = a8, b8

    sift = cv2.SIFT_create(nfeatures=8000)
    k1, d1 = sift.detectAndCompute(a_s, None)
    k2, d2 = sift.detectAndCompute(b_s, None)
    if d1 is None or d2 is None or len(k1) < 30 or len(k2) < 30:
        return None, None, "特征不足，无法自动对齐"
    good = [m for m, n in cv2.BFMatcher().knnMatch(d2, d1, k=2)
            if m.distance < 0.75 * n.distance]
    if len(good) < 30:
        return None, None, "匹配点过少，无法自动对齐"
    src = np.float32([k2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    Hf, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if Hf is None or inl is None or int(inl.sum()) < 30:
        return None, None, "对齐失败（匹配点不稳定）"
    inlier_ratio = float(inl.sum()) / len(good)

    # ECC 像素级精化（在特征尺寸上修亚像素残差）
    Bwarp0 = cv2.warpPerspective(b_s, Hf, (a_s.shape[1], a_s.shape[0]),
                                 flags=cv2.INTER_LINEAR)
    ecc_dx = ecc_dy = 0.0
    try:
        gB = cv2.cvtColor((np.clip(Bwarp0, 0, 1) * 255).astype(np.uint8),
                          cv2.COLOR_RGB2GRAY)
        waf = np.eye(2, 3, dtype=np.float32)
        cc, waf = cv2.findTransformECC(
            a_s, gB, waf, cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-7), None, 5)
        A3 = np.vstack([waf, [0, 0, 1]])
        Hf = A3 @ Hf
        ecc_dx, ecc_dy = float(waf[0, 2]), float(waf[1, 2])
    except cv2.error:
        pass   # ECC 失败不影响主流程（SIFT 已达亚像素）

    # H 换算回原尺寸：H_full = S_a^-1 · Hf · S_b
    if use_small:
        Sa = np.array([[sa, 0, 0], [0, sa, 0], [0, 0, 1]], np.float64)
        Sb = np.array([[sb, 0, 0], [0, sb, 0], [0, 0, 1]], np.float64)
        Hfull = np.linalg.inv(Sa) @ Hf @ Sb
    else:
        Hfull = Hf

    Bw = cv2.warpPerspective(np.ascontiguousarray(img_b, np.float32), Hfull,
                             (wa, ha), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
    valid = (Bw > -0.5).all(axis=-1)
    if valid.mean() < 0.10:
        return None, None, "两图重叠部分过小"
    mask = cv2.erode(valid.astype(np.uint8),
                     np.ones((7, 7), np.uint8)).astype(bool)

    dx, dy = float(Hfull[0, 2]), float(Hfull[1, 2])
    info = (f"已自动对齐：平移 dx={dx:+.2f}px dy={dy:+.2f}px，"
            f"重叠区 {mask.mean()*100:.1f}%（匹配内点率 {inlier_ratio*100:.0f}%）")
    if abs(ecc_dx) > 1e-4 or abs(ecc_dy) > 1e-4:
        info += f"，ECC 像素级精化 {ecc_dx:+.2f}/{ecc_dy:+.2f}px"
    return Bw, mask, info


def load_image(path: str):
    """读图 -> (float32 [0,1] RGB full, float32 RGB preview长边1800, info str)"""
    if Image is None:
        raise RuntimeError("需要 Pillow 读取图片")
    from PIL import ImageOps
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    im = im.convert("RGB")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    prev = resize_long(arr, 1800)
    info = f"{im.width}×{im.height}px"
    return arr, prev, info
