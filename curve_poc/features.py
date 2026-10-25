"""Feature extraction from one tensile curve (time, strain, stress).

Every feature is a plain, explainable quantity an engineer can check by eye.
Locations (where on the strain axis something happened) are returned alongside,
so a flag can point at the deviating segment.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import medfilt

FEATURES = [
    "E_gpa",        # apparent Young's modulus from the linear stretch 10-50 % of Rm
    "nonlin",       # 1 - R^2 of that linear fit
    "curv",         # relative slope change inside the elastic fit window
    "toe_pct",      # strain-axis intercept of the elastic line, %
    "rp02",         # proof stress Rp0.2, MPa
    "rm",           # tensile strength, MPa
    "rp_rm",        # Rp0.2 / Rm
    "ag_pct",       # strain at maximum force, %
    "a_pct",        # strain at fracture, %
    "neck_pct",     # a_pct - ag_pct
    "drawdown",     # largest fall from the running maximum before Rm, fraction of Rm
    "step_drop",    # largest single-sample fall before Rm, fraction of Rm
    "spikes",       # count of spike or noise-burst samples (second-difference test)
    "jump",         # largest strain increment per unit time / median
    "init_off",     # stress of the first sample / Rm (zero or preload error)
]


def robust_sigma(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def extract(c: dict) -> tuple[dict, dict]:
    t, eps, s = c["t"], c["strain"], c["stress"]
    n = len(s)
    sf = medfilt(s, 5)

    i_pk = int(np.argmax(sf))
    rm = float(sf[i_pk])
    below = np.where(sf[i_pk:] < 0.5 * rm)[0]
    i_fr = i_pk + int(below[0]) - 1 if len(below) else n - 1
    i_fr = max(i_fr, i_pk)

    # elastic fit between 10 % and 50 % of Rm
    lo = int(np.argmax(sf >= 0.10 * rm))
    hi = int(np.argmax(sf >= 0.50 * rm))
    if hi - lo < 6:
        hi = min(lo + 6, n - 1)
    x, y = eps[lo:hi + 1], s[lo:hi + 1]
    slope, icpt = np.polyfit(x, y, 1)
    res = y - (slope * x + icpt)
    r2 = 1 - res.var() / max(y.var(), 1e-12)
    mid = (lo + hi) // 2
    s1 = np.polyfit(eps[lo:mid + 1], s[lo:mid + 1], 1)[0]
    s2 = np.polyfit(eps[mid:hi + 1], s[mid:hi + 1], 1)[0]
    curv = (s2 - s1) / slope
    toe = -icpt / slope

    # proof stress: offset line crossing
    line = slope * (eps - toe - 0.002)
    cand = np.where(sf[hi:i_pk + 1] <= line[hi:i_pk + 1])[0]
    rp = float(sf[hi + cand[0]]) if len(cand) else rm
    i_el = int(np.argmax(sf >= 0.9 * rp))

    run_max = np.maximum.accumulate(sf[:i_pk + 1])
    drawdown = float(np.max(run_max - sf[:i_pk + 1]) / rm)

    step, i_step = 0.0, 0
    if i_pk > 1:
        d1 = np.diff(sf[:i_pk + 1])
        i_step = int(np.argmin(d1))
        step = max(0.0, -float(d1[i_step])) / rm

    # noise level from second differences (robust to slope, immune to the many
    # exact zeros that a median-filter residual has on steep ramps)
    d2 = s[2:i_fr + 1] - 2 * s[1:i_fr] + s[:i_fr - 1]
    sig = max(robust_sigma(d2) / np.sqrt(6.0), 0.05)
    # A single-sample spike of height A leaves the second-difference signature
    # (A, -2A, A); a load step (grip slip) leaves (D, -D, ~0) and a ramp leaves ~0,
    # so only genuine spikes match. A noise burst shows up as many large second
    # differences in one curve.
    thr = 12 * sig
    big = np.abs(d2) > thr
    pat = np.zeros(len(d2), dtype=bool)
    if len(d2) > 2:
        a_, b_, c_ = d2[:-2], d2[1:-1], d2[2:]
        rb = np.abs(b_)
        pat[1:-1] = ((rb > thr) & (a_ * b_ < 0) & (c_ * b_ < 0)
                     & (np.abs(a_) > 0.25 * rb) & (np.abs(a_) < 0.75 * rb)
                     & (np.abs(c_) > 0.25 * rb) & (np.abs(c_) < 0.75 * rb))
    # a burst = the densest cluster of large second differences (gaps <= 5 samples);
    # lone corners, e.g. at the yield point, are not part of it
    idx_big = np.where(big)[0]
    burst_idx = np.array([], dtype=int)
    if len(idx_big) >= 6:
        clusters = np.split(idx_big, np.where(np.diff(idx_big) > 5)[0] + 1)
        best = max(clusters, key=len)
        if len(best) >= 6:
            burst_idx = best
    spike_idx = np.union1d(np.where(pat)[0], burst_idx).astype(int) + 1

    dedt = np.diff(eps) / np.diff(t)
    jump = float(np.max(dedt) / max(np.median(dedt), 1e-12))
    i_jump = int(np.argmax(dedt))

    f = dict(
        E_gpa=slope / 1000.0,
        nonlin=float(1 - r2),
        curv=float(curv),
        toe_pct=float(toe * 100),
        rp02=rp,
        rm=rm,
        rp_rm=rp / rm,
        ag_pct=float(eps[i_pk] * 100),
        a_pct=float(eps[i_fr] * 100),
        neck_pct=float((eps[i_fr] - eps[i_pk]) * 100),
        drawdown=drawdown,
        step_drop=step,
        spikes=int(len(spike_idx)),
        jump=jump,
        init_off=float(s[0] / rm),
    )
    loc = dict(
        i_pk=i_pk, i_fr=i_fr, i_step=i_step, i_jump=i_jump, i_el=i_el,
        fit=(float(eps[lo]), float(eps[hi])),
        spike_idx=spike_idx,
        e_el_end=float(eps[i_el]), e_pk=float(eps[i_pk]), e_fr=float(eps[i_fr]),
        sf=sf,
    )
    return f, loc
