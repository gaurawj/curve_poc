"""Synthetic tensile-test curves with labelled faults.

Engineering stress (MPa) vs strain (fraction), sampled over time with a denser
grid in the elastic region, as a real machine would do. All numbers are
illustrative. The purpose is to exercise the pipeline end to end and to give the
POC a labelled benchmark before real data is available. Metrics on this data are
NOT performance claims.
"""
from __future__ import annotations

import numpy as np

RATE = 0.002          # strain rate, 1/s (constant, simplified)
DT_FINE = 0.01        # s, sampling interval in the elastic region
DT_COARSE = 0.10      # s, sampling interval after E_SWITCH
E_SWITCH = 0.02       # strain where sampling becomes coarser
DROP = np.array([0.45, 0.15, 0.05, 0.02, 0.01, 0.005])   # load fall after fracture

MATERIALS = {
    # steel with yield plateau (Lueders strain `lud`)
    "S235": dict(kind="steel", E=205_000.0, Re=285.0, Rm=430.0, Ag=0.19, Af=0.30, lud=0.018),
    # aluminium alloy, smooth yielding
    "AlMg": dict(kind="alu", E=69_000.0, Re=275.0, Rm=315.0, Ag=0.085, Af=0.12, lud=0.0),
}

FAULTS = [
    "grip_slippage",
    "misalignment",
    "premature_fracture",
    "extensometer_slip",
    "noise_spikes",
    "preload_offset",
]


# ---------------------------------------------------------------- clean curve
def _params(mat: str, rng: np.random.Generator) -> dict:
    m = MATERIALS[mat]
    E = m["E"] * (1 + rng.normal(0, 0.02))
    Re = m["Re"] * (1 + rng.normal(0, 0.03))
    Rm = max(m["Rm"] * (1 + rng.normal(0, 0.02)), Re * 1.12)
    Ag = max(m["Ag"] * (1 + rng.normal(0, 0.07)), 0.5 * m["Ag"])
    Af = Ag + (m["Af"] - m["Ag"]) * (1 + rng.normal(0, 0.10))
    lud = m["lud"] * (1 + rng.normal(0, 0.10)) if m["lud"] else 0.0
    return dict(kind=m["kind"], E=E, Re=Re, Rm=Rm, Ag=Ag, Af=Af, lud=lud)


def _true_stress(eps: np.ndarray, p: dict) -> np.ndarray:
    E, Re, Rm, Ag, Af = p["E"], p["Re"], p["Rm"], p["Ag"], p["Af"]
    sf = 0.75 * Rm
    if p["kind"] == "steel":
        ey = Re / E
        rel = 0.97 * Re
        e_h = ey + p["lud"]
        s = np.where(eps <= ey, E * eps, 0.0)
        plateau = rel + (Re - rel) * np.exp(-(eps - ey) / 0.0004) \
            + 0.003 * rel * np.sin(2 * np.pi * (eps - ey) / 0.0025)
        s = np.where((eps > ey) & (eps <= e_h), plateau, s)
        x = np.clip((eps - e_h) / (Ag - e_h), 0, 1)
        k = 4.0
        hard = rel + (Rm - rel) * (1 - np.exp(-k * x)) / (1 - np.exp(-k))
        s = np.where((eps > e_h) & (eps <= Ag), hard, s)
    else:
        n = np.log((Ag - Rm / E) / 0.002) / np.log(Rm / Re)
        sg = np.linspace(0, Rm, 6000)
        eg = sg / E + 0.002 * (sg / Re) ** n
        s = np.where(eps <= Ag, np.interp(eps, eg, sg), 0.0)
    x = np.clip((eps - Ag) / (Af - Ag), 0, 1)
    neck = Rm - (Rm - sf) * x ** 1.6
    return np.where(eps > Ag, neck, s)


def _time_grid(af: float) -> np.ndarray:
    t_switch = E_SWITCH / RATE
    t1 = np.arange(0.0, t_switch, DT_FINE)
    t2 = np.arange(t_switch, af / RATE + DT_COARSE, DT_COARSE)
    return np.concatenate([t1, t2])


def _append_fracture(t, eps, s, s_end):
    t_d = t[-1] + DT_COARSE * np.arange(1, len(DROP) + 1)
    eps_d = eps[-1] + RATE * DT_COARSE * np.arange(1, len(DROP) + 1)
    return (np.concatenate([t, t_d]), np.concatenate([eps, eps_d]),
            np.concatenate([s, s_end * DROP]))


def _build_clean(mat: str, rng: np.random.Generator):
    p = _params(mat, rng)
    t = _time_grid(p["Af"])
    eps = RATE * t
    keep = eps <= p["Af"]
    t, eps = t[keep], eps[keep]
    toe = rng.uniform(1e-4, 8e-4) if rng.random() < 0.35 else 0.0   # benign slack
    s = _true_stress(np.maximum(eps - toe, 0.0), p)
    t, eps, s = _append_fracture(t, eps, s, s[-1])
    ey = p["Re"] / p["E"] + (0.002 if p["kind"] == "alu" else 0.0)
    e_h = ey + (p["lud"] if p["kind"] == "steel" else 0.0)
    meta = dict(p, toe=toe, ey=ey, e_h=e_h)
    return t, eps, s, meta


def _idx(eps: np.ndarray, e: float) -> int:
    return int(np.clip(np.searchsorted(eps, e), 0, len(eps) - 1))


# --------------------------------------------------------------------- faults
def _slip(t, eps, s, m, rng, sev=1.0):
    """Grip slippage: sudden load drop that recovers over a few mm."""
    if rng.random() < 0.4:
        e0 = rng.uniform(0.3, 0.9) * m["ey"]                       # elastic region
    else:
        e0 = rng.uniform(m["e_h"] + 0.005, 0.8 * m["Ag"])          # plastic region
    i0 = _idx(eps, e0)
    i1 = _idx(eps, e0 + rng.uniform(0.003, 0.01))
    d = min(rng.uniform(0.03, 0.12) * sev * m["Rm"], 0.5 * s[i0])
    w = 1 - (eps[i0:i1 + 1] - eps[i0]) / max(eps[i1] - eps[i0], 1e-9)
    s[i0:i1 + 1] -= d * w
    return t, eps, s, (float(eps[i0]), float(eps[i1]))


def _misalign(t, eps, s, m, rng, sev=1.0):
    """Bending from misalignment: lower apparent modulus and a rounded knee."""
    f_e = 1 - rng.uniform(0.07, 0.22) * sev
    b = rng.uniform(0.02, 0.06) * sev
    e_eff = np.maximum(eps - m["toe"], 0.0)
    ek = 2.5 * m["ey"]
    deficit = (1 - f_e) * m["E"] * e_eff * np.exp(-(e_eff / ek) ** 2)
    ramp = 1 - np.exp(-(e_eff / m["ey"]) ** 2)
    out = np.maximum(s * (1 - b * ramp) - deficit, 0.0)
    return t, eps, out, (0.0, float(3 * m["ey"]))


def _premature(t, eps, s, m, rng, sev=1.0):
    """Fracture well before the expected elongation (defect, grip-region break)."""
    frac = 1 - (1 - rng.uniform(0.3, 0.97)) * sev       # lower severity = closer to Ag
    e_b = max(frac * m["Ag"], 1.2 * m["ey"])
    ib = _idx(eps, e_b)
    t, eps, s = t[:ib + 1], eps[:ib + 1], s[:ib + 1]
    t, eps, s = _append_fracture(t, eps, s, s[-1])
    return t, eps, s, (float(eps[ib]) - 0.002, float(eps[-1]))


def _ext_slip(t, eps, s, m, rng, sev=1.0):
    """Extensometer slip: sudden strain jump with no change in load."""
    if rng.random() < 0.4:
        e0 = rng.uniform(0.3, 0.9) * m["ey"]
    else:
        e0 = rng.uniform(m["e_h"] + 0.005, 0.8 * m["Ag"])
    i0 = _idx(eps, e0)
    jump = rng.uniform(0.003, 0.015) * sev
    eps[i0 + 1:] += jump
    return t, eps, s, (float(eps[i0]), float(eps[i0 + 1]))


def _offset(t, eps, s, m, rng, sev=1.0):
    """Load not zeroed / specimen preloaded by clamping."""
    s0 = rng.uniform(8, 40) * sev * rng.choice([-1, 1])
    return t, eps, s + s0, (0.0, 0.001)


STRUCTURAL = {
    "grip_slippage": _slip,
    "misalignment": _misalign,
    "premature_fracture": _premature,
    "extensometer_slip": _ext_slip,
    "preload_offset": _offset,
}


def _spikes(s, eps, m, rng, sev=1.0):
    n = len(s)
    lo, hi = 20, n - 12
    if rng.random() < 0.5:
        k = int(rng.integers(2, 7))
        idx = np.sort(rng.choice(np.arange(lo, hi), k, replace=False))
        s[idx] += rng.choice([-1, 1], k) * rng.uniform(0.015, 0.06, k) * sev * m["Rm"]
        seg = (float(eps[idx[0]]), float(eps[idx[-1]]))
    else:
        i0 = int(rng.integers(lo, hi - 25))
        s[i0:i0 + 20] += rng.normal(0, rng.uniform(3, 6) * sev, 20)
        seg = (float(eps[i0]), float(eps[i0 + 20]))
    return s, seg


# ------------------------------------------------------------------- datasets
def make_curve(mat: str, fault: str | None, rng: np.random.Generator, cid: str,
               sev: float = 1.0) -> dict:
    """`sev` scales fault magnitude: 1.0 = clear faults, 0.25 = subtle ones."""
    t, eps, s, meta = _build_clean(mat, rng)
    seg = None
    if fault in STRUCTURAL:
        t, eps, s, seg = STRUCTURAL[fault](t, eps.copy(), s.copy(), meta, rng, sev)
    s = s + rng.normal(0, 1, len(s)) * (0.25 + 3e-4 * np.abs(s))      # load-cell noise
    eps = eps + rng.normal(0, 5e-7, len(eps))                         # strain noise
    if fault == "noise_spikes":
        s, seg = _spikes(s, eps, meta, rng, sev)
    return dict(id=cid, material=mat, label=fault or "normal", t=t,
                strain=eps, stress=s, seg=seg, meta=meta)


def make_dataset(mat: str, n_normal: int, n_per_fault: int, seed: int,
                 contamination: float = 0.0, tag: str = "", severity: float = 1.0) -> list[dict]:
    """`contamination` mixes unlabelled-in-practice faults into a 'historical' set."""
    rng = np.random.default_rng(seed)
    curves = []
    for _ in range(n_normal):
        curves.append(make_curve(mat, None, rng, f"{mat}-{tag}{len(curves):04d}"))
    for fault in FAULTS:
        for _ in range(n_per_fault):
            curves.append(make_curve(mat, fault, rng, f"{mat}-{tag}{len(curves):04d}", severity))
    for _ in range(int(round(contamination * n_normal))):
        fault = FAULTS[int(rng.integers(len(FAULTS)))]
        curves.append(make_curve(mat, fault, rng, f"{mat}-{tag}{len(curves):04d}", severity))
    order = rng.permutation(len(curves))
    return [curves[i] for i in order]
