"""Plausibility rules + ML anomaly score, with cause and segment attribution.

Design choices (matching how ZwickRoell describes its own AI):
  * advisory only: the detector never changes a result, it annotates the curve
  * deterministic rules carry the explanations (each rule maps to one likely cause)
  * an Isolation Forest per material catches shapes no rule anticipated
  * everything is trained per material (and per standard, once real data exists)
    on historical curves, runs offline, and is cheap enough for a lab PC
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from features import FEATURES, robust_sigma

CAUSES = {
    "extensometer_slip": "Extensometer slip or knife-edge movement",
    "noise_spikes": "Electrical noise or load-cell / cable issue",
    "grip_slippage": "Grip slippage",
    "premature_fracture": "Premature fracture (specimen defect, break at the grip, or damaged specimen)",
    "preload_offset": "Zero or preload error",
    "misalignment": "Misalignment or bending (low apparent modulus)",
    "modulus_high": "Apparent modulus too high (check extensometer calibration or gauge length)",
    "strength_out_of_range": "Strength outside the historical range (check cross-section input or material batch)",
    "unexplained_shape": "Unusual curve shape, no specific rule matched (review manually)",
}
PRIORITY = {"extensometer_slip": 1, "noise_spikes": 2, "grip_slippage": 3,
            "premature_fracture": 4, "preload_offset": 5, "misalignment": 6,
            "modulus_high": 6, "strength_out_of_range": 7, "unexplained_shape": 9}


@dataclass
class Finding:
    key: str
    segment: tuple[float, float]       # strain interval (fraction)
    phase: str
    detail: str

    @property
    def text(self) -> str:
        return CAUSES[self.key]

    @property
    def priority(self) -> int:
        return PRIORITY[self.key]


@dataclass
class Report:
    curve_id: str
    ml_score: float
    rule_flag: bool
    ml_flag: bool
    findings: list[Finding] = field(default_factory=list)

    @property
    def flagged(self) -> bool:
        return self.rule_flag or self.ml_flag

    @property
    def primary(self) -> Finding | None:
        return min(self.findings, key=lambda f: f.priority) if self.findings else None


def _phase(e: float, loc: dict) -> str:
    if e < loc["e_el_end"]:
        return "elastic region"
    if e < loc["e_pk"]:
        return "yield / hardening region"
    if e <= loc["e_fr"]:
        return "necking region"
    return "end of test"


def _ml_matrix(df: pd.DataFrame) -> np.ndarray:
    x = df[FEATURES].copy()
    x["nonlin"] = np.log10(np.clip(x["nonlin"], 1e-9, None))
    x["spikes"] = np.log1p(x["spikes"])
    x["jump"] = np.log(np.clip(x["jump"], 1.0, None))
    x["step_drop"] = np.log10(x["step_drop"] + 1e-4)
    return x.to_numpy(dtype=float)


class CurveAnomalyDetector:
    """One instance per material (and per standard, with real data)."""

    def __init__(self, ml_quantile: float = 0.97, z: float = 4.0, seed: int = 0):
        self.ml_quantile, self.z, self.seed = ml_quantile, z, seed

    # ------------------------------------------------------------------ fit
    def fit(self, hist: pd.DataFrame) -> "CurveAnomalyDetector":
        self.med = hist[FEATURES].median()
        self.sig = hist[FEATURES].apply(robust_sigma)
        x = _ml_matrix(hist)
        self.scaler = RobustScaler().fit(x)
        xs = self.scaler.transform(x)
        self.iforest = IsolationForest(n_estimators=300, random_state=self.seed).fit(xs)
        self.ml_thr = float(np.quantile(-self.iforest.score_samples(xs), self.ml_quantile))
        self._xs_med = np.median(xs, axis=0)
        self._xs_sig = np.maximum(np.apply_along_axis(robust_sigma, 0, xs), 1e-3)
        return self

    def _z(self, name: str, v: float, floor_rel: float = 0.0, floor_abs: float = 0.0) -> float:
        s = max(self.sig[name], floor_rel * abs(self.med[name]), floor_abs)
        return (v - self.med[name]) / s

    # -------------------------------------------------------------- analyze
    def analyze(self, f: pd.Series, loc: dict, c: dict) -> Report:
        eps = c["strain"]
        findings: list[Finding] = []

        def add(key, seg, detail):
            findings.append(Finding(key, (float(seg[0]), float(seg[1])), _phase(seg[0], loc), detail))

        # R1 strain jump without load change
        if f["jump"] > max(3.0, 3 * self.med["jump"]):
            i = loc["i_jump"]
            add("extensometer_slip", (eps[i], eps[i + 1]),
                f"strain advanced {f['jump']:.0f}x faster than the median rate in a single sample")

        # R2 isolated spikes / noise bursts
        if f["spikes"] >= 1:
            idx = loc["spike_idx"]
            lo, hi = eps[idx.min()], eps[idx.max()]
            if hi - lo < 1e-3:
                lo, hi = lo - 5e-4, hi + 5e-4
            add("noise_spikes", (lo, hi),
                f"{int(f['spikes'])} spike / noise-burst samples (second difference above 12 noise sigmas)")

        # R3 sudden load drop (grip slippage)
        thr_drop = max(0.003, self.med["step_drop"] + 10 * max(self.sig["step_drop"], 5e-4))
        if f["step_drop"] > thr_drop:
            i, sf = loc["i_step"], loc["sf"]
            after = np.where(sf[i + 1:loc["i_pk"] + 1] >= sf[i])[0]
            j = i + 1 + int(after[0]) if len(after) else i + 1
            j = min(j, i + 1 + int(0.02 / max(np.median(np.diff(eps)), 1e-9)))
            add("grip_slippage", (eps[i], eps[min(j, len(eps) - 1)]),
                f"load fell {f['step_drop'] * 100:.1f} % of Rm in a single sample before the maximum force")

        # R4 fracture far earlier than usual
        if f["a_pct"] < 0.62 * self.med["a_pct"]:
            add("premature_fracture", (eps[loc["i_fr"]] - 0.003, eps[loc["i_fr"]]),
                f"fracture at {f['a_pct']:.1f} % strain vs typical {self.med['a_pct']:.1f} % for this material")

        # R5 zero / preload error
        if abs(f["init_off"]) > max(0.01, 8 * max(self.sig["init_off"], 1e-3)):
            add("preload_offset", (eps[0], eps[4]),
                f"first-sample load is {f['init_off'] * 100:+.1f} % of Rm")

        # R6 elastic response
        zE = self._z("E_gpa", f["E_gpa"], floor_rel=0.01)
        if abs(zE) > self.z:
            key = "misalignment" if zE < 0 else "modulus_high"
            pct = (f["E_gpa"] / self.med["E_gpa"] - 1) * 100
            add(key, loc["fit"],
                f"apparent modulus {f['E_gpa']:.0f} GPa vs typical {self.med['E_gpa']:.0f} GPa ({pct:+.0f} %)")

        # R7 strength outside the historical range
        zR = self._z("rm", f["rm"], floor_rel=0.01)
        if abs(zR) > 5:
            add("strength_out_of_range", (loc["e_pk"] - 0.005, loc["e_pk"] + 0.005),
                f"Rm {f['rm']:.0f} MPa vs typical {self.med['rm']:.0f} MPa")

        rule_flag = bool(findings)

        # ML layer
        xs = self.scaler.transform(_ml_matrix(pd.DataFrame([f])))
        score = float(-self.iforest.score_samples(xs)[0])
        ml_flag = score > self.ml_thr
        if ml_flag and not rule_flag:
            dev = np.abs((xs[0] - self._xs_med) / self._xs_sig)
            top = [FEATURES[i] for i in np.argsort(dev)[::-1][:2]]
            add("unexplained_shape", (0.0, loc["e_fr"]),
                f"anomaly score {score:.3f} above threshold {self.ml_thr:.3f}; most unusual: {', '.join(top)}")
        return Report(c["id"], score, rule_flag, ml_flag, findings)


def explain(rep: Report) -> str:
    """Human-readable annotation, the kind of note a testXpert user could see."""
    if not rep.flagged:
        return f"{rep.curve_id}: no plausibility concerns."
    lines = [f"{rep.curve_id}: FLAGGED for review (advisory only, results unchanged)"]
    for k, fd in enumerate(sorted(rep.findings, key=lambda x: x.priority), 1):
        a, b = fd.segment
        lines.append(f"  {k}. {fd.text}")
        lines.append(f"     where: {a * 100:.2f} to {b * 100:.2f} % strain ({fd.phase})")
        lines.append(f"     why:   {fd.detail}")
    return "\n".join(lines)
