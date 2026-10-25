"""Adapter for REAL test data (assumption: the actual testXpert export format is unknown).

Expected input: one CSV per test with columns for time [s], strain [% or mm/mm] and
stress [MPa] (or force [N] plus the cross-section area, to be converted). Adapt
COLUMN_MAP and the unit handling once ZwickRoell supplies a sample export.
The detector only needs the dict that `load_csv` returns, so nothing else changes.
Labels come from a separate file: labels.csv with columns id,label,note
(label = normal or one of the fault names in synth.FAULTS, or other).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COLUMN_MAP = {"t": "Time", "strain": "Strain", "stress": "Stress"}   # ASSUMPTION
STRAIN_IN_PERCENT = True                                             # ASSUMPTION


def load_csv(path: str, cid: str, material: str, label: str = "unknown", sep: str = ",") -> dict:
    df = pd.read_csv(path, sep=sep)
    t = df[COLUMN_MAP["t"]].to_numpy(float)
    eps = df[COLUMN_MAP["strain"]].to_numpy(float)
    if STRAIN_IN_PERCENT:
        eps = eps / 100.0
    s = df[COLUMN_MAP["stress"]].to_numpy(float)
    ok = np.isfinite(t) & np.isfinite(eps) & np.isfinite(s)
    return dict(id=cid, material=material, label=label, t=t[ok], strain=eps[ok],
                stress=s[ok], seg=None, meta={})
