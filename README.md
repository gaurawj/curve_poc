# Curve Plausibility and Anomaly Detection - POC

Flags suspect stress-strain curves, names the deviating strain segment, suggests a
likely cause. Advisory only: it never changes a result. Runs offline on a standard PC.

## Run it
    pip install -r requirements.txt
    python run_poc.py            # about a minute; writes results/
Outputs in `results/`: `metrics.md` (headline table), `examples.png`,
`example_annotations.txt` (what a user would see), `test_flags.csv`, `metrics.json`.

## What is inside
- `synth.py` synthetic S235 (yield plateau) and AlMg (smooth yield) curves with six
  labelled faults and a severity knob (1.0 clear, 0.25 subtle).
- `features.py` 15 explainable curve features (modulus, Rp0.2, Rm, Ag, A, load drops, spikes, ...).
- `detect.py` seven deterministic rules (each tied to one cause and a strain segment)
  plus an Isolation Forest per material for shapes no rule anticipates.
- `run_poc.py` trains on "historical" curves (3 % unlabelled faults mixed in, as in real life), tests on labelled ones.
- `loaders.py` adapter for real exports (column names and units are assumptions).

## Results on synthetic data (severity 1, rules + ML)
Precision 0.989, recall 0.994, F1 0.992, false-positive rate 0.7 %; cause correct 0.98, segment hit 0.99.
Recall falls with subtle faults: 0.994 / 0.908 / 0.728 at severity 1.0 / 0.5 / 0.25
(misalignment drops to 0.12 at 0.25).

**Caveat:** synthetic curves written by the same author as the detector. These numbers show
the pipeline works; they are not evidence of performance on real tests.
Known weak spots: some elastic-region grip slips are attributed to misalignment; near-Ag
premature fractures are caught only as "unexplained shape"; ~0.5-0.7 % false alarms on clean curves.

## Plan with ZwickRoell (the real POC)
Goal: precision and recall on real known-bad tests, per material and standard.
1. Data request (week 1): 300-1,000 historical curves for 1-2 materials/standards (ISO 6892-1 steel, a polymer or aluminium),
   raw time/force/extension plus specimen geometry; sample export format; anonymised is fine.
2. Labelling (weeks 1-2): a lab expert marks each curve normal / bad plus cause, in `labels.csv`;
   about 50-100 known-bad tests are enough to start. Two reviewers on a sample to measure agreement.
3. Adapt and run (weeks 2-3): fix `loaders.py`, retrain per material, tune thresholds on a separate split.
4. Report (week 4): precision, recall, false alarms per 1,000 tests, cause-attribution accuracy, review of misses with the lab.
Success criteria (to agree up front): recall >= 90 % on clear faults, false alarms <= 1-2 % of tests,
correct cause in the top suggestion >= 80 %, every flag explainable to an operator.

## How it stays advisory
Annotation only, no write-back to results, deterministic rules carry the explanations,
the ML model is trained offline per material and does not keep learning (matches ZwickRoell's own validated-AI stance).

## Risks
Real exports differ from the assumptions here; real faults are subtler and messier than synthetic ones;
label quality limits every metric; thresholds need per-standard tuning; polymers and composites need new features.
