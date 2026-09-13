# ecg-acquisition-context-audit

Electrocardiograms are not recorded at random: a clinician decides to order one,
at a particular moment, under a local protocol. This repository holds the
pipeline, the pre-declared comparison files, the run records and the aggregate
results behind an audit asking how much of a published waveform model's
prognostic discrimination is recoverable from that acquisition context alone,
and whether holding acquisition context fixed removes the waveform model's
advantage.

The short answers, all of which this repository regenerates: a model with no
waveform samples reaches 0.7972 AUROC for ICU admission within 24 hours,
recovering 0.9366 of the waveform arm's above-chance discrimination there; the
waveform arm keeps most of its advantage when acquisition context is held fixed;
and none of the four registered numeric predictions was met at the endpoint it
named.

## Regenerate the tables and figures

```
pip install -r requirements.txt
python reproduce.py
```

That runs against the aggregate result files committed here. It needs no
credentialed data, no GPU, and about a minute. It writes the table panels to
`results/table_main.md`, the index that every number in the paper resolves to
`results/table_main_cells.csv`, and the six figures to `results/fig/`.

## Re-derive the results from the raw data

Longer, and it needs the data below plus a CUDA GPU for the waveform arms. Set
`ECG_DATA_ROOT` to a directory outside this repository first; every script reads
its inputs and writes its outputs there, and refuses to write inside the
repository.

```
export ECG_DATA_ROOT=/path/to/data          # Windows: $env:ECG_DATA_ROOT = "..."

python scripts/build_cohort.py              # the published cohort and split
python scripts/build_features.py            # acquisition-context features
python scripts/build_tabular.py             # the benchmark's own 463 columns
python scripts/leakage_test.py              # availability test on every feature
python scripts/fetch_waveforms.py           # credentialed; see Data below
python scripts/preprocess_waveforms.py
python scripts/run_arm.py --arm R3_acqctx_pre --label deterioration_mortality_365d --seed 0
python scripts/run_waveform_arm.py          # ~79 GPU-hours for all seeds
python scripts/build_matched_cohort.py      # coarsened exact strata
python scripts/run_matched_arm.py           # trains nothing; a statistic

python scripts/score_reproduction.py        # the waveform reproduction
python scripts/score_arms.py                # nested arms, both test families
python scripts/score_matched.py             # the within-stratum comparison
python scripts/score_calibration.py
python scripts/score_decision_curve.py
python reproduce.py                         # tables and figures
```

`run_arm.py` fits one arm for one label at one seed; the reported results are
five seeds for each of nine arms at each of three labels.

## Data

Nothing here redistributes patient data, and none is included.

- **MIMIC-IV-Ext-MDS-ED v1.0.0**, the benchmark audited, from PhysioNet.
- **MIMIC-IV**, **MIMIC-IV-ED** and **MIMIC-IV-ECG**, its parent datasets.

All four are available to credentialed users under the PhysioNet Credentialed
Health Data Use Agreement, which requires completing the CITI training and
signing the agreement. Obtain them there and point `ECG_DATA_ROOT` at the
directory holding them.

The published cohort and the published 20-fold patient-stratified split are
reproduced without modification. Row-level predictions are not included: the
run records carry a SHA-256 and a row count for each prediction file instead,
which is enough to verify a regenerated one without redistributing it.

## Layout

| Path | What is in it |
|---|---|
| `scripts/` | the pipeline, the statistics, and the table and figure generators |
| `configs/` | `base.yaml` plus one overlay per arm; the merged result is hashed to produce a run id |
| `comparisons/` | the pre-declared comparison files, committed before the runs they govern |
| `results/` | the aggregate tables and figures every number in the paper comes from |
| `runs/` | one directory per run: its manifest, resolved config, metrics and provenance |
| `literature/` | the structured search behind the related-work section: the protocol, every query as issued, every screening decision with its reason, and the table of located studies |

The paper's related-work section argues an absence, and an absence is only as
strong as the search behind it. That search is in `literature/`, in four parts.
`protocol.md` was written before any query ran. `search-record.md` separates the
strategy specification from what was actually issued, because a rendered
platform strategy nobody executed is the commonest unreproducible element of a
published search section. `screening-decisions.md` carries every excluded record
with a specific reason, and says which count does not exist for this review and
why it is not estimated rather than giving an estimate. `located-studies.md` is
the row-per-study table of every located study against the non-signal comparator
arm it reported.

One thing that record does not contain, stated here because it is the kind of
gap a reader should not have to find: three of the four concept blocks were
logged without a taken-forward column, and it reads as not recorded rather than
as nothing taken. What each search contributed is in the screening decisions,
which are complete.

## On two things a reader will notice

`comparisons/`, `configs/` and `runs/` are shipped byte-for-byte rather than
tidied, and they carry identifiers of the form `D-041` that refer to entries in
the authors' internal decision log. They are left in place deliberately. The
configuration files are hashed to produce the run ids, so editing one would
desynchronise every run from its own manifest, and the comparison files are the
registration evidence itself: a file edited for presentation is no longer the
record it is offered as. `comparisons/CHECKSUMS.sha256` lets you confirm the
copies here are the committed ones.

The identifiers carry no information beyond their own numbering; everything they
point at that a reader needs is stated where it is used.

## How to cite

The archive: [10.5281/zenodo.22732984](https://doi.org/10.5281/zenodo.22732984)

The paper: see `CITATION.cff`, and the manuscript once published.

## License

MIT, in `LICENSE`. It covers the code, the configuration and the aggregate
result files. It does not cover the PhysioNet datasets, which are not here and
are governed by their own agreement.
