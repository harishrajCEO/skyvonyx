# Dataset licence manifest

**Source of truth in code:** `DATASET_LICENCES` in `skyvonyx/datasets.py`. This file is the human-readable copy.
`tests/test_training_eval.py` fails if the two disagree on which datasets exist.

**Rule:** a trained model is marked `commercial_ok: true` only if **every** dataset it touched is
`commercial: True` **and** `verified: True`. Non-commercial, unverified, unknown, or undeclared sources all block.
This covers data terms only. The detector licence (ultralytics YOLOv8 = AGPL-3.0) is a separate question.

Not legal advice. For anything going into a paid contract, have the actual licence text read by someone accountable for it.

| id | commercial | verified | terms as currently understood | action needed |
|---|---|---|---|---|
| `dota` | **No** | no | Reported academic-use only (Google Earth imagery) | Confirm on the official DOTA site. Do not use for a sold model. |
| `xview` | **No** | no | Reported CC BY-NC-SA 4.0 | Confirm. Do not use for a sold model. |
| `hrsid` | unknown | no | **Unverified** | Read the release terms (repo/paper) before training a pilot model on it |
| `ssdd` | unknown | no | **Unverified** - several redistributions exist with different terms | Record which copy you downloaded and its terms |
| `ls-ssdd-v1` | unknown | no | Unverified | Sentinel-1 based; check terms - a better resolution match for an S1 pilot than HRSID |
| `xview3-sar` | unknown | no | Unverified | Sentinel-1 GRD based; check terms |
| `sardet-100k` | unknown | no | Unverified - aggregates other datasets | Inherits the strictest terms of its parts |
| `sen1floods11` | unknown | no | Unverified | For the flood-extent work, not ship detection |
| `sentinel-1` | Yes | yes | Copernicus Sentinel data: free, full and open, commercial use allowed | Attribute: "Contains modified Copernicus Sentinel data [year]" |
| `sentinel-2` | Yes | yes | As Sentinel-1 | Same attribution |
| `skyvonyx-labels` | Yes | yes | Labels drawn by the Skyvonyx team | The imagery underneath needs its own row (e.g. `sentinel-1`) |
| `customer-imagery` | unknown | no | Governed by each customer contract | Default: no training reuse unless the contract says so |

## How to clear a dataset

1. Find the licence text the publisher actually ships (repo LICENSE file, download page, or paper).
2. Add below: dataset id, exact licence name, a quote of the commercial-use clause, URL, date, your name.
3. Only then edit `DATASET_LICENCES` (`commercial`, `verified`) in the same commit.

## Verification log

| date | id | licence found | commercial clause (quote) | source URL | checked by |
|---|---|---|---|---|---|
| | | | | | |

## Practical consequence for the SAR ship pilot

A model trained on HRSID or SSDD is **not commercial-ok** until those rows are verified. Two clean paths:

- **Fastest:** label your own ships on Sentinel-1 scenes (`skyvonyx-labels` + `sentinel-1`, both already clear). Then
  fine-tune on that, using HRSID/SSDD only for an internal pretraining experiment.
- **Or:** verify HRSID/SSDD terms first, then the same model can be used in a paid pilot.
