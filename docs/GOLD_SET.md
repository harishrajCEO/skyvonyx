# Building the Sentinel-1 ship gold set

The gold set is the only thing that turns "accuracy NOT MEASURED" into a number you may say to a customer.
It is 20 Sentinel-1 scenes that **no model has ever trained on**, labelled by one person and checked by a second.

Time budget (estimate - measure your own): ~30-60 min per scene to label, ~15 min per scene to check.

## 1. Choose the 20 scenes

| Rule | Why |
|---|---|
| Same product the pilot will use: **Sentinel-1 IW GRD** (or the sigma0 product you actually process), VV polarisation | The number only applies to the imagery type it was measured on |
| Spread them: open sea, busy port, coastline, rough sea state, at least 2 dates per area | One calm-sea scene makes any detector look excellent |
| Include **3-4 scenes with few or no ships** | False alarms on empty sea are what a customer notices first |
| Record for each: product ID, acquisition date, orbit direction, area name | Goes into the `--note` and the report |
| **Never** reuse a scene (or an overlapping scene from the same pass) that is in any training set | Otherwise the test measures memory, not skill |

Keep a sheet `gold_scenes.csv`: `scene_id, product_id, date, area, sea_state_note, expected_density`.

Crop each scene to the area of interest if it is huge, but keep it a **GeoTIFF with its CRS** - the pipeline and the
scoring both rely on it. Store under `MyDrive/skyvonyx/gold/s1_ships_v1/images/`. No customer imagery here unless the
contract allows it.

## 2. Label in CVAT

1. Create a CVAT project `s1_ships_gold_v1` with **one label: `ship`**.
2. Upload the scenes. Contrast matters on SAR: use CVAT's brightness/contrast controls so dim ships are visible.
3. Labeller A draws a **tight horizontal box** around every ship, including small ones.
4. Written rules - decide these before anyone labels, put them in the CVAT project description:
   - Minimum size labelled: e.g. anything **>= 3x3 pixels** that is clearly a vessel.
   - Ships moored side-by-side: one box each if separable, one box if not (write which).
   - Azimuth ambiguities / sidelobe crosses: box the ship only, not the bright cross arms.
   - Platforms, wind turbines, buoys: **not** ships. (Add a second label later if a customer wants them.)
   - If unsure: tag the box with attribute `uncertain=true` rather than skipping it.
5. Export: **COCO 1.0** -> `gold_A.json`.

## 3. Second reviewer

Two numbers come out of this step, and both go in the report.

1. **Independent agreement** (do this on at least 5 scenes): labeller B labels the same scenes **without seeing A's
   boxes**, exports `gold_B.json`, then:
   ```
   python -m skyvonyx.evaluate --pred gold_B.json --gold gold_A.json --conf 0 \
       --out agreement.json --note "Reviewer B vs reviewer A, 5 S1 scenes"
   ```
   The resulting F1 is your **reviewer agreement**. If two humans agree only 85%, a model claiming 95% is not believable.
2. **Adjudication:** B then reviews A's boxes on all 20 scenes, fixes misses and mistakes, and A and B resolve the
   disagreements together. Export the final file as `gold_final.json`. This is the gold set.

## 4. Run the pipeline on the gold scenes

Through the queue, exactly as a customer job would run (Colab worker notebook):

```python
create_job(ROOT, "gold_s1_ships_v1", sorted(Path(".../gold/s1_ships_v1/images").glob("*.tif")),
           detector="yolov8", detector_weights=str(ROOT / "models/best.pt"),
           segmenter="none", sensor="sar", tile=1024, overlap=0.2, conf=0.01)
```

`conf=0.01` keeps low-scoring boxes so AP can be computed; the precision/recall operating point is chosen at scoring.
`sensor` and `tile` must match what the model was trained with (see its `model_card.json`).

## 5. Score

```
python -m skyvonyx.evaluate \
    --pred  jobs/review/gold_s1_ships_v1/output \
    --gold  gold_final.json \
    --iou 0.5 --conf 0.25 \
    --out   gold_s1_ships_v1_accuracy.json \
    --note  "<the test-set description below>"
```

Outputs `gold_s1_ships_v1_accuracy.json` and `.md`. Try a few `--conf` values (0.15, 0.25, 0.4) and pick the operating
point the customer cares about (missed ships vs false alarms); report the one you will actually deliver at.

## 6. The `--note` template

Fill every field. A number without this is not quotable.

```
Sentinel-1 IW GRD VV, 10 m pixel spacing; 20 scenes (<n> with ships, <n> empty sea);
areas: <e.g. Mumbai/JNPT port, Gulf of Kutch, open Arabian Sea>; acquired <YYYY-MM> to <YYYY-MM>,
<asc/desc> passes; <N> ships labelled; min labelled size 3x3 px;
labelled by <A>, adjudicated by <B>; reviewer agreement F1 <x.xx> on 5 scenes;
model <run name> (weights sha <16 chars>), trained on <dataset ids>, none of these scenes in training;
IoU 0.5, confidence <0.xx>, scored after SAHI slicing and re-stitch.
```

## 7. What you may then say

> "On 20 held-out Sentinel-1 IW scenes of <areas>, our pipeline found <recall>% of <N> ships with <precision>%
> precision (IoU 0.5), against labels checked by two reviewers who agreed <F1> with each other."

Not "XX% accurate". Not without the description. Not for a different sensor, resolution or region.
Rerun the whole scoring whenever the model, tile size, sensor setting or merge threshold changes.
