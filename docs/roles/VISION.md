# Role: Vision track (Person A)

You build everything from **pixels to `BayReading`**, plus the robot bridge. You never need the Brain's code to work or test: your output is a file of `BayReading` JSON lines that must match `docs/CONTRACT.md`.

## What you own
- `shelfpulse/sources/camera/` — camera service: grab frame (colour + depth), lens correction, crop/warp each camera's bays to a front-on image at 10 px/cm.
- `shelfpulse/sources/robot/` — robot bridge: read `Mission`s, drive a `FakeVendorAPI` (replays a walk-along video + `pose.csv`), stitch frames into one image per bay, write `RobotStatus`, reject frames where a person covers > 30% of a bay.
- `shelfpulse/perception/` — privacy blur, quality score, rectify, detector, embedder, sku_index, identify, depth, labels, and **`analyze(bay_image) -> BayReading`** (the single entry point the Brain will call).
- `shelfpulse/layout/camera_planner.py` + `scripts/plan_cameras.py` — camera count and positions from the layout.
- `tools/synth/` — synthetic shelf image generator with perfect ground truth.
- `training/`, `tests/vision/`.

## Build order (finish and test each before the next)
1. **Contract plumbing:** `perception/analyze.py` returning a hard-coded valid `BayReading`; a CLI `python -m shelfpulse.perception.run --input <images|video> --out runs/<id>/bay_readings.jsonl`. Contract tests pass.
2. **Synthetic shelf generator** (`tools/synth/`): paste gallery pack-shots onto a shelf background at known rows and x positions; options to remove packs (gaps), misplace one, add an occluding blob, add wide-lens distortion. Writes the image **and** the true `BayReading`. This is your main test bed.
3. **Camera planner:** formula N = ceil((L - 0.2) / (W - 0.2)), W = min(2d·tan45°, 2d·tan35°, 4000 px / 10 px per cm). Test: 2.0 m aisle, 12 m run → 5 cameras at 1.2, 3.6, 6.0, 8.4, 10.8 m; 1.5 m aisle → 7.
4. **Rectify + detect:** pretrained YOLO (or a SKU-110K-trained checkpoint) for packs; gaps from shelf-row regions with no packs; rows from shelf rails. Score on synthetic images, then on SKU-110K test images.
5. **Identify:** DINOv2 (or CLIP) embeddings + FAISS gallery; re-rank by location hint (`data/label_maps/` or `data/planograms/`), size in cm vs `sku_master.csv`, and OCR text. Output real SKU, `AMBIGUOUS:a|b` or `UNKNOWN`.
6. **Depth:** from the stereo depth map, recess of each front pack → `depth_left`; `null` if no depth.
7. **Robot bridge** with `FakeVendorAPI` + stitcher + label reading (`labels` filled only for robot images).
8. **Camera service** loop on a recorded fixed-view video (1 frame per minute, sped up).

## How you test (no Brain needed)
| Test bed | What it checks | Target to report |
|---|---|---|
| Synthetic shelves (`tools/synth`) | Exact facings, gaps, misplaced, occlusion vs ground truth | Facing count error, SKU accuracy, gap recall |
| SKU-110K test split | Pack detection in dense real shelves | mAP@0.5 |
| Team demo shelf: tripod video (camera) + walk-along video (robot) + 10 hand-labelled frames in `tests/vision/gold/` | Real-world end to end | Per-slot status accuracy vs gold |
| `contracts/fixtures/missions/*.jsonl` | Robot bridge follows missions, reports skipped bays | All bays done or skipped, statuses valid |
| `tests/contract` | Every output validates against `contracts.py` | 100% |

`python tools/eval_readings.py --pred runs/<id>/bay_readings.jsonl --gold <gold.jsonl>` compares predictions with ground truth; write it early.

## Hand-off to Brain
- Each week (or when quality improves), commit a short real output to `contracts/fixtures/bay_readings/demo_<date>.jsonl` (≤ 200 lines) so the Brain can test on it.
- Never change a field's meaning without a contract PR.

## Defaults
- Bay images: front-on, 10 px/cm, 1200 x 2100 px for a 1.2 m x 2.1 m bay.
- Quality < 0.5 → still emit the reading with that quality; Brain decides.
- Keep models swappable behind small classes (`Detector`, `Embedder`), weights path from config.
