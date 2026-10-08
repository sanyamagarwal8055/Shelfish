# ShelfPulse interface contract (v1.0)

This file is the **only** thing the Vision track and the Brain track share. Each side can build and test alone as long as both obey it.

**Changing it:** open a PR labelled `contract`, bump `CONTRACT_VERSION` in `shelfpulse/contracts.py`, and get the other person's approval before merging. Never change it silently.

```
 Vision track (Person A)                              Brain track (Person B)
 camera / robot images ──► perception ──► BayReading ──► matcher → shelf state → tasks
                                   ▲                                   │
           robot bridge ◄──────────┼────────── Mission ◄─── robot planner
                                   │
                    reads data/label_maps/*.json  ◄── written by Brain
```

## 1. Shared, read-only data (both read, owner writes)

| File | Owner | Purpose |
|---|---|---|
| `configs/store_layout.yaml` | both (contract PR) | Runs, faces, bays, aisle positions |
| `data/sku_master.csv` | both (contract PR) | `sku_id,name,category,width_cm,height_cm,depth_cm,margin_inr,pack_keywords,home_bay` |
| `data/planograms/<bay_id>.json` | Brain | Digital planogram, if the store has one |
| `data/label_maps/<bay_id>.json` | Brain | Planogram rebuilt from robot label reads; same format as planograms |
| `data/gallery/<sku_id>/*.jpg` | Vision | Pack-shot photos |

## 2. IDs and coordinates

- **bay_id** = `<run>-<face>-<NN>`: run `G1`..`G10`, face `L` or `R`, bay `01`..`10` counted from the front (checkout side). End caps: `<run>-E-F` (front) and `<run>-E-B` (back). Example: `G1-L-04`.
- **row** = shelf level, `0` = base deck (bottom) to `5` = top.
- **x_cm** = distance from the bay's left edge as seen from the aisle, in cm (0 to 120). Positions are the **left edge** of a pack, gap or label.
- **t** = ISO 8601 with timezone, e.g. `2026-10-08T11:20:00+05:30`.

## 3. `BayReading` (Vision → Brain)

One JSON object per analysed bay image. Written one per line to `runs/<run_id>/bay_readings.jsonl` (phase 1) or published on the `bay_readings` stream (phase 2).

```json
{
  "contract_version": "1.0",
  "bay_id": "G1-L-04",
  "source": "camera",
  "t": "2026-10-08T11:20:00+05:30",
  "frame_ref": "frames/G1-L-04/2026-10-08T11-20-00.jpg",
  "quality": 0.92,
  "px_per_cm": 10.0,
  "rows": [
    {
      "row": 3,
      "occluded": [[80.0, 120.0]],
      "packs": [
        {"sku": "RICE_1KG", "conf": 0.91, "x_cm": 34.0, "w_cm": 9.5, "h_cm": 26.0, "stack": 1, "depth_left": 3},
        {"sku": "AMBIGUOUS:DAL_1KG|DAL_500G", "conf": 0.55, "x_cm": 61.0, "w_cm": 9.0, "h_cm": 22.0, "stack": 1, "depth_left": null}
      ],
      "gaps": [{"x_cm": 43.5, "w_cm": 17.5}],
      "labels": []
    }
  ]
}
```

| Field | Rule |
|---|---|
| `source` | `"camera"` or `"robot"` |
| `quality` | 0-1. Below 0.5 means "don't trust this frame"; Brain treats the bay as unseen |
| `rows` | Every visible row, even empty ones. A row not listed = not seen |
| `occluded` | x ranges (cm) hidden by people or trolleys. Brain never raises alerts there |
| `packs[].sku` | A real `sku_id`, `"AMBIGUOUS:<a>|<b>"`, or `"UNKNOWN"` (not in gallery) |
| `packs[].stack` | Packs piled vertically in that facing (1 if not stacked) |
| `packs[].depth_left` | Packs remaining behind and including the front one; `null` if depth unknown |
| `gaps` | Visible empty shelf stretches wider than 5 cm |
| `labels` | Robot images only: `{"x_cm": 34.0, "sku": "RICE_1KG", "price": 89.0}`. Always `[]` from cameras |

## 4. `Mission` (Brain → Vision's robot bridge)

```json
{"contract_version": "1.0", "mission_id": "M-0007", "kind": "mission",
 "created_at": "2026-10-08T11:35:00+05:30",
 "bays": ["G3-L-05", "G7-R-02", "G9-E-F"],
 "reasons": {"G3-L-05": "blocked", "G7-R-02": "verify", "G9-E-F": "promo"},
 "speed_mps": 0.3}
```

- `kind`: `"sweep"` (all bays, fixed route, 0.4 m/s) or `"mission"` (listed bays in the given order).
- Brain decides **which bays and in what order**; the robot bridge turns bays into waypoints.
- Phase 1: `runs/<run_id>/missions.jsonl`. Phase 2: `missions` stream.

## 5. `RobotStatus` (Vision's robot bridge → Brain)

```json
{"contract_version": "1.0", "t": "2026-10-08T11:41:10+05:30", "state": "running",
 "x_m": 14.6, "y_m": 16.0, "mission_id": "M-0007",
 "done_bays": ["G7-R-02"], "skipped_bays": [], "pending_bays": ["G3-L-05", "G9-E-F"]}
```

`state`: `idle`, `docked`, `running`, `blocked`. A bay in `skipped_bays` was blocked by people and should be re-queued.

## 6. Planogram / label map JSON (Brain writes, Vision reads)

```json
{"bay_id": "G1-L-04", "source": "label_map", "updated": "2026-10-08T07:24:00+05:30",
 "rows": [[{"position": 0, "sku_id": "RICE_1KG", "x_start_cm": 0, "x_end_cm": 30, "facings": 3, "min_facings": 1}]]}
```

`rows[i]` is shelf row `i`. Vision uses it only as a hint ("which SKU should be at this x").

## 7. Fixtures (committed, small)

| Path | Made by | Used by |
|---|---|---|
| `contracts/fixtures/demo_store/` | both, step 0 | store_layout, sku_master, planograms for the demo bays |
| `contracts/fixtures/bay_readings/*.jsonl` | Vision (real output, refreshed weekly) | Brain's integration tests |
| `contracts/fixtures/scenarios/*.jsonl` | Brain (hand-written) | Brain's unit tests |
| `contracts/fixtures/missions/*.jsonl` | Brain | Vision's robot-bridge tests |

Both sides run `pytest tests/contract` before every PR: it validates every fixture against `shelfpulse/contracts.py`.
