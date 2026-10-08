# Role: Brain track (Person B)

You build everything from **`BayReading` to verified tasks**, plus the robot planner, API and dashboard. You never need images or models: you test on `BayReading` streams that a store simulator generates, and on fixture files the Vision track commits.

## What you own
- `shelfpulse/planogram/` — schema, loader, matcher (**already built in step 1**: adapt it to take `BayReading.rows` as input), `label_map.py` (build planograms from robot `labels`, write `data/label_maps/<bay_id>.json`).
- `shelfpulse/state/` — `slot_tracker.py` (**built**; set k=2, n=3, clear_after=2), `shelf_state.py` (fusion of camera + robot readings), `quantity.py`.
- `shelfpulse/decision/` — store_data, diagnosis, priority, tasks, verifier (**built**; keep theft-first order).
- `shelfpulse/robot_planner/` — scheduler (sweeps 07:00 / 15:00, mission gating), mission_queue, route (order bays on the aisle graph). Writes `Mission`s, reads `RobotStatus`.
- `shelfpulse/brain.py` — main loop: read `bay_readings.jsonl` (phase 1) or the stream (phase 2) → tasks.
- `sim/` — store simulator. `storage/`, `api/`, `apps/` (dashboard + staff page). `tests/brain/`.

## Build order
1. **Contract plumbing:** `brain.py --readings <file.jsonl> --out runs/<id>/` reads `BayReading`s and writes `events.jsonl`, `tasks.jsonl`, `missions.jsonl`. Contract tests pass.
2. **Adapter:** turn `BayReading.rows` into the matcher's input (packs with sku and x, gaps, occluded ranges). Existing step-1 tests must still pass.
3. **Store simulator (`sim/`)** — your main test bed. A clock-driven model of a few bays: true shelf contents, POS sales that deplete shelves, backroom stock, staff restocks after a delay, occlusions, misplacements, theft. It emits `BayReading`s exactly as a perfect (or noisy) Vision track would: cameras every 60 s for G1-G5 bays, robot readings at sweep times and when a `Mission` arrives. Also writes `pos.csv` and `inventory.csv`.
4. **Shelf state + quantity:** newest confident reading wins; camera vs robot disagreement within 300 s → slot "unsure" + mission; robot `depth_left` resets the estimate; between scans, last count minus POS sales, clamped to visible facings.
5. **Robot planner:** triggers (blocked 15+ min, ambiguous SKU, verify off-camera restock, promo), gating (max 2/hour, max 8 bays, no 18:00-21:00, low POS traffic), route ordering on the aisle graph from `store_map`.
6. **Label map:** build from robot `labels`; drift report vs digital planogram.
7. **API + dashboard + staff page.**

## Scenarios (each is a `sim/scenarios/<name>.yaml` + a test asserting the outcome)
| Scenario | Expected result |
|---|---|
| `restock_simple` | Rice slot empties; backroom has stock → one P1 RESTOCK task; restock → task VERIFIED with time-to-restore |
| `true_stockout` | No stock anywhere → manager re-order task, no restock task |
| `phantom_stock` | System says 18, shelf empty, no sales → CYCLE_COUNT task |
| `sweep_theft` | 8 packs vanish in 2 min with 0 sales → silent loss-prevention note, checked before restock |
| `misplaced` | Shampoo seen in a staples slot → RETURN task to its home bay |
| `occlusion_then_robot` | Camera bay occluded 15+ min → mission sent; robot reading resolves it |
| `noisy_frame` | One bad frame → no alert |
| `conflict` | Camera and robot disagree within 5 min → slot unsure, no alert, re-check |
| `facing_up` | Front looks full, sales say 2 left → LOW from the estimate |
| `peak_hours` | Trigger at 19:00 → no mission until 21:00 |
| `ambiguous` | `AMBIGUOUS:` packs → no MISPLACED alert, mission queued |

`python -m sim.run --scenario <name> --out runs/<id>/` then `pytest tests/brain`.

## Testing on real Vision output
- `contracts/fixtures/bay_readings/*.jsonl` (committed by Vision) must run through `brain.py` without errors; add a test per new fixture.
- Commit `contracts/fixtures/missions/*.jsonl` from your planner so Vision can test the robot bridge.

## Defaults
- Never raise alerts inside `occluded` ranges or from readings with quality < 0.5.
- `UNKNOWN` sku → `UNKNOWN_ITEM` task (enrol product), not MISPLACED.
- All times timezone-aware; the simulator's clock drives everything (no `datetime.now()` in logic).
