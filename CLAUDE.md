# CLAUDE.md — ShelfPulse (shared by the whole team)

ShelfPulse: shelf cameras + one robot find empty, low and misplaced products, explain why, rank the fix in rupees per hour, send a task to staff, and verify the fix. README.md is the full spec (layout, pseudocode); the internal design doc holds all numbers.

## Two tracks, two people
Two people build this in parallel, each with their own Claude Code. Your role (Vision or Brain) is set in `CLAUDE.local.md`. **If `CLAUDE.local.md` is missing, stop and ask which role you are before writing code.**

| Track | Owns (may edit) | Must not edit |
|---|---|---|
| Vision (Person A) | `shelfpulse/sources/`, `shelfpulse/perception/`, `shelfpulse/layout/camera_planner.py`, `scripts/plan_cameras.py`, `scripts/build_gallery_index.py`, `configs/perception.yaml`, `configs/cameras.yaml`, `tools/`, `training/`, `tests/vision/` | Brain folders |
| Brain (Person B) | `shelfpulse/planogram/`, `shelfpulse/state/`, `shelfpulse/decision/`, `shelfpulse/robot_planner/`, `shelfpulse/storage/`, `shelfpulse/api/`, `shelfpulse/brain.py`, `configs/brain.yaml` (tracker k/n/clear_after, priority weights, simulator settings), `apps/`, `sim/`, `tests/brain/` | Vision folders |
| Shared (contract PR only) | `shelfpulse/contracts.py`, `docs/CONTRACT.md`, `contracts/fixtures/demo_store/`, `configs/store_layout.yaml`, `data/sku_master.csv`, `shelfpulse/layout/store_map.py`, `shelfpulse/bus.py`, `shelfpulse/config.py`, `configs/robot.yaml`, `tests/contract/` | — |

The two tracks meet only through the contract below. Never import the other track's internals; import only the shared modules: `shelfpulse.contracts`, `shelfpulse.layout.store_map`, `shelfpulse.bus` and `shelfpulse.config`.

@docs/CONTRACT.md

## Locked design numbers (do not change without the team)
- Store ~15,000 sq ft, 10 runs G1-G10 x 12 m, 2.0 m aisles, 200 bays (1.2 x 2.1 m, 6 rows, 45 cm deep).
- 50 stereo PoE cameras on G1-G5 (5 per face), 1 frame per minute. 1 leased robot: sweeps 07:00 and 15:00, missions max 8 bays, max 2 per hour, none 18:00-21:00.
- Alerts: OUT on 2 of 3 frames; clear after 2 OK frames. Missions when a camera is blocked 15+ min.

## Git workflow
- Branch from `main`: `vision/<topic>` or `brain/<topic>`. Small PRs; the other person reviews.
- Before every PR: `pytest tests/contract tests/<your track>` and `ruff check .` must pass.
- Contract change = PR labelled `contract` + `CONTRACT_VERSION` bump + approval from the other person.

## Commands
- Install: `pip install -r requirements.txt`
- Tests: `pytest -q tests/contract tests/vision` or `tests/brain`
- Lint: `ruff check . && ruff format .`

## Rules for everyone
- Every tunable number lives in `configs/*.yaml`.
- Code must run on a laptop CPU; GPU optional.
- Privacy: no face recognition, no identity tracking; blur people before saving any frame.
- Never commit datasets, model weights, videos or `runs/` output (all git-ignored). Fixtures under `contracts/fixtures/` stay small (< 1 MB each).
- Exception: `data/gallery/` (pack-shot photos for product identification and the demo) is committed: each photo < 200 KB, all credited in `data/gallery/SOURCES.csv` (CC BY-SA, Open Food Facts and sister sites). Rebuild it with `python -m tools.fetch_gallery`.
- Do not invent accuracy numbers in docs; report only what the evaluation scripts measure.
- Ask before adding a heavy dependency.
