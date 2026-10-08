# Demo store fixture

Small, committed snapshot both tracks can test against without the full store data.

- `store_layout.yaml`, `sku_master.csv`: exact copies of `configs/store_layout.yaml` and
  `data/sku_master.csv`. `tests/contract` fails if they drift; update both in the same contract PR.
- `planograms/<bay_id>.json`: planograms for the demo bays used in `docs/CONTRACT.md` examples
  (`G1-L-04`, `G1-L-05`, `G3-L-05` on camera runs; `G7-R-02` robot-only, as a label map).

SKU dimensions and margins are illustrative demo values, not real store data.
