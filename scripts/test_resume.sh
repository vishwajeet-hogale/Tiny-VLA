#!/usr/bin/env bash
# Part 5, step 4: check that --resume continues a run exactly where it stopped.
#
#   A: 400 steps in one go
#   B: 200 steps, stop, then --resume to 400
# Then compare the metrics.csv rows. If resuming restores everything (weights, optimizer, GradScaler,
# RNG, data position), B's rows after step 200 should match A's up to GPU non-determinism
# (cudnn.benchmark picks algorithms that aren't bit-exact).
#
#   bash scripts/test_resume.sh
set -euo pipefail
OUT=outputs/part5/resume_test
rm -rf "$OUT"
COMMON=(--steps 400 --log-freq 100 --save-freq 200)

echo "=== A: 400 steps straight ==="
python -m tiny_vla.train --output-dir "$OUT/A" "${COMMON[@]}"

echo "=== B: 200 steps, then resume to 400 ==="
python -m tiny_vla.train --output-dir "$OUT/B" --steps 200 --log-freq 100 --save-freq 200
python -m tiny_vla.train --output-dir "$OUT/B" "${COMMON[@]}" --resume

echo "=== Compare ==="
python - "$OUT" <<'EOF'
import csv, sys
out = sys.argv[1]
a = {r["step"]: r for r in csv.DictReader(open(f"{out}/A/metrics.csv"))}
b = {r["step"]: r for r in csv.DictReader(open(f"{out}/B/metrics.csv"))}
print(f"{'step':>5} {'A train_loss':>13} {'B train_loss':>13} {'A val_l1':>9} {'B val_l1':>9}")
worst = 0.0
for step in a:
    ra, rb = a[step], b[step]
    print(f"{step:>5} {float(ra['train_loss']):>13.5f} {float(rb['train_loss']):>13.5f} "
          f"{float(ra['val_l1']):>9.5f} {float(rb['val_l1']):>9.5f}")
    worst = max(worst, abs(float(ra["train_loss"]) - float(rb["train_loss"])) / float(ra["train_loss"]))
print(f"largest relative train-loss difference: {worst:.2%}")
print("RESUME TEST", "PASSED" if worst < 0.02 else "FAILED (>2% apart)")
EOF
