#!/bin/bash
# Rebuilds the LaTeX table rows, the Holm correction and the data figures from the committed JSON files.  No GPU and no datasets.
set -euo pipefail
cd "$(dirname "$0")/.."; export EMOZOOM_ROOT="$PWD"; PY="${PYTHON:-python}"; mkdir -p results_rebuilt
"$PY" experiments/paper2_tables_newmain_20261029.py > results_rebuilt/main_tables.tex
for n in diffs ctxvar gctctrl buckets dl matchhist bases noise overlap structmiss fb harm seeds fairhist stationary perrate_war perrate_uar perrate_mf1; do
  "$PY" experiments/paper2_supp_tables_newmain_20261029.py "$n" > "results_rebuilt/supp_$n.tex"
done
"$PY" experiments/paper2_newmain_holm_20261107.py > results_rebuilt/holm.txt
"$PY" experiments/paper2_newmain_fig3_20261029.py > /dev/null
"$PY" experiments/paper2_newmain_fig5_plot_20261029.py > /dev/null
echo "done: results_rebuilt/ and paper/figures/newmain/"
