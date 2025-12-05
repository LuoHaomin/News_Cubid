#!/bin/bash
set -euo pipefail

# Timestamped logfile name
TIME=$(date "+%Y-%m-%d-%H-%M-%S")
LOGFILE="${TIME}.log"

# Environment prep
if [[ -d "./env" ]]; then
  echo "Activating venv: ./env"
  # shellcheck disable=SC1091
  source ./env/bin/activate
fi
export PYTHONPATH=.:${PYTHONPATH:-}

run_step() {
  local desc="$1"; shift
  echo "=============================="
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${desc}"
  echo "Command: $*"
  "$@"
}

# Preprocess
run_step "处理数据 (online)" python preprocess/data.py --mode online --logfile "${LOGFILE}"

# Recall
run_step "itemcf 召回 (online)" python recall/recall_itemcf.py --mode online --logfile "${LOGFILE}"
run_step "binetwork 召回 (online)" python recall/recall_binetwork.py --mode online --logfile "${LOGFILE}"
run_step "w2v 召回 (online)" python recall/recall_w2v.py --mode online --logfile "${LOGFILE}"
run_step "召回合并 (online)" python recall/recall.py --mode online --logfile "${LOGFILE}"

# Pointwise
run_step "pointwise 特征 (online)" python rank/pointwise/rank_feature.py --mode online --logfile "${LOGFILE}"
run_step "pointwise 预测 (online)" python rank/pointwise/rank_lgb.py --mode online --logfile "${LOGFILE}"

# Listwise (optional but kept for completeness; requires已训练的 listwise 模型)
run_step "listwise 特征 (online)" python rank/listwise/rank_feature_listwise.py --mode online --logfile "${LOGFILE}"
run_step "listwise 预测 (online)" python rank/listwise/rank_lambdamart.py --mode online --logfile "${LOGFILE}"

echo "完成，提交文件位于 prediction_result/"
