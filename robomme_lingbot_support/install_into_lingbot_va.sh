#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/mnt/hwdata/wangsen/WAM/lingbot-va}"
BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${REPO_ROOT}/tools/robomme"
mkdir -p "${REPO_ROOT}/script"

cp "${BUNDLE_ROOT}/tools/robomme/"*.py "${REPO_ROOT}/tools/robomme/"
cp "${BUNDLE_ROOT}/script/robomme_"*.sh "${REPO_ROOT}/script/"
cp "${BUNDLE_ROOT}/wan_va/configs/va_robomme_cfg.py" "${REPO_ROOT}/wan_va/configs/"
cp "${BUNDLE_ROOT}/wan_va/configs/va_robomme_train_cfg.py" "${REPO_ROOT}/wan_va/configs/"
cp "${BUNDLE_ROOT}/wan_va/configs/va_robomme_i2va.py" "${REPO_ROOT}/wan_va/configs/"

cd "${REPO_ROOT}"
python tools/robomme/register_robomme_config.py

chmod +x script/robomme_prepare_dataset.sh script/robomme_train.sh
echo "Installed RoboMME LingBot-VA support into ${REPO_ROOT}"
