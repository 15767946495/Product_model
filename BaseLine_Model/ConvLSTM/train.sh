#!/bin/bash
# ConvLSTM 基线训练(DeepCropNet 9 玉米带州,验证 2021)
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
python convlstm.py
