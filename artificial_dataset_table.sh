#!/bin/bash
python calibration_pipeline.py --epochs 1000 --ratio 0.2
python calibration_pipeline.py --epochs 1000 --ratio 0.6
python calibration_pipeline.py --epochs 1000 --ratio 1.0