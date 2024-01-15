#!/bin/bash
python calibration_pipeline_images.py --lr 0.00003 --epochs 100 --image_size 224 --weight_decay 0.01 --ratio 0.2
python calibration_pipeline_images.py --lr 0.00003 --epochs 100 --image_size 224 --weight_decay 0.01 --ratio 0.6
python calibration_pipeline_images.py --lr 0.00003 --epochs 100 --image_size 224 --weight_decay 0.01 --ratio 1.0
