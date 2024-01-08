#!/bin/bash
python calibration_pipeline_images.py --lr 0.0003 --epochs 100 --image_size 224 --weight_decay 0.1
python calibration_pipeline_images.py --lr 0.0001 --epochs 100 --image_size 224 --weight_decay 0.1
python calibration_pipeline_images.py --lr 0.00003 --epochs 100 --image_size 224 --weight_decay 0.1
python calibration_pipeline_images.py --lr 0.00001 --epochs 100 --image_size 224 --weight_decay 0.1
python calibration_pipeline_images.py --lr 0.0001 --epochs 100 --image_size 224 --weight_decay 0.3
python calibration_pipeline_images.py --lr 0.00003 --epochs 100 --image_size 224 --weight_decay 0.3
python calibration_pipeline_images.py --lr 0.0001 --epochs 100 --image_size 224 --weight_decay 0.01
python calibration_pipeline_images.py --lr 0.00003 --epochs 100 --image_size 224 --weight_decay 0.01
python calibration_pipeline_images.py --lr 0.0001 --epochs 100 --image_size 224 --weight_decay 0.1 --freeze_backbone
python calibration_pipeline_images.py --lr 0.00003 --epochs 100 --image_size 224 --weight_decay 0.1 --freeze_backbone
