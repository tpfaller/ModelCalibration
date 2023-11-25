# ModelCalibration

## Setup

```
sudo apt install python3-pip
python3 -m pip install --upgrade pip
sudo apt-get install python3-venv
```


```
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Instructions

#### Train classifier
```
python train.py \
--model mobilenet_v3_large \
--opt adamw \
--data-path data \
--output-dir output \
--workers 4 \
--epochs 10 \
--augmix-severity 0 \
--ra-reps 0 \
--ra-magnitude 0 \
--lr 0.003
```