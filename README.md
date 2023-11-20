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
--model resnet18 \
--data-path data \
--output-dir output \
--workers 4 \
--epochs 1
```