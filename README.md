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
--data-path data/debug \
--output-dir output/debug \
--workers 4 \
--epochs 10 \
--weights DEFAULT \
--lr 0.0003
```

#### Start Hypertuning
```
mlflow server --host 127.0.0.1 --port 8080
```

```
python hyperparameter_tuning.py \
--model mobilenet_v3_large \
--opt adamw \
--data-path data/debug \
--output-dir output/debug \
--workers 4 \
--epochs 5 \
--weights DEFAULT
```