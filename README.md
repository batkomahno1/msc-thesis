# MSc Thesis
This is the code and paper for my master's thesis.

## Refer to:
```
thesis.pdf
```


## Setup
```
CD into project directory!

Create virtualenv:
virtualenv -p python3 venv

Activate venv:
source venv/bin/activate

Check venv activated properly:
pip3 -V

For A100 GPUs do this:
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

Install requirements:
pip3 install -r requirements.txt

One-liner:
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html && pip3 install -r requirements.txt

Check the GPUS:
nvidia-smi OR watch -n 0.1 nvidia-smi

Deactivate and delete the environment:
deactivate
rm -rf venv
```

## Run experiments
```
python3 run_all.py --help
```

## Detect adv samples
```
python3 detect.py --help

```

## Run experiments
```
python3 run_dp.py --help
```

## Notes
```
GAN weights are not saved in the output folder!
```
