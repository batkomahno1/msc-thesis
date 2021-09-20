# MSc Thesis

## To Run
```
Create virtualenv:
virtualenv -p python3 venv

Activate venv:
source venv/bin/activate

Check venv activated properly:
pip3 -V

Install requirements:
pip3 install -r requirements.txt

Check the GPUS:
nvidia-smi OR watch -n 0.1 nvidia-smi

Run epxeriment:
time python3 run_all.py [options]
  --nb_iter <int> iterations per experiment
  --batch_size <int> default=64
  --verbose <True|False> default=False
  --test  <True|False> default=False

Deactivate and delete the environment:
deactivate
rm -rf venv
```
## Notes
```
GAN weights are not saved in the output folder!

Number of GPUs must be used can be changed in individual GAN executables
```
### Todo
```
1. document the code
2. try bigger batches
```
