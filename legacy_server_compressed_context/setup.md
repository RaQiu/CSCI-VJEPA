## Setup 

Do Git Clone 
`git clone git@github.com:ppriyank/ICCV-CSCI-Person-ReID.git`

```
conda create --name pathak python=3.8
conda activate pathak

python3 -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install timm==0.9.7
python -m pip install matplotlib tensorboardX Ninja decord gdown termcolor
python -m pip install scikit-learn tabulate tensorboard lmdb yacs pandas einops 
python -m pip install albumentations h5py scipy 
python -m pip install torchcontrib

python -m pip install openmim
mim install 'mmcv==2.0.0'
mim install 'mmengine'
mim install 'mmagic'
<!-- python -m pip install  mmengine 'mmcv==2.0.0' mmagic -->
python -m pip install pytorch-msssim jpeg4py transformers==4.30.0
pip install transformers --upgrade
pip install diffusers==0.24.0
python -m pip install fastreid

python -m pip install torchinfo utilss scikit-learn grad-cam
python -m pip install seaborn scikit-learn grad-cam
python -m pip install --upgrade timm
python -m pip install grad-cam
```

