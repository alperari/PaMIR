### Install Conda
cd ~
curl -L -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash miniconda.sh -b -p "$HOME/miniconda3"

"$HOME/miniconda3/bin/conda" init bash
source ~/.bashrc
conda --version


### Create Python 3.8 env
conda create -n pamir-py38 python=3.8
conda activate pamir-py38

#### Install torch
conda install -y pytorch==1.7.0 torchvision==0.8.1 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch

#### Install other dependencies (without opendr)
pip install -r requirements.txt

### Sanity check
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import cv2; print(cv2.__version__)"
python -c "import trimesh; print('trimesh ok')"