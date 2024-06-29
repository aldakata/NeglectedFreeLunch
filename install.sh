conda create --yes --prefix ../torch python=3.9
conda activate ../torch
conda install --yes pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install --yes matplotlib
conda install --yes scipy