conda create --yes --prefix ../torch python=3.9
conda activate ../torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install scipy
pip install -U ipykernel
pip install matplotlib
pip install tqdm