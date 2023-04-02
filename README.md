# Zero-1-to-3: Zero-shot One Image to 3D Object


##  Usage
###  Novel View Synthesis
```
conda create -n my_venv python=3.8
conda activate my_venv
cd augment_3D
pip install -r requirements.txt
git clone https://github.com/CompVis/taming-transformers.git
pip install -e taming-transformers/
git clone https://github.com/openai/CLIP.git
pip install -e CLIP/
```

Download checkpoint under `augment_3D` through following source:

```
wget https://cv.cs.columbia.edu/zero123/assets/105000.ckpt

```

Run for novel view synthesis:

```
python main.py
```

Note that this app uses around 22 GB of VRAM, so it may not be possible to run it on any GPU.

