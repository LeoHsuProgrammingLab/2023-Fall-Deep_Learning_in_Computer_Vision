#!/bin/bash
# Download pretrained weights
python3 -c "import clip; clip.load('ViT-L/14')"

# Download P2_model.ckpt
gdown -O ./P2_model.ckpt https://drive.google.com/uc?id=1b6JnMoPlZVkQid4nqlrHSXy988YAbIoS