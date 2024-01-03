#!/bin/bash
git submodule update --init --recursive
python3 ./nerf_pl/eval.py --root_dir $1 -o $2
# TODO - run your inference Python3 code