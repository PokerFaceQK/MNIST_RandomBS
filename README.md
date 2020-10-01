# Basic MNIST Example with random batch sizes
This is a repository forked from the official [pytorch examples](https://github.com/pytorch/examples).
This project is mainly for testing how the batch size would affect training. 
In many object detection tasks, batch size of the classification part is not fixed, unlike the number of images -- it is actually up to how many ROIs are in those images.

In this repo, the batch sizes are randomly generated from $U(0, bs_{max})$.
```bash
pip install -r requirements.txt
python main.py
# CUDA_VISIBLE_DEVICES=2 python main.py  # to specify GPU id to ex. 2
```
