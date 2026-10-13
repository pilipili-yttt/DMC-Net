# DMC-Net


Implementation of **Dynamic Mask Optimization with Regional Regularization for RGB-Guided Depth Completion**.

## Training

Run commands from the project directory with a CUDA-enabled PyTorch environment. Set the dataset paths and training options in `config.yml`.

```bash
python main.py -m train -c 0
```

For multi-GPU training, use `-c 0,1`.

## Testing

Set the dataset paths in the checkpoint directory's `config.yml`, and set `train.load.model` to the checkpoint filename.

```bash
python main.py -m test -c 0 -p /path/to/checkpoint_directory
```


**Pretrained weights will be released after the paper is accepted.**
