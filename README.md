## Installation

```bash
conda env create -f environment.yml
conda activate mimicry
```

## Getting started

1. Download pre-trained networks `*.pkl` files from [Huggingface](https://huggingface.co/awafa/cSG2) and put them under `./checkpoints`.

2. Select one of the following datasets and go to the corresponding directory:
  - MNIST: `./mnist`
  - FashionMNIST: `./f-mnist`
  <!-- - SVHN: `./svhn` -->
  <!-- - CIFAR10: `./cifar10` -->

3. Adjust the default config in `./<DATASET>/config.py`.

4. Run the file `./<DATASET>/search.py` to generate frontier pairs.



## Acknowledgement

This code is developed based on [StyleGAN3](https://github.com/NVlabs/stylegan3) and [PTI](https://github.com/tianhaoxie/DragGAN_PTI/tree/27a9821085ce4d9b788aaf4bbb52b9b982b25bcd?tab=readme-ov-file)