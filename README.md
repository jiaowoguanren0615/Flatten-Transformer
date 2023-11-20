# Flatten-Transformer
This is a warehouse for Flatten-Transformer-pytorch-model, The code is mainly derived from the official source code (https://github.com/LeapLabTHU/FLatten-Transformer) and modified based on it. It can now be applied to train your own image datasets.

## Vision Transformer with Focused Linear Attention ([https://arxiv.org/pdf/2311.04157.pdf](https://arxiv.org/abs/2308.00442))

## Precautions
Before you use the code to train your own data set, please first enter the ___train_gpu.py___ file and modify the ___data_root___, ___batch_size___ and ___nb_classes___ parameters. If you want to draw the confusion matrix and ROC curve, you only need to remove the comments of ___Plot_ROC___ and ___Predictor___ at the end of the code. The comment of the function is enough, and the third parameter can be changed to the path of your own model weights file(.pth).

## Train this model
### train model with single-machine single-card：
```
python train_gpu.py
```

### train model with single-machine multi-card：
```
python -m torch.distributed.launch --nproc_per_node=8 train_gpu.py
```

### train model with single-machine multi-card: 
(using a specified part of the cards: for example, I want to use the second and fourth cards)
```
CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.launch --nproc_per_node=2 train_gpu.py
```

### train model with multi-machine multi-card:
(For the specific number of GPUs on each machine, modify the value of --nproc_per_node. If you want to specify a certain card, just add CUDA_VISIBLE_DEVICES= to specify the index number of the card before each command. The principle is the same as single-machine multi-card training)
```
On the first machine: python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=<Master node IP address> --master_port=<Master node port number> train_gpu.py

On the second machine: python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr=<Master node IP address> --master_port=<Master node port number> train_gpu.py
```

## Citation
```
@InProceedings{han2023flatten,
  title={FLatten Transformer: Vision Transformer using Focused Linear Attention},
  author={Han, Dongchen and Pan, Xuran and Han, Yizeng and Song, Shiji and Huang, Gao},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023}
}
```
