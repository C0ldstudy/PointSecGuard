The project is implemented from the paper "On Adversarial Robustness of Point Cloud Semantic Segmentation".

## Start
Downloading S3DIS from `https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip` by python is not available due to the expired certificate. So either try to use `wget https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip --no-check-certificate` or download it manually and then move the files to `examples/sem_seg_dense/data/deepgcn/S3DIS`.

```
cd examples/sem_seg_dense
python test.py --attack <attack method>
```

Attack methods can be changed by chaning the attack method.

## Requirements
* [Pytorch>=1.4.0](https://pytorch.org)
* [pytorch_geometric>=1.3.0](https://pytorch-geometric.readthedocs.io/en/latest/)

Currently, pytorch_geometric does not provide torch 1.7.1. But we can use torch 1.7.0 version of pytorch_geometric. Available versions of pytorch_geometric can be checked in the [link](https://pytorch-geometric.com/whl/)

## Code Architecture
    .
    ├── utils                   # Common useful modules
    ├── gcn_lib                 # gcn library
    │   ├── dense               # gcn library for dense data (B x C x N x 1)
    │   └── sparse              # gcn library for sparse data (N x C)
    ├── sem_seg_dense           # code for point clouds semantic segmentation on S3DIS

Great thanks for the [Deep GCNs project](https://github.com/lightaime/deep_gcns_torch).
The attack code is implemented based on the [adversarial-attacks-pytorch](https://github.com/Harry24k/adversarial-attacks-pytorch).
