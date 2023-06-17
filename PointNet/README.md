# Pytorch Implementation of the Attacks on PointNet++

## Semantic Segmentation
### Data Preparation
Download 3D indoor parsing dataset (**S3DIS**) [here](http://buildingparser.stanford.edu/dataset.html)  and save in `data/Stanford3dDataset_v1.2_Aligned_Version/`.
```
cd data_utils
python collect_indoor3d_data.py
```
Processed data will save in `data/stanford_indoor3d/`.
### Run
```
python NB_nontarget_test_semseg.py --data_path <data_path>
```

Great thanks for the [PointNet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch).
The attack code is implemented based on the [adversarial-attacks-pytorch](https://github.com/Harry24k/adversarial-attacks-pytorch).

## Reference:
[halimacc/pointnet3](https://github.com/halimacc/pointnet3)<br>
[fxia22/pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch)<br>
[charlesq34/PointNet](https://github.com/charlesq34/pointnet) <br>
[charlesq34/PointNet++](https://github.com/charlesq34/pointnet2)
