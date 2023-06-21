## Attack on RandLA-Net

### Install the dependencies
Install ares:
```shell
cd ares/
pip install -e .
```
Install RandLA-Net requreiments:
```shell
conda create -n randlanet python=3.5
source activate randlanet
pip install -r helper_requirements.txt
sh compile_op.sh
pip install open3d-python==0.3.0
conda install cudatoolkit=9.0
conda install cudnn
pip install tensorflow-gpu==1.15
pip install scikit-learn==0.22.1
```


### S3DIS
S3DIS dataset can be found
<a href="https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1">here</a>.
Download the files named "Stanford3dDataset_v1.2_Aligned_Version.zip". Uncompress the folder and move it to
`/data/S3DIS`.

- Preparing the dataset:
```
python utils/data_prepare_s3dis.py
```
- Evaluation
```
python main_S3DIS.py --attack_type <attack_type> --attack_target <attack_target>
```



Great thanks for the [RandLA-Net](https://github.com/QingyongHu/RandLA-Net).
The attack code is implemented based on the [ares](https://github.com/thu-ml/ares/tree/contest).


