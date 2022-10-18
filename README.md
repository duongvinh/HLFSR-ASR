# Light Field Super-Resolution Network Using Joint Spatio-Angular and Epipolar Information
This repository contains official pytorch implementation of "Light Field Super-Resolution Network Using Joint Spatio-Angular and Epipolar Information" submitted in IEEE Transactions on Computational Imaging 2022, by Vinh Van Duong, Thuc Nguyen Huu, Jonghoon Yim, and Byeungwoo Jeon.

## Results
We share pre-trained models of our HLFSR-ASR-C32 and HLFSR-ASR-C64, which are avaliable at https://drive.google.com/drive/u/2/folders/10mILrWAUx3XMfgLostJCI3rG-eYTJRJs

## Code
### Dependencies
* Python 3.6
* Pyorch 1.3.1 + torchvision 0.4.2 + cuda 92
* Matlab

### Dataset
Please download the dataset in the official repository of [DistgASR](https://github.com/YingqianWang/DistgASR).

### Train:
* Run **`Generate_Data_for_Training.m`** to generate training data.
* Run `train.py` to perform network training.
* Checkpoint will be saved to **`./log/`.

### Test:
* Run `Generate_Data_for_Test.m` to generate test data.
* Run `test.py` to perform network inference.
* The PSNR and SSIM values of each dataset will be saved to `./log/`.
<br><br>

## Citation
If you find this work helpful, please consider citing the following papers:<br> 
```Citation
@article{
  title={Light Field Super-Resolution Network Using Joint Spatio-Angular and Epipolar Information},
  author={Vinh Van Duong, Thuc Nguyen Huu, Jonghoon Yim, and Byeungwoo Jeon},
  journal={submitted to IEEE Transactions on Computational Imaging},
  year={2022},
  publisher={IEEE}
}
```
```Citation
@Article{DistgLF,
    author    = {Wang, Yingqian and Wang, Longguang and Wu, Gaochang and Yang, Jungang and An, Wei and Yu, Jingyi and Guo, Yulan},
    title     = {Disentangling Light Fields for Super-Resolution and Disparity Estimation},
    journal   = {IEEE TPAMI}, 
    year      = {2022},   
}


```
## Acknowledgement
Our work and implementations are inspired and based on the following project: <br> 
[DistgASR](https://github.com/YingqianWang/DistgASR)<br> 
We sincerely thank the authors for sharing their code and amazing research work!
