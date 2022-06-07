# Multi-Scale Feature Integration For Chinese Scene Text Recognition
## Introduction

This is the implementation of the paper this paper.
This code is based on the [aster.pytorch](https://github.com/ayumiymk/aster.pytorch), we sincerely thank ayumiymk for his awesome repo and help.



## Environment

```
PyTorch == 1.1.0
torchvision == 0.3.0
```

Details can be found in requirements.txt



## Datasets

![dataset.png](https://s2.loli.net/2022/06/07/yUS5kzT4jbFIxJu.png)

We use the benchmarking chinese scene text dataset [1]. The *lmdb* scene, web and document datasets are available in [BaiduCloud](https://pan.baidu.com/s/1OlAAvSOUl8mA2WBzRC8RCg) (psw:v2rm) and [GoogleDrive](https://drive.google.com/drive/folders/1J-3klWJasVJTL32FOKaFXZykKwN6Wni5?usp=sharing).

[1] first collect the publicly available scene datasets including **RCTW**, **ReCTS**, **LSVT**, **ArT**, **CTW** resulting in 636,455 samples, which are randomly shuffled and then divided at a ratio of 8:1:1 to construct the training, validation, and testing datasets. Details of each scene datasets are introduced as follows:

- **RCTW** [2] provides 12,263 annotated Chinese text images from natural scenes. We derive 44,420 text lines from the training set and use them in our benchmark. The testing set of RCTW is not used as the text labels are not available. 
- **ReCTS** [3] provides 25,000 annotated street-view Chinese text images, mainly derived from natural signboards. We only adopt the training set and crop 107,657 text samples in total for our benchmark. 
- **LSVT** [4] is a large scale Chinese and English scene text dataset, providing 50,000 full-labeled (polygon boxes and text labels) and 400,000 partial-labeled (only one text instance each image) samples. We only utilize the full-labeled training set and crop 243,063 text line images for our benchmark.
- **ArT** [5] contains text samples captured in natural scenes with various text layouts (e.g., rotated text and curved texts). Here we obtain 49,951 cropped text images from the training set, and use them in our benchmark.
- **CTW** [6] contains annotated 30,000 street view images with rich diversity including planar, raised, and poorly-illuminated text images. Also, it provides not only character boxes and labels, but also character attributes like background complexity, appearance, etc. Here we crop 191,364 text lines from both the training and testing sets.

[1] combine all the subdatasets, resulting in 636,455 text samples. We randomly shuffle these samples and split them at a ratio of 8:1:1, leading to 509,164 samples for training, 63,645 samples for validation, and 63,646 samples for testing. 

### Train

- Update the path in train.sh, then

```
sh train.sh
```

### Test

- Update the path in the test.sh, then

```
sh test.sh
```



## Experiments

The results of our method on the chinese scene datasets. ACC / NED follow the percentage format and decimal format, respectively. Please click the hyperlinks to see the detailed experimental results, following the format of (*index* *[pred]* *[gt]*).

|          |   ACC    |    NED    | Parameters |
| :------: | :------: | :-------: | :--------: |
| CRNN[7]  |   53.4   |   0.734   |   12.4M    |
| ASTER[8] |   54.5   |   0.695   |   27.2M    |
| MORAN[9] |   51.8   |   0.686   |   28.5M    |
| SRN[10]  | **60.1** |   0.778   |   64.3M    |
| SEED[11] |   49.6   |   0.661   |   73.5M    |
|   Ours   |   58.7   | **0.796** |   46.54M   |

Ablation study of the proposed structure. “SE” stands for Squeeze-and-Excitation Networks. “ASPP” means the DenseASPP model.

|  Method  |   ACC    |    NED    | Parameters |
| :------: | :------: | :-------: | :--------: |
| Baseline |   53.2   |   0.761   |   29.06M   |
|   +SE    |   54.8   |   0.773   |   29.22M   |
|  +ASPP   |   56.4   |   0.784   |   46.38M   |
|   Ours   | **58.7** | **0.796** |   46.54M   |

## References

### Datasets

The dataset

Details of each scene datasets are introduced as follows:

[1] Chen, Jingye, et al. "Benchmarking Chinese Text Recognition: Datasets, Baselines, and an Empirical Study." *arXiv preprint arXiv:2112.15093* (2021).

[2] Shi B, Yao C, Liao M, et al. ICDAR2017 competition on reading chinese text in the wild (RCTW-17). ICDAR, 2017. 

[3] Zhang R, Zhou Y, Jiang Q, et al. Icdar 2019 robust reading challenge on reading chinese text on signboard. ICDAR, 2019. 

[4] Sun Y, Ni Z, Chng C K, et al. ICDAR 2019 competition on large-scale street view text with partial labeling-RRC-LSVT. ICDAR, 2019. 

[5] Chng C K, Liu Y, Sun Y, et al. ICDAR2019 robust reading challenge on arbitrary-shaped text-RRC-ArT. ICDAR, 2019. 

[6] Yuan T L, Zhu Z, Xu K, et al. A large chinese text dataset in the wild. Journal of Computer Science and Technology, 2019.

### Methods

[7] Shi B, Bai X, Yao C. An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition. TPAMI, 2016.

[8] Shi B, Yang M, Wang X, et al. Aster: An attentional scene text recognizer with flexible rectification. TPAMI, 2018.

[9] Luo C, Jin L, Sun Z. Moran: A multi-object rectified attention network for scene text recognition. PR, 2019.

[10] Yu D, Li X, Zhang C, et al. Towards accurate scene text recognition with semantic reasoning networks. CVPR, 2020.

[11] Qiao Z, Zhou Y, Yang D, et al. Seed: Semantics enhanced encoder-decoder framework for scene text recognition. CVPR, 2020.





## Citation





## Acknowledgements

We sincerely thank those researchers who collect the subdatasets for Chinese text recognition. Specially, we would like to thank Chen, Jingye for the proposed  benchmark. 

