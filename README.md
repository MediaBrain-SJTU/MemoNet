# Remember Intentions: Retrospective-Memory-based Trajectory Prediction

**Official PyTorch code** for CVPR'22 paper "Remember Intentions: Retrospective-Memory-based Trajectory Prediction".

[[Paper]](https://arxiv.org/abs/2203.11474)&nbsp;[[Zhihu]](https://zhuanlan.zhihu.com/p/492362530)

![system design](./imgs/memonet.jpg)

**Abstract**: To realize trajectory prediction, most previous methods adopt the parameter-based approach, which encodes all the seen past-future instance pairs into model parameters. However, in this way, the model parameters come from all seen instances, which means a huge amount of irrelevant seen instances might also involve in predicting the current situation, disturbing the performance. To provide a more explicit link between the current situation and the seen instances, we imitate the mechanism of retrospective memory in neuropsychology and propose MemoNet, an instance-based approach that predicts the movement intentions of agents by looking for similar scenarios in the training data. In MemoNet, we design a pair of memory banks to explicitly store representative instances in the training set, acting as prefrontal cortex in the neural system, and a trainable memory addresser to adaptively search a current situation with similar instances in the memory bank, acting like basal ganglia. During prediction, MemoNet recalls previous memory by using the memory addresser to index related instances in the memory bank. We further propose a two-step trajectory prediction system, where the first step is to leverage MemoNet to predict the destination and the second step is to fulfill the whole trajectory according to the predicted destinations. Experiments show that the proposed MemoNet improves the FDE by 20.3\%/10.2\%/28.3\% from the previous best method on SDD/ETH-UCY/NBA datasets. Experiments also show that our MemoNet has the ability to trace back to specific instances during prediction, promoting more interpretability.


We give an example of trajectories predicted by our model and the corresponding ground truth as following:

![system design](./imgs/predictions.png)

Below is an example of prediction interpretability where the first column stands for the current agent. The last three columns stand for the memory instances found by the current agent.
![system design](./imgs/interpretability.png)


## [2022/09] Update: ETH's code & model is available!

You can find the code and the instructions in the **ETH** folder.

## Installation

### Environment

* Tested OS: Linux / RTX 3090
* Python == 3.7.9
* PyTorch == 1.7.1+cu110

### Dependencies

Install the dependencies from the `requirements.txt`:
```linux
pip install -r requirements.txt
```

### Pretrained Models

We provide a complete set of pre-trained models including:

* intention encoder-decoder:
* learnable addresser:
* generated memory bank:
* fulfillment encoder-decoder:

You can download the pretrained models/data from [here](https://drive.google.com/drive/folders/1qx5vbNgyM9aMH9jB_F07w3QIxzzi6StW?usp=sharing).


### File Structure

After the prepartion work, the whole project should has the following structure:

```
./MemoNet
├── ReadMe.md
├── data                            # datasets
│   ├── test_all_4096_0_100.pickle
│   └── train_all_512_0_100.pickle
├── models                          # core models
│   ├── layer_utils.py
│   ├── model_AIO.py
│   └── ...
├── requirements.txt
├── run.sh
├── sddloader.py                    # sdd dataloader
├── test_MemoNet.py                 # testing code
├── train_MemoNet.py                # training code
├── trainer                         # core operations to train the model
│   ├── evaluations.py
│   ├── test_final_trajectory.py
│   └── trainer_AIO.py
└── training                        # saved models/memory banks
    ├── saved_memory
    │   ├── sdd_social_filter_fut.pt
    │   ├── sdd_social_filter_past.pt
    │   └── sdd_social_part_traj.pt
    ├── training_ae
    │   └── model_encdec
    ├── training_selector
    │   ├── model_selector
    │   └── model_selector_warm_up
    └── training_trajectory
        └── model_encdec_trajectory
```



## Training

Important configurations.

* `--mode`: verify the current training mode, 
* `--model_ae`: pretrained model path,
* `--info`: path name to store the models,
* `--gpu`: number of devices to run the codes,

Training commands.

```linux
bash run.sh
```


## Reproduce

To get the reported results, following

```linux
python test_MemoNet.py --reproduce True --info reproduce --gpu 0
```

And the code will output: 

```linux
./training/training_trajectory/model_encdec_trajectory
Test FDE_48s: 12.659514427185059 ------ Test ADE: 8.563031196594238
----------------------------------------------------------------------------------------------------
```



## Acknowledgement

Thanks for the framework provided by `Marchetz/MANTRA-CVPR20`, which is source code of the published work MANTRA in CVPR-2020. The github repo is [MANTRA code](https://github.com/Marchetz/MANTRA-CVPR20). We borrow the framework and interface from the code.

We also thank for the pre-processed data provided by the works of PECNet ([paper](https://link.springer.com/chapter/10.1007%2F978-3-030-58536-5_45),[code](https://github.com/j2k0618/PECNet_nuScenes)).

## Citation

If you use this code, please cite our paper:

```
@InProceedings{MemoNet_2022_CVPR,
author = {Xu, Chenxin and Mao, Weibo and Zhang, Wenjun and Chen, Siheng},
title = {Remember Intentions: Retrospective-Memory-based Trajectory Prediction},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2022}
}
```
