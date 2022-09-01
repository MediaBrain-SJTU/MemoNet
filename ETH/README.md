# MemoNet on ETH/UCY

In this repository, we provide the training/testing code for MomoNet on the ETH/UCY dataset.

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
----ETH\
    |----datasets\                      # ETH/UCY datasets 
    |----requirements.txt
    |----utils\
    |    |----utils.py
    |    |----torch.py
    |    |----config.py
    |----models\                        # trainging/testing models
    |    |----model_train_trajectory.py
    |    |----model_test_trajectory.py
    |    |----layer_utils.py
    |----trainer\
    |    |----train_trajectory_AIO.py
    |    |----test_trajectory_AIO.py
    |----test.py
    |----pretrain\
    |----README.md
    |----train.py
    |----data\                          # files for dataloader
    |    |----convert_ethucy.py
    |    |----ethucy_split.py
    |    |----preprocessor.py
    |    |----map.py
    |    |----dataloader.py
    |    |----homography_warper.py
    |----cfg\                           # configure files
    |    |----zara1.yml
    |    |----hotel.yml
    |    |----univ.yml
    |    |----eth.yml
    |    |----zara2.yml
```



## Training

Important configurations.

* `--cfg`: configure file to load,
* `--info`: path name to store the models,
* `--gpu`: number of devices to run the codes,

Training commands:

```linux
python train.py --cfg <eth/hotel/univ/zara1/zara2> --info <training info here> --gpu <gpu id here>
```


## Testing

To get the evaluation results, following

```linux
python test.py --cfg <eth/hotel/univ/zara1/zara2> --info <training info here> --gpu <gpu id here>
```


## Acknowledgement

Thanks for the framework provided by `Marchetz/MANTRA-CVPR20`, which is source code of the published work MANTRA in CVPR-2020. The github repo is [MANTRA code](https://github.com/Marchetz/MANTRA-CVPR20). We borrow the framework and interface from the code.

We also thank for the pre-processed data provided by the works of AgentFormer ([paper](https://arxiv.org/abs/2103.14023),[code](https://github.com/Khrylx/AgentFormer)).

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
