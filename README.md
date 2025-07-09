# HotMoE


 Our Model Weight: [![ModelScope](https://img.shields.io/badge/ModelScope-HotMoE-blue?logo=amazons3&style=for-the-badge)](https://modelscope.cn/models/taryya/HotMoE/)
 
 Pretrain Model: [Google Drive(OSTrack)](https://drive.google.com/drive/folders/1nU40_fpIVUd4ht3T_qB8uSBhQeSzBTD9 )

 Raw Result:   [Google Drive](https://drive.google.com/drive/folders/1coxIFkzUhJeJphKAyCnpN5qJ2xQwoL0v?usp=sharing)
 
  :sparkling_heart: The implementation of MoE in this code is relatively simple. Please refer to my [XTrack code](https://github.com/supertyd/XTrack/tree/main), which contains a more detailed implementation.  :sparkling_heart:

![HotMoE](stream_line.gif)
## Usage
### Installation
Create and activate a conda environment, we've tested on this env:
You can follow the env setting of [OSTrack](https://github.com/botaoye/OSTrack).

### Data Preparation
Download the datasets from [official project](https://www.hsitracking.com/contest/).
```
$<PATH_of_Data>
-- validation
    -- HSI-NIR
        |-- basketball3
        ...
    -- HSI-NIR-FalseColor
        |-- basketball3
        ...
    -- HSI-RedNIR
        |-- ball&mirror9
        ...
    -- HSI-RedNIR-FalseColor
        |-- ball&mirror9
        ...
    -- HSI-VIS
        |-- ball
        ...
    -- HSI-VIS-FalseColor
        |-- ball
        ...
```

### Path Setting
Run the following command to set paths:
```
cd <PATH_of_HotMoE>
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
You can also modify paths by these two files:
```
./lib/train/admin/local.py  # paths for training
./lib/test/evaluation/local.py  # paths for testing
```

### Testing
- Download our model.

```
cd <PATH_of_HotMoE/tracking>
python test_hsi_mgpus_all.py --dataset_name HOT23VAL
```

If you want to test the real speed of our model, please run :
```
python test.py
```


## Citation
```bibtex
@article{sun2025hotmoe,
  title={HotMoE: Exploring Sparse Mixture-of-Experts for Hyperspectral Object Tracking},
  author={Sun, Wenfang and Tan, Yuedong and Li, Jingyuan and Hou, Shuwei and Li, Xiaobo and Shao, Yingzhao and Wang, Zhe and Song, Beibei},
  journal={IEEE Transactions on Multimedia},
  year={2025},
  publisher={IEEE}
}

```


