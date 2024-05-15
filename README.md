# Image Deraining via Self-supervised Reinforcement Learning
This  is official implementation of [arXiv](https://arxiv.org/abs/2403.18270) paper.
### Enviornment Requirements
1. Create a virtual environment using `conda` or `virtualenv`.
2. Install the package. (The version may be various depends on your devices.)
    ```
    pip install -r requirements.txt
    ```

### Dataset preparation
1. Download dataset, and sort by yourself like below structure. 
    ```
    .
    ├── datset                  
    │   ├── Rain100L                    
    │   │   ├── test
    │   │   │   ├── input
    │   │   │   ├── gt
    ```
2. Generate txt files which list the paths of images that you want to deal with. See `dataset/Rain100L/testing.txt` as example.

### Rain Mask Generation
This part is implemented by Matlab, which modify from the source code of [TIP 2012](https://ieeexplore.ieee.org/document/6099619).
1. Make sure the requirement packages (such as SPAMS) is installed.  
2. Modfiy `file_path` and `rain_component_path` in `rain_mask/extract_mask.m` and run it.  
3. Modfiy `src_dir` and `binary_mask_dir` in `binarization.py` and run the command
    ```
    cd rain_mask
    python binarazation.py
    ```

### Pseudo-Derained Reference $y^{pr}$ Generation
Modify `dataset_path`, `save_path`, and `target_path` in the `stochastic_filling.py` and run the command.
```
python stochastic_filling.py
```

### RL-based Self-supervised Deraining Scheme
Modify the default values of `image_dir_path`, `data_path`, and `save_dir_path` in the `main.py` and run the command.
```
cd ../
python main.py
```  
or just use command line argparser
```
cd ../
python main.py --image_dir_path './dataset/' --data_path './dataset/Rain100L/testing.txt' --save_dir_path './Results/Rain100L/test/SRL-Derain/'
```
PS: For the rainy image that is too big to derained, we will use `main_overlapped.py` instead of `main.py`. The image will be random cropped during training and overlapped inference by patches.

### Multiple training strategy
In ablation study, we also provide the derained results by using multiple training strategy, where the agents are train on training set and inference on testing set.
```
python main_multiple.py --mode 'train' --data_path './dataset/Rain100L/training.txt' --save_dir_path './Results/Rain100L/test/SRL-Derain_multiple/'
 
python main_multiple.py --mode 'test' --data_path './dataset/Rain100L/testing.txt' --save_dir_path './Results/Rain100L/test/SRL-Derain_multiple/derained_result/' --model_weight_path './Results/Rain100L/test/SRL-Derain_multiple/model_weight/last/model.npz' 

```

# Demo
The demo video is available at [google drive](https://drive.google.com/file/d/1YuELv-RMYLfuZw0ceTaVOH551ZCfWuBK/view?usp=sharing).
