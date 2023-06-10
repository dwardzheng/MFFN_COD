## MFFN: Multi-view Feature Fusion Network for Camouflaged Object Detection

<p align="center">
  <img src="./img/overview.png" width="600">
</p>

This is a PyTorch implementation of the [MFFN paper](https://openaccess.thecvf.com/content/WACV2023/papers/Zheng_MFFN_Multi-View_Feature_Fusion_Network_for_Camouflaged_Object_Detection_WACV_2023_paper.pdf):

```
@InProceedings{Zheng_2023_WACV, 
	author = {Dehua Zheng and Xiaochen Zheng and Laurence Yang and Yuan Gao and Chenlu Zhu and Yiheng Ruan}, 
	title = {MFFN: Multi-view Feature Fusion Network for Camouflaged Object Detection}, 
	booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)}, 
	year = {2023}, 
}
```

### Installation

```
conda create -n MFFN python=3.8
conda activate MFFN
pip install torch==1.8.1 torchvision
git clone https://github.com/dwardzheng/MFFN_COD.git
cd MFFN_COD
pip install -r requirements.txt
```
### Training

```
python main.py --model-name=MFFN --config=configs/MFFN/MFFN_R50.py --datasets-info ./configs/_base_/dataset/dataset_configs.json --info demo
```

### Evaluation

```
./test.sh 0 
prediction result(链接：https://pan.baidu.com/s/18Bn3NFw6ES0p7eqw3AldoA 
提取码：mffn)
```

### Visualization
Visualization of camouflaged animal detection. The state-of-the-art and classic single-view COD model SINet is confused by the background sharing highly similarities with target objects and missed a lot of boundary and region shape information (indicated by orange arrows). Our multi-view scheme will eliminate these distractors and perform more efficiently and effectively.

<p align="center">
  <img src="./img/cct.png" width="300">
</p>

### License
This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.


