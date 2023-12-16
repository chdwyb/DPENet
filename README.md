# From Heavy Rain Removal to Detail Restoration: A Faster and Better Network

[![Paper](https://img.shields.io/badge/Paper-PR-blue)](https://www.sciencedirect.com/science/article/abs/pii/S0031320323009020)  [![Project](https://img.shields.io/badge/Project-GitHub-gold)](https://github.com/chdwyb/DPENet)

Abstract: The profound accumulation of precipitation during intense rainfall events can markedly degrade the quality of images, leading to the erosion of textural details. Despite the improvements observed in existing learning-based methods specialized for heavy rain removal, it is discerned that a significant proportion of these methods tend to overlook the precise reconstruction of the intricate details. In this work, we introduce a simple dual-stage progressive enhancement network, denoted as DPENet, aiming to achieve effective deraining while preserving the structural accuracy of rain-free images. This approach comprises two key modules, a rain streaks removal network (R$^2$Net) focusing on accurate rain removal, and a details reconstruction network (DRNet) designed to recover the textural details of rain-free images. Firstly, we introduce a dilated dense residual block (DDRB) within R$^2$Net, enabling the aggregation of high-level and low-level features. Secondly, an enhanced residual pixel-wise attention block (ERPAB) is integrated into DRNet to facilitate the incorporation of contextual information. To further enhance the fidelity of our approach, we employ a comprehensive loss function that accentuates both the marginal and regional accuracy of rain-free images. Extensive experiments conducted on publicly available benchmarks demonstrates the noteworthy efficiency and effectiveness of our proposed DPENet.

![network](./image/network.png)

## Dependencies
```python
PyTorch 1.7.1
Python 3.6.5
CUDA 10.1
```

## Train and test
DPENet is trained on Rain1400, Rain100H, Rain100L and Rain800，prepare them for training and testing. You can also use your datasets, meanwhile change the path to yours.

After that, you can train and test the DPENet by

```python
python main.py
```

## Quick test
pre-trained models can be found at ./logs, meanwhile change the path to yours. you can test the DPENet by

```python
python test.py
```

## Citation
If you find this project useful in your research, please consider citing:

```
@article{wen2023from,
title = {From heavy rain removal to detail restoration: A faster and better network},
journal = {Pattern Recognition},
pages = {110205},
year = {2023},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2023.110205},
}
```

## Contact us
Please contact us if there are any questions or suggestions (wyb@chd.edu.cn).

