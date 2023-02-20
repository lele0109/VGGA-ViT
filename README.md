# VGGA-ViT<br />
This repo holds the code of VGGA-ViT: A VGG attention vision transformer network for benign and malignant classification of breast ultrasound images.<br />
## Requirements<br />
* Pytorch==1.6.0(>=1.6.0 and <1.9.0 should work but not tested)<br />
****
## Model Overview<br />
![image](https://github.com/lele0109/VGGA-ViT/blob/master/pic.png)
****
## Training<br />
1. Prepare dataset<br />
2. Download pre-training weight file from the website "https://download.pytorch.org/models/vgg16-397923af.pth"<br />
3. Change the file path in the code<br />
4. Running training code
```python
CUDA_VISIBLE_DEVICES=0 python train.py --model vggvit --num_workers 0 --dataset ruxian --epochs 100
```
****
## Citation<br />
If you use this code, please cite following paper, thanks.<br />
Qu X, Lu H, Tang W, Wang S, Zheng D, Hou Y, Jiang J. A VGG attention vision transformer network for benign and malignant classification of breast ultrasound images. Med Phys. 2022 Sep;49(9):5787-5798. doi: 10.1002/mp.15852. Epub 2022 Jul 30. PMID: 35866492.<br />
****
## Questions<br />
Please drop an email to zjl000109@163.com
