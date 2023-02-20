# VGGA-ViT<br />
## Citation:<br />
If you use this code, please cite following paper, thanks.<br />
Qu X, Lu H, Tang W, Wang S, Zheng D, Hou Y, Jiang J. A VGG attention vision transformer network for benign and malignant classification of breast ultrasound images. Med Phys. 2022 Sep;49(9):5787-5798. doi: 10.1002/mp.15852. Epub 2022 Jul 30. PMID: 35866492.
****
## Training:<br />
```python
CUDA_VISIBLE_DEVICES=1 python train.py --model vggvit --num_workers 0 --dataset ruxian --epochs 100
```
VGG weight file can be downloaded from the website "https://download.pytorch.org/models/vgg16-397923af.pth"<br />
The path in the code needs to be modified at runtime<br />
