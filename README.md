# VGGA-ViT<br />
## Citation:<br />
Please consider citing us if you find this work helpful:<br />
  ```
@article{Qu2022vgga,
    author    = {Xiaolei Qu and Hongyan Lu and Wenzhong Tang and Jue Jiang},
    title     = {A VGG attention vision transformer network for benign and malignant classification of breast ultrasound images},
    journal   = {Med Phys doi:10.1002/mp.15852},
    year      = {2022}
}
```
****
## Training:<br />
```python
CUDA_VISIBLE_DEVICES=1 python train.py --model vggvit --num_workers 0 --dataset ruxian --epochs 100<br />
```
VGG weight file can be downloaded from the website "https://download.pytorch.org/models/vgg16-397923af.pth"<br />
The path in the code needs to be modified at runtime<br />
