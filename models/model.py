"""model.py - Model and module class for ViT.
   They are built to mirror those in the official Jax implementation.
"""

from re import X
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import vgg

from .transformer import Transformer
from .utils import load_pretrained_weights, as_tuple
from .configs import PRETRAINED_MODELS


class PositionalEmbedding1D(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))

    def forward(self, x):
        """Input has shape `(batch_size, seq_len, emb_dim)`"""
        return x + self.pos_embedding


class ViT(nn.Module):
    """
    Args:
        name (str): Model name, e.g. 'B_16'
        pretrained (bool): Load pretrained weights
        in_channels (int): Number of channels in input data
        num_classes (int): Number of classes, default 1000

    References:
        [1] https://openreview.net/forum?id=YicbFdNTTy
    """

    def __init__(
            self,
            name: Optional[str] = None,
            pretrained: bool = False,
            patches: int = 16,
            dim: int = 768,
            ff_dim: int = 3072,
            num_heads: int = 12,
            num_layers: int = 12,
            attention_dropout_rate: float = 0.0,
            dropout_rate: float = 0.1,
            representation_size: Optional[int] = None,
            load_repr_layer: bool = False,
            classifier: str = 'token',
            positional_embedding: str = '1d',
            in_channels: int = 3,
            image_size: Optional[int] = None,
            num_classes: Optional[int] = None,
    ):
        super().__init__()

        # Configuration
        if name is None:
            check_msg = 'must specify name of pretrained model'
            assert not pretrained, check_msg
            assert not resize_positional_embedding, check_msg
            if num_classes is None:
                num_classes = 1000
            if image_size is None:
                image_size = 384
        else:  # load pretrained model
            assert name in PRETRAINED_MODELS.keys(), \
                'name should be in: ' + ', '.join(PRETRAINED_MODELS.keys())
            config = PRETRAINED_MODELS[name]['config']
            patches = config['patches']
            dim = config['dim']
            ff_dim = config['ff_dim']
            num_heads = config['num_heads']
            num_layers = config['num_layers']
            attention_dropout_rate = config['attention_dropout_rate']
            dropout_rate = config['dropout_rate']
            representation_size = config['representation_size']
            classifier = config['classifier']
            if image_size is None:
                image_size = PRETRAINED_MODELS[name]['image_size']
            if num_classes is None:
                num_classes = PRETRAINED_MODELS[name]['num_classes']
        self.image_size = image_size

        # Image and patch sizes
        h, w = as_tuple(image_size)  # image sizes
        fh, fw = as_tuple(patches)  # patch sizes
        gh, gw = h // fh, w // fw  # number of patches
        seq_len = gh * gw

        # Patch embedding
        self.patch_embedding = nn.Conv2d(in_channels, dim, kernel_size=(fh, fw), stride=(fh, fw))

        # Class token
        if classifier == 'token':
            self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
            seq_len += 1

        # Positional embedding
        if positional_embedding.lower() == '1d':
            self.positional_embedding = PositionalEmbedding1D(seq_len, dim)
        else:
            raise NotImplementedError()

        # Transformer
        self.transformer = Transformer(num_layers=num_layers, dim=dim, num_heads=num_heads,
                                       ff_dim=ff_dim, dropout=dropout_rate)

        # Representation layer
        if representation_size and load_repr_layer:
            self.pre_logits = nn.Linear(dim, representation_size)
            pre_logits_size = representation_size
        else:
            pre_logits_size = dim

        # Classifier head
        self.norm = nn.LayerNorm(pre_logits_size, eps=1e-6)
        self.fc = nn.Linear(pre_logits_size, num_classes)

        # Initialize weights
        self.init_weights()

        # Load pretrained model
        if pretrained:
            pretrained_num_channels = 3
            pretrained_num_classes = PRETRAINED_MODELS[name]['num_classes']
            pretrained_image_size = PRETRAINED_MODELS[name]['image_size']
            load_pretrained_weights(
                self, name,
                load_first_conv=(in_channels == pretrained_num_channels),
                load_fc=(num_classes == pretrained_num_classes),
                load_repr_layer=load_repr_layer,
                resize_positional_embedding=(image_size != pretrained_image_size),
            )

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(
                    m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)

        self.apply(_init)
        nn.init.constant_(self.fc.weight, 0)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.normal_(self.positional_embedding.pos_embedding,
                        std=0.02)  # _trunc_normal(self.positional_embedding.pos_embedding, std=0.02)
        nn.init.constant_(self.class_token, 0)

    def forward(self, x):
        """Breaks image into patches, applies transformer, applies MLP head.

        Args:
            x (tensor): `b,c,fh,fw`
        """
        b, c, fh, fw = x.shape
        x = self.patch_embedding(x)  # b,d,gh,gw
        x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d
        if hasattr(self, 'class_token'):
            x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)  # b,gh*gw+1,d
        if hasattr(self, 'positional_embedding'):
            x = self.positional_embedding(x)  # b,gh*gw+1,d 
        x = self.transformer(x)  # b,gh*gw+1,d
        # print('transout',x.shape)#[16, 197, 768]
        if hasattr(self, 'pre_logits'):
            x = self.pre_logits(x)
            x = torch.tanh(x)
            # print('q',x.shape)
        if hasattr(self, 'fc'):
            x = self.norm(x)[:, 0]  # b,d
            # print('w',x.shape)#[16, 768]
            x = self.fc(x)  # b,num_classes
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)


class GlobalAveragepooling(nn.Module):
    def __init__(self):
        super(GlobalAveragepooling, self).__init__()

    def forward(self, feat):
        return F.adaptive_avg_pool2d(feat, (1, 1))


# VGG16后续模型处理预训练输出
class AUSlinear(nn.Module):
    def __init__(self):
        super(AUSlinear, self).__init__()
        self.squeeze = nn.Sequential()
        self.squeeze.add_module('GlobalAveragepooling', GlobalAveragepooling())
        self.squeeze_m = nn.AdaptiveMaxPool2d(output_size=(1, 1), return_indices=False)
        self.fc1 = nn.Linear(1024, 8)
        self.fc2 = nn.Linear(8, 64)
        self.sigmoid = nn.Sigmoid()

        self.avepooling = nn.AvgPool2d((4, 4))
        self.headmodel = nn.Sequential()
        self.headmodel.add_module('Flatten', Flatten())
        self.fc3 = nn.Linear(512, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(64, 32)

    def forward(self, x):
        squeeze_ = self.squeeze(x)
        squeeze_ = squeeze_.view(-1, 512)
        squeezem = self.squeeze_m(x)
        squeezem = squeezem.view(-1, 512)
        squeeze_ = torch.cat((squeeze_, squeezem), 1)
        squeeze_ = self.fc1(squeeze_)
        squeeze_ = F.relu(squeeze_)
        squeeze_ = self.fc2(squeeze_)
        squeeze_ = self.sigmoid(squeeze_)
        headmodel_ = self.avepooling(x)
        headmodel_ = self.headmodel(headmodel_)
        headmodel_ = self.fc3(headmodel_)
        headmodel_ = self.bn1(headmodel_)
        headmodel_ = F.relu(headmodel_)
        headmodel_ = self.dropout(headmodel_)
        out = torch.mul(squeeze_, headmodel_)
        out = self.fc4(out)
        return out


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class VGG1(nn.Module):
    def __init__(self):
        super(VGG1, self).__init__()
        vgg16 = models.vgg16()
        vgg16.load_state_dict(torch.load('/home/hdd/lhy/CMC-master/vgg16-397923af.pth'))

        self.encoder1 = vgg16.features
        self.classifier = nn.Linear(512 * 7 * 7, 128)
        self.l2norm = Normalize(2)

    def forward(self, x):
        out = self.encoder1(x)
        # out=torch.flatten(out,start_dim=1)
        # out=self.classifier(out)
        # out = self.l2norm(out)
        return out


class VGG(nn.Module):
    """Encoder for instance discrimination and MoCo"""

    def __init__(self):
        super(VGG, self).__init__()

        self.encoder = VGG1()
        self.encoder = nn.DataParallel(self.encoder)

    def forward(self, x):
        return self.encoder(x)


class sdlVggformer(nn.Module):
    def __init__(self):
        super(sdlVggformer, self).__init__()
        self.ViTblock = ViT(name='B_16_imagenet1k', pretrained=True, num_classes=32, image_size=224)
        self.Vggblock = VGG()
        self.AUSAMblock = AUSlinear()

        self.batchnorm = nn.BatchNorm1d(64)  # may remove#aus 32 vit 32 ausvit 64
        self.resultsigmoid = nn.Sigmoid()  # may remove
        self.resultdropout = nn.Dropout(p=0.5)  # may remove
        self.learnfc = nn.Linear(64, 64)  # may remove#aus 32 vit 32 ausvit 64
        self.resultfc = nn.Linear(64, 2)

    def forward(self, x):
        vit_out = self.ViTblock(x)
        vgg_out = self.Vggblock(x)
        vgg_out = self.AUSAMblock(vgg_out)
        v2out = torch.cat((vit_out, vgg_out), 1)
        # v2out=vit_out
        v2out = self.resultsigmoid(v2out)  # may remove
        v2out = self.batchnorm(v2out)  # may remove
        v2out = self.resultdropout(v2out)  # may remove
        v2out = self.learnfc(v2out)  # may remove
        v2out = self.resultfc(v2out)
        return v2out


if __name__ == '__main__':
    model = models.vgg16()
    print(model)
