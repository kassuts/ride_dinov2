from model.dinov2 import Expert_vision_transformer, vision_transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from .fb_resnets import ResNet
from .fb_resnets import ResNeXt
from .fb_resnets import Expert_ResNet
from .fb_resnets import Expert_ResNeXt 
from .ldam_drw_resnets import resnet_cifar
from .ldam_drw_resnets import expert_resnet_cifar


class Model(BaseModel):
    requires_target = False

    def __init__(self, num_classes, backbone_class=None):
        super().__init__()
        if backbone_class is not None: # Do not init backbone here if None
            self.backbone = backbone_class(num_classes)

    def _hook_before_iter(self):
        self.backbone._hook_before_iter()

    def forward(self, x, mode=None):
        x = self.backbone(x)

        assert mode is None
        return x

class ResNet10Model(Model):
    def __init__(self, num_classes, reduce_dimension=False, layer3_output_dim=None, layer4_output_dim=None, use_norm=False, num_experts=1, **kwargs):
        super().__init__(num_classes, None)
        if num_experts == 1:
            self.backbone = ResNet.ResNet(ResNet.BasicBlock, [1, 1, 1, 1], dropout=None, num_classes=num_classes, use_norm=use_norm, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim, **kwargs)
        else: 
            self.backbone = Expert_ResNet.ResNet(ResNet.BasicBlock, [1, 1, 1, 1], dropout=None, num_classes=num_classes, use_norm=use_norm, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim, num_experts=num_experts, **kwargs)
 
class ResNet32Model(Model): # From LDAM_DRW
    def __init__(self, num_classes, reduce_dimension=False, layer2_output_dim=None, layer3_output_dim=None, use_norm=False, num_experts=1, **kwargs):
        super().__init__(num_classes, None)
        if num_experts == 1:
            self.backbone = resnet_cifar.ResNet_s(resnet_cifar.BasicBlock, [5, 5, 5], num_classes=num_classes, reduce_dimension=reduce_dimension, layer2_output_dim=layer2_output_dim, layer3_output_dim=layer3_output_dim, use_norm=use_norm, **kwargs)
        else:
            self.backbone = expert_resnet_cifar.ResNet_s(expert_resnet_cifar.BasicBlock, [5, 5, 5], num_classes=num_classes, reduce_dimension=reduce_dimension, layer2_output_dim=layer2_output_dim, layer3_output_dim=layer3_output_dim, use_norm=use_norm, num_experts=num_experts, **kwargs)


class ResNet32Model_B(Model): # From LDAM_DRW
    def __init__(self, num_classes, reduce_dimension=False, layer2_output_dim=None, layer3_output_dim=None, use_norm=False, num_experts=1, **kwargs):
        super().__init__(num_classes, None)
        if num_experts == 1:
            self.backbone = resnet_cifar.ResNet_s(resnet_cifar.BasicBlockB, [5, 5, 5], num_classes=num_classes, reduce_dimension=reduce_dimension, layer2_output_dim=layer2_output_dim, layer3_output_dim=layer3_output_dim, use_norm=use_norm, **kwargs)
        else:
            self.backbone = expert_resnet_cifar.ResNet_s(expert_resnet_cifar.BasicBlockB, [5, 5, 5], num_classes=num_classes, reduce_dimension=reduce_dimension, layer2_output_dim=layer2_output_dim, layer3_output_dim=layer3_output_dim, use_norm=use_norm, num_experts=num_experts, **kwargs)


class ResNet34Model(Model):
    def __init__(self, num_classes, reduce_dimension=False, layer3_output_dim=None, layer4_output_dim=None,
                 use_norm=False, num_experts=1, **kwargs):
        super().__init__(num_classes, None)
        if num_experts == 1:
            self.backbone = ResNet.ResNet(ResNet.BasicBlock, [3, 4, 6, 3], dropout=None, num_classes=num_classes,
                                          use_norm=use_norm, reduce_dimension=reduce_dimension, reduce_first_kernel=True,
                                          layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim,
                                          **kwargs)
        else:
            self.backbone = Expert_ResNet.ResNet(Expert_ResNet.BasicBlock, [3, 4, 6, 3], dropout=None, num_classes=num_classes,
                                                 use_norm=use_norm, reduce_dimension=reduce_dimension, reduce_first_kernel=True,
                                                 layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim,
                                                 num_experts=num_experts, **kwargs)

class ResNet50Model(Model):
    def __init__(self, num_classes, reduce_dimension=False, layer3_output_dim=None, layer4_output_dim=None, use_norm=False, num_experts=1, **kwargs):
        super().__init__(num_classes, None)
        if num_experts == 1:
            self.backbone = ResNet.ResNet(ResNet.Bottleneck, [3, 4, 6, 3], dropout=None, num_classes=num_classes, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim, use_norm=use_norm, **kwargs)
        else:
            self.backbone = Expert_ResNet.ResNet(Expert_ResNet.Bottleneck, [3, 4, 6, 3], dropout=None, num_classes=num_classes, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim, use_norm=use_norm, num_experts=num_experts, **kwargs)
  
class ResNeXt50Model(Model):
    def __init__(self, num_classes, reduce_dimension=False, layer3_output_dim=None, layer4_output_dim=None, use_norm=False, num_experts=1, **kwargs):
        super().__init__(num_classes, None)
        if num_experts == 1:
            self.backbone = ResNeXt.ResNext(ResNeXt.Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4, dropout=None, num_classes=num_classes, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim, use_norm=use_norm, **kwargs)
        else:
            self.backbone = Expert_ResNeXt.ResNext(Expert_ResNeXt.Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4, dropout=None, num_classes=num_classes, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim, num_experts=num_experts, use_norm=use_norm, **kwargs)
 
class ResNet101Model(Model):
    def __init__(self, num_classes, reduce_dimension=False, layer3_output_dim=None, layer4_output_dim=None, use_norm=False, num_experts=1, **kwargs):
        super().__init__(num_classes, None)
        if num_experts == 1:
            self.backbone = ResNet.ResNet(ResNet.Bottleneck, [3, 4, 23, 3], dropout=None, num_classes=num_classes, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim, use_norm=use_norm, **kwargs)
        else:
            self.backbone = Expert_ResNet.ResNet(Expert_ResNet.Bottleneck, [3, 4, 23, 3], dropout=None, num_classes=num_classes, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim, use_norm=use_norm, num_experts=num_experts, **kwargs)

   
def init_weights(model, weights_path="./model/pretrained_model_places/resnet152.pth", caffe=False, classifier=False):
    """Initialize weights"""
    print('Pretrained %s weights path: %s' % ('classifier' if classifier else 'feature model',  weights_path))
    weights = torch.load(weights_path)
    weights1 = {}
    if not classifier:
        if caffe: 
            # lower layers are the shared backbones
            for k in model.state_dict():
                if 'layer3s' not in k and 'layer4s' not in k:
                    weights1[k] =  weights[k] if k in weights else model.state_dict()[k]
                elif 'num_batches_tracked' in k:
                    weights1[k] =  weights[k] if k in weights else model.state_dict()[k]
                    
                elif 'layer3s.0.' in k and 'num_batches_tracked' not in k:
                    weights1[k] = weights[k.replace('layer3s.0.','layer3.')]
                elif 'layer3s.1.' in k and 'num_batches_tracked' not in k:
                    weights1[k] = weights[k.replace('layer3s.1.','layer3.')]
                elif 'layer3s.2.' in k and 'num_batches_tracked' not in k:
                    weights1[k] = weights[k.replace('layer3s.2.','layer3.')]                       
                elif 'layer4s.0.' in k and 'num_batches_tracked' not in k:
                    weights1[k] = weights[k.replace('layer4s.0.','layer4.')]
                elif 'layer4s.1.' in k and 'num_batches_tracked' not in k:
                    weights1[k] = weights[k.replace('layer4s.1.','layer4.')]
                elif 'layer4s.2.' in k and 'num_batches_tracked' not in k:
                    weights1[k] = weights[k.replace('layer4s.2.','layer4.')]
 
        else:
            weights = weights['state_dict_best']['feat_model']
            weights = {k: weights['module.' + k] if 'module.' + k in weights else model.state_dict()[k]
                       for k in model.state_dict()}
    else:
        weights = weights['state_dict_best']['classifier']
        weights = {k: weights['module.fc.' + k] if 'module.fc.' + k in weights else model.state_dict()[k]
                   for k in model.state_dict()}
    model.load_state_dict(weights1)
    return model

class ResNet152Model(Model):
    def __init__(self, num_classes, reduce_dimension=False, layer3_output_dim=None, layer4_output_dim=None, share_layer3=False, use_norm=False, num_experts=1, **kwargs):
        super().__init__(num_classes, None)
        if num_experts == 1:
            self.backbone = ResNet.ResNet(ResNet.Bottleneck, [3, 8, 36, 3], dropout=None, num_classes=num_classes, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim, use_norm=use_norm, **kwargs)
        else:
            self.backbone = Expert_ResNet.ResNet(Expert_ResNet.Bottleneck, [3, 8, 36, 3], dropout=None, num_classes=num_classes, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim, share_layer3=share_layer3, use_norm=use_norm, num_experts=num_experts, **kwargs)
            self.backbone =  init_weights(model = self.backbone, weights_path="./model/pretrained_model_places/resnet152.pth", caffe=True)
 
class ResNeXt152Model(Model):
    def __init__(self, num_classes, reduce_dimension=False, layer3_output_dim=None, layer4_output_dim=None, use_norm=False, num_experts=1, **kwargs):
        super().__init__(num_classes, None)
        if num_experts == 1:
            self.backbone = ResNeXt.ResNext(ResNeXt.Bottleneck, [3, 8, 36, 3], groups=32, width_per_group=4, dropout=None, num_classes=num_classes, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim)
        else:
            self.backbone = Expert_ResNeXt.ResNext(Expert_ResNeXt.Bottleneck, [3, 8, 36, 3], groups=32, width_per_group=4, dropout=None, num_classes=num_classes, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim, num_experts=num_experts)

class DINOv2Model(Model):
    def __init__(self, num_classes, img_size=518,  patch_size=14, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, num_experts=1, returns_feat=True, expert_start_layer=10, pretrained_path=None):
        super().__init__(num_classes, None)
        if num_experts == 1:
            self.backbone = vision_transformer.DinoVisionTransformer(img_size, patch_size, embed_dim, depth, num_heads, mlp_ratio, num_classes, pretrained_path)
        else:
            self.backbone = Expert_vision_transformer.RIDEDinoVisionTransformer(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, num_experts=num_experts, num_classes=num_classes, returns_feat=returns_feat, expert_start_layer=expert_start_layer, pretrained_path=pretrained_path)
