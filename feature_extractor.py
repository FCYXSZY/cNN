# feature_extractor.py
import torch
import torch.nn as nn
from torchvision import models


class FeatureExtractor(nn.Module):
    """
    负责从图像中提取特征的模块。
    支持不同的预训练CNN模型作为基础。
    """

    def __init__(self, model_name="resnet50"):
        super().__init__()

        # 根据 model_name 加载不同的预训练模型
        if model_name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif model_name == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif model_name == "vgg16":
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        elif model_name == "densenet121":
            model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        elif model_name == "efficientnet_b0":
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        elif model_name == "inception_v3":
            model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=True)
        else:
            raise ValueError(f"Unsupported feature extractor model: {model_name}")

        # 移除模型的最后一层（分类层），只保留特征提取部分
        # 并动态获取输出特征维度
        if isinstance(model, (models.ResNet, models.DenseNet)):
            self.features = nn.Sequential(*list(model.children())[:-1])
            self.out_features_dim = model.fc.in_features
        elif isinstance(model, models.VGG):
            self.features = model.features
            self.out_features_dim = model.classifier[0].in_features
        elif isinstance(model, models.EfficientNet):
            self.features = model.features
            self.out_features_dim = model.classifier[1].in_features
        elif isinstance(model, models.InceptionV3):
            # InceptionV3在eval模式下，forward函数返回tensor，无需InceptionOutputs处理
            # 但为了保持和训练时（可能有aux_logits）的统一，通常需要修改其结构
            # 对于特征提取，我们只保留其特征部分，并获取最后一个pooling层输出的维度
            # 确保其在推理时行为一致
            self.features = nn.Sequential(
                model.Conv2d_1a_3x3, model.Conv2d_2a_3x3, model.Conv2d_2b_3x3,
                model.maxpool1, model.Conv2d_3b_1x1, model.Conv2d_4a_3x3,
                model.maxpool2, model.Mixed_5b, model.Mixed_5c, model.Mixed_5d,
                model.Mixed_6a, model.Mixed_6b, model.Mixed_6c, model.Mixed_6d, model.Mixed_6e,
                model.Mixed_7a, model.Mixed_7b, model.Mixed_7c, model.avgpool  # 最后一个是avgpool
            )
            self.out_features_dim = model.fc.in_features  # 通过fc层的in_features获取
        else:
            raise ValueError(f"Unknown model type for feature extraction: {model_name}")

    def forward(self, x):
        features = self.features(x)
        if features.dim() > 2:
            features = torch.flatten(features, 1)
        return features

    def get_output_dim(self):
        return self.out_features_dim

