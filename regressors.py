# regressors.py
import torch
import torch.nn as nn
from torchvision import models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import joblib


# --- PyTorch 回归头 (用于分离特征提取器模式) ---
class PytorchRegressor(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()  # 保持Sigmoid，因为当前设计中分离模式的标签是0-1
        )

    def forward(self, x):
        return self.regressor(x)


# --- 端到端深度学习回归模型 ---
class FullCNNRegressor(nn.Module):
    def __init__(self, model_name="resnet50"):
        super().__init__()
        if model_name == "resnet50":
            base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif model_name == "resnet18":
            base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif model_name == "vgg16":
            base_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        elif model_name == "densenet121":
            base_model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        elif model_name == "efficientnet_b0":
            base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        elif model_name == "inception_v3":
            base_model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=True)
        else:
            raise ValueError(f"Unsupported full CNN model: {model_name}")

        # 替换最后一层全连接层 (分类头) 为回归头
        if isinstance(base_model, (models.ResNet, models.DenseNet)):
            num_ftrs = base_model.fc.in_features
            base_model.fc = nn.Linear(num_ftrs, 1)  # 移除Sigmoid
        elif isinstance(base_model, models.VGG):
            num_ftrs = base_model.classifier[0].in_features
            base_model.classifier = nn.Sequential(
                nn.Linear(num_ftrs, 4096),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, 1)  # 移除Sigmoid
            )
        elif isinstance(base_model, models.EfficientNet):
            num_ftrs = base_model.classifier[1].in_features
            base_model.classifier = nn.Linear(num_ftrs, 1)  # 移除Sigmoid
        elif isinstance(base_model, models.InceptionV3):
            num_ftrs = base_model.fc.in_features
            base_model.fc = nn.Linear(num_ftrs, 1)  # 移除Sigmoid
        else:
            raise ValueError(f"Cannot adapt regressor head for this model type: {model_name}")

        self.model = base_model  # 完整的CNN模型

    def forward(self, x):
        return self.model(x)

    # --- Sklearn 模型构建函数 (保持不变) ---


def get_sklearn_model_pipeline(model_name_ui="随机森林"):
    regressor = None
    if model_name_ui == "随机森林":
        regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_name_ui == "支持向量回归":
        regressor = SVR(kernel='rbf', C=10, gamma=0.1)
    elif model_name_ui == "梯度提升回归":
        regressor = GradientBoostingRegressor(n_estimators=100, random_state=42)
    elif model_name_ui == "堆叠回归":
        estimators = [
            ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
            ('gbr', GradientBoostingRegressor(n_estimators=50, random_state=42)),
            ('svr', SVR(kernel='rbf', C=1.0, gamma=0.1))
        ]
        final_estimator = RidgeCV(alphas=[0.1, 1.0, 10.0])
        regressor = StackingRegressor(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=3,
            n_jobs=-1,
            passthrough=True
        )
    else:
        raise ValueError(f"不支持的Sklearn模型类型: {model_name_ui}")

    feature_preprocessing_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95, random_state=42))
    ])

    return regressor, feature_preprocessing_pipeline
