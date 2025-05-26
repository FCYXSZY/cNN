# train_predict.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from pathlib import Path
from PIL import Image

# 从config.py导入配置
from config import DATA_DIR, SCORE_FILE_NAME, MODEL_SAVE_BASE_PATH, \
    DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_LR

# 从其他模块导入
from utils import ScoreDataset, get_transforms, get_image_size_by_model_name
from feature_extractor import FeatureExtractor
from regressors import PytorchRegressor, get_sklearn_model_pipeline, FullCNNRegressor

# --- 配置 Matplotlib 支持中文 ---
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ------------------------------------

class TrainingAndPredictionEngine:
    """
    负责管理整个训练和预测流程的引擎。
    包含数据准备、模型切换、训练循环和预测功能。
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Device Count: {torch.cuda.device_count()}")
            print(f"Current CUDA Device Name: {torch.cuda.get_device_name(0)}")

        # 初始化模型实例为None，根据UI选择动态加载
        self.feature_extractor = None
        self.pytorch_regressor = None
        self.full_cnn_regressor = None
        self.sklearn_regressor = None
        self.sklearn_feature_pipeline = None

        self.current_model_type = None  # 当前选择的Gradio模型类型（中文）
        self.active_base_cnn_name = None  # 当前深度学习模式下选择的基础CNN名称（英文）

        self.loss_history = []
        self.dataloader = None

        Path(os.path.dirname(MODEL_SAVE_BASE_PATH)).mkdir(exist_ok=True, parents=True)

    def _get_internal_model_name(self, ui_model_name):
        """将UI模型名称映射到内部统一的英文名称，用于文件保存和加载。"""
        mapping = {
            "深度学习": "pytorch_detached",
            "端到端深度学习": "pytorch_full_cnn",
            "随机森林": "random_forest",
            "支持向量回归": "svr",
            "梯度提升回归": "gradient_boosting",
            "堆叠回归": "stacking"
        }
        return mapping.get(ui_model_name, "unknown_model")

    def switch_model_type(self, model_type_str, base_cnn_name="resnet50"):
        self.current_model_type = model_type_str
        self.active_base_cnn_name = base_cnn_name

        print(f"已切换到 {self.current_model_type} 模型模式, 基础CNN: {self.active_base_cnn_name}.")

        # 实例化或重新实例化模型
        if self.current_model_type == "深度学习":
            self.feature_extractor = FeatureExtractor(model_name=self.active_base_cnn_name).to(self.device)
            self.feature_extractor.eval()
            feature_dim = self.feature_extractor.get_output_dim()
            self.pytorch_regressor = PytorchRegressor(in_features=feature_dim).to(self.device)
            # 清除其他模型的引用
            self.full_cnn_regressor = None
            self.sklearn_regressor = None
            self.sklearn_feature_pipeline = None
        elif self.current_model_type == "端到端深度学习":
            self.full_cnn_regressor = FullCNNRegressor(model_name=self.active_base_cnn_name).to(self.device)
            # 清除其他模型的引用
            self.feature_extractor = None
            self.pytorch_regressor = None
            self.sklearn_regressor = None
            self.sklearn_feature_pipeline = None
        else:  # Sklearn模型 (随机森林, SVR, 梯度提升回归, 堆叠回归)
            # Sklearn模型仍需要FeatureExtractor来提取特征
            self.feature_extractor = FeatureExtractor(model_name=self.active_base_cnn_name).to(self.device)
            self.feature_extractor.eval()
            self.sklearn_regressor, self.sklearn_feature_pipeline = get_sklearn_model_pipeline(self.current_model_type)
            # 清除PyTorch模型的引用
            self.pytorch_regressor = None
            self.full_cnn_regressor = None

        return f"已切换到 {self.current_model_type} 模型模式, 基础CNN: {self.active_base_cnn_name}."

    def prepare_data_for_training(self, batch_size=DEFAULT_BATCH_SIZE):
        image_paths = []
        scores = []

        score_file_path = Path(DATA_DIR) / SCORE_FILE_NAME
        if not score_file_path.exists():
            return False, f"错误: 训练数据文件 {score_file_path} 不存在。请先在‘原始数据导入’或‘训练数据管理’标签页保存数据。"

        try:
            with open(score_file_path, 'r') as f:
                for line in f:
                    filename, score_str = line.strip().split(',')
                    full_image_path = Path(DATA_DIR) / filename
                    if full_image_path.exists():
                        image_paths.append(str(full_image_path))
                        scores.append(float(score_str))
                    else:
                        print(f"警告: 图像文件 {full_image_path} 不存在，已跳过。")

        except Exception as e:
            return False, f"错误: 读取分数文件 {score_file_path} 失败: {e}"

        if not image_paths:
            return False, "没有找到有效的图片数据用于训练。请检查 'data' 文件夹。"

        if self.active_base_cnn_name is None:
            print("警告: active_base_cnn_name 未设置，使用默认resnet50尺寸。")
            current_image_size = get_image_size_by_model_name("resnet50")
        else:
            current_image_size = get_image_size_by_model_name(self.active_base_cnn_name)

        transform = get_transforms(train=True, image_size=current_image_size)
        dataset = ScoreDataset(image_paths, scores, transform=transform)
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=os.cpu_count() // 2 or 1,
            drop_last=True
        )

        return True, f"数据准备完成，共 {len(dataset)} 张图片，将使用 {len(self.dataloader)} 个批次进行训练。"

    def train_model(self, epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR):
        self.loss_history = []
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_xlabel("训练步数")
        ax.set_ylabel("损失")
        ax.set_title("训练损失曲线")

        if self.dataloader is None or len(self.dataloader.dataset) == 0:
            ax.text(0.5, 0.5, "无数据，请先加载图片并保存。", horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12, color='red')
            plt.tight_layout()
            return fig

        if self.current_model_type == "深度学习":
            if self.pytorch_regressor is None or self.feature_extractor is None:
                ax.text(0.5, 0.5, "深度学习模型（分离模式）未正确初始化。请重试。", horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes, fontsize=12, color='red')
                plt.tight_layout()
                return fig

            model = self.pytorch_regressor
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            model.train()
            self.feature_extractor.eval()

            print(f"开始训练深度学习模型 (PyTorch, 分离模式, 基础CNN: {self.active_base_cnn_name})，共 {epochs} 轮次...")
            for epoch in range(epochs):
                running_loss = 0.0
                for batch_idx, (images, labels) in enumerate(self.dataloader):
                    images = images.to(self.device)
                    labels = labels.unsqueeze(1).to(self.device)

                    with torch.no_grad():
                        features = self.feature_extractor(images)

                    optimizer.zero_grad()
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    self.loss_history.append(loss.item())

                print(f"Epoch {epoch + 1}/{epochs}, 平均损失: {running_loss / len(self.dataloader):.4f}")

            ax.clear()
            ax.plot(self.loss_history, color='blue')
            ax.set_xlabel("训练步数")
            ax.set_ylabel("损失")
            ax.set_title(
                f"深度学习模型训练完成 (分离模式, 基础CNN: {self.active_base_cnn_name}, 最终损失: {self.loss_history[-1]:.4f})")
            plt.tight_layout()

            self._save_model_artifacts("深度学习")

        elif self.current_model_type == "端到端深度学习":
            if self.full_cnn_regressor is None:
                ax.text(0.5, 0.5, "端到端深度学习模型未正确初始化。请重试。", horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes, fontsize=12, color='red')
                plt.tight_layout()
                return fig

            model = self.full_cnn_regressor
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            model.train()

            print(f"开始训练端到端深度学习模型 (基础CNN: {self.active_base_cnn_name})，共 {epochs} 轮次...")
            for epoch in range(epochs):
                running_loss = 0.0
                for batch_idx, (images, labels) in enumerate(self.dataloader):
                    images = images.to(self.device)
                    labels = labels.unsqueeze(1).to(self.device)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    self.loss_history.append(loss.item())

                print(f"Epoch {epoch + 1}/{epochs}, 平均损失: {running_loss / len(self.dataloader):.4f}")

            ax.clear()
            ax.plot(self.loss_history, color='blue')
            ax.set_xlabel("训练步数")
            ax.set_ylabel("损失")
            ax.set_title(
                f"端到端深度学习模型训练完成 (基础CNN: {self.active_base_cnn_name}, 最终损失: {self.loss_history[-1]:.4f})")
            plt.tight_layout()

            self._save_model_artifacts("端到端深度学习")

        elif self.current_model_type in ["随机森林", "支持向量回归", "梯度提升回归", "堆叠回归"]:
            if self.sklearn_regressor is None or self.sklearn_feature_pipeline is None or self.feature_extractor is None:
                ax.text(0.5, 0.5, "Sklearn模型或其特征提取器未正确初始化。请重试。", horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes, fontsize=12, color='red')
                plt.tight_layout()
                return fig

            print(f"正在提取所有图片的特征用于Sklearn模型训练 (基础CNN: {self.active_base_cnn_name})...")
            all_features = []
            all_labels = []
            self.feature_extractor.eval()

            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(self.dataloader):
                    images = images.to(self.device)
                    features = self.feature_extractor(images).cpu().numpy()
                    all_features.extend(features)
                    all_labels.extend(labels.numpy())

            X = np.array(all_features)
            y = np.array(all_labels)

            if X.shape[0] == 0:
                ax.text(0.5, 0.5, "无特征数据，请检查图片加载。", horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes, fontsize=12, color='red')
                plt.tight_layout()
                return fig

            print("正在对提取的特征进行预处理 (标准化, PCA)...")
            X_processed = self.sklearn_feature_pipeline.fit_transform(X)

            print(f"正在训练Sklearn {self.current_model_type} 模型...")
            self.sklearn_regressor.fit(X_processed, y)
            print(f"Sklearn {self.current_model_type} 模型训练完成。")

            ax.clear()
            ax.text(0.5, 0.5, f"Sklearn {self.current_model_type} 训练完成",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12, color='green')
            ax.set_title(f"Sklearn模型训练状态")
            ax.axis('off')
            plt.tight_layout()

            self._save_model_artifacts(self.current_model_type)

        else:
            ax.text(0.5, 0.5, "未选择有效的模型类型进行训练。", horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12, color='red')
            plt.tight_layout()
            return fig

        return fig

    def _save_model_artifacts(self, model_type_str):
        """内部方法：保存模型文件，根据模型类型和基础CNN名称命名"""
        internal_name = self._get_internal_model_name(model_type_str)

        if internal_name == "pytorch_detached":
            feat_extractor_path = f"{MODEL_SAVE_BASE_PATH}_{self.active_base_cnn_name}_features.pth"
            regressor_path = f"{MODEL_SAVE_BASE_PATH}_{self.active_base_cnn_name}_pytorch_detached_regressor.pth"
            torch.save(self.feature_extractor.state_dict(), feat_extractor_path)
            torch.save(self.pytorch_regressor.state_dict(), regressor_path)
            print(
                f"PyTorch模型组件 (分离模式, 基础CNN: {self.active_base_cnn_name}) 已保存到: {feat_extractor_path} 和 {regressor_path}")
        elif internal_name == "pytorch_full_cnn":
            full_cnn_path = f"{MODEL_SAVE_BASE_PATH}_{self.active_base_cnn_name}_full_cnn.pth"
            torch.save(self.full_cnn_regressor.state_dict(), full_cnn_path)
            print(f"端到端深度学习模型 ({self.active_base_cnn_name}) 已保存到: {full_cnn_path}")
        else:  # Sklearn 模型
            regressor_path = f"{MODEL_SAVE_BASE_PATH}_{internal_name}_regressor.pkl"
            pipeline_path = f"{MODEL_SAVE_BASE_PATH}_{internal_name}_feature_pipeline.pkl"
            joblib.dump(self.sklearn_regressor, regressor_path)
            joblib.dump(self.sklearn_feature_pipeline, pipeline_path)
            print(f"Sklearn {model_type_str} 模型和特征管道已保存。")

    def _load_model_artifacts(self, model_type_str, base_cnn_name_to_load):
        """内部方法：加载模型文件，根据模型类型和基础CNN名称选择"""
        internal_name = self._get_internal_model_name(model_type_str)

        if internal_name == "pytorch_detached":
            try:
                self.feature_extractor = FeatureExtractor(model_name=base_cnn_name_to_load).to(self.device)
                feat_extractor_path = f"{MODEL_SAVE_BASE_PATH}_{base_cnn_name_to_load}_features.pth"
                self.feature_extractor.load_state_dict(
                    torch.load(feat_extractor_path, map_location=self.device))
                self.feature_extractor.eval()

                feature_dim = self.feature_extractor.get_output_dim()
                self.pytorch_regressor = PytorchRegressor(in_features=feature_dim).to(self.device)
                regressor_path = f"{MODEL_SAVE_BASE_PATH}_{base_cnn_name_to_load}_pytorch_detached_regressor.pth"
                self.pytorch_regressor.load_state_dict(
                    torch.load(regressor_path, map_location=self.device))
                self.pytorch_regressor.eval()
                print(f"PyTorch模型组件 (分离模式, 基础CNN: {base_cnn_name_to_load}) 已加载。")
                return True
            except FileNotFoundError as e:
                print(f"PyTorch模型文件 (分离模式, 基础CNN: {base_cnn_name_to_load}) 未找到: {e}")
                self.pytorch_regressor = None
                self.feature_extractor = None
                return False
        elif internal_name == "pytorch_full_cnn":
            try:
                self.full_cnn_regressor = FullCNNRegressor(model_name=base_cnn_name_to_load).to(self.device)
                full_cnn_path = f"{MODEL_SAVE_BASE_PATH}_{base_cnn_name_to_load}_full_cnn.pth"
                self.full_cnn_regressor.load_state_dict(
                    torch.load(full_cnn_path, map_location=self.device))
                self.full_cnn_regressor.eval()
                print(f"端到端深度学习模型 ({base_cnn_name_to_load}) 已加载。")
                return True
            except FileNotFoundError as e:
                print(f"端到端深度学习模型文件 ({base_cnn_name_to_load}) 未找到: {e}")
                self.full_cnn_regressor = None
                return False
        else:  # Sklearn 模型
            try:
                self.feature_extractor = FeatureExtractor(model_name=base_cnn_name_to_load).to(self.device)
                self.feature_extractor.eval()

                regressor_path = f"{MODEL_SAVE_BASE_PATH}_{internal_name}_regressor.pkl"
                pipeline_path = f"{MODEL_SAVE_BASE_PATH}_{internal_name}_feature_pipeline.pkl"
                self.sklearn_regressor = joblib.load(regressor_path)
                self.sklearn_feature_pipeline = joblib.load(pipeline_path)
                print(f"Sklearn {model_type_str} 模型和特征管道 (基础CNN: {base_cnn_name_to_load}) 已加载。")
                return True
            except FileNotFoundError as e:
                print(f"Sklearn模型文件 {regressor_path} 或 {pipeline_path} 未找到: {e}")
                self.sklearn_regressor = None
                self.sklearn_feature_pipeline = None
                self.feature_extractor = None
                return False
        return False

    def predict_score(self, image_path, model_type_str, base_cnn_name_for_predict):
        if not self._load_model_artifacts(model_type_str, base_cnn_name_for_predict):
            return "模型未训练或未加载！请先训练对应模型。"

        current_image_size = get_image_size_by_model_name(base_cnn_name_for_predict)
        transform = get_transforms(train=False, image_size=current_image_size)
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(self.device)
        except Exception as e:
            return f"图片加载或预处理失败: {e}"

        output_score = 0
        if model_type_str == "深度学习":
            self.pytorch_regressor.eval()
            self.feature_extractor.eval()
            with torch.no_grad():
                features = self.feature_extractor(image_tensor)
                output_score = self.pytorch_regressor(features).item() * 100
        elif model_type_str == "端到端深度学习":
            self.full_cnn_regressor.eval()
            with torch.no_grad():
                raw_output = self.full_cnn_regressor(image_tensor).item()
                output_score = max(0, min(1, raw_output)) * 100
        else:  # Sklearn模型
            self.feature_extractor.eval()
            with torch.no_grad():
                features = self.feature_extractor(image_tensor).cpu().numpy()
            processed_features = self.sklearn_feature_pipeline.transform(features)
            output_score = self.sklearn_regressor.predict(processed_features)[0] * 100

        output_score = max(0, min(100, output_score))
        return f"预测分数: {output_score:.2f} (百分制)"
