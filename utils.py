# utils.py
import os
import re
import shutil
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import gradio as gr  # <-- 保持这一行！

# 从 config.py 导入常量
from config import DATA_DIR, SCORE_FILE_NAME


# --- 数据集类 ---
class ScoreDataset(Dataset):
    def __init__(self, image_paths, scores, transform=None):
        self.image_paths = image_paths
        self.scores = scores
        self.transform = transform

        if not self.scores:
            self.min_label = 0
            self.max_label = 100
        else:
            self.min_label = min(self.scores)
            self.max_label = max(self.scores)
            if self.max_label == self.min_label:
                self.max_label += 1

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        score = self.scores[idx]

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)

            score_normalized = (score - self.min_label) / (self.max_label - self.min_label + 1e-7)

            return image, torch.tensor(score_normalized, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading or processing image {img_path}: {e}")
            return torch.zeros(3, 224, 224), torch.tensor(0.5, dtype=torch.float32)


# --- 图片转换函数 ---
def get_transforms(train=True, image_size=224):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


# --- UI交互辅助函数 (原始数据导入Tab) ---

def load_images_from_folder_for_import(folder_path):
    all_image_paths = []
    all_scores = []

    if not os.path.isdir(folder_path):
        return [], "错误: 文件夹不存在。", "", None, None, ([], [], -1)

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        return [], "警告: 文件夹中没有找到图片。", "", None, None, ([], [], -1)

    for img_name in sorted(image_files):
        full_path = os.path.join(folder_path, img_name)
        try:
            match = re.match(r'(\d+)', img_name)
            if match:
                score = int(match.group(1))
                score = max(0, min(100, score))
            else:
                score = 50

            all_image_paths.append(full_path)
            all_scores.append(score)

        except Exception as e:
            print(f"处理文件 {img_name} 失败: {e}")
            continue

    if not all_image_paths:
        return [], "警告: 没有找到有效图片或解析分数失败。", "", None, None, ([], [], -1)

    initial_index = 0
    initial_image_name = Path(all_image_paths[initial_index]).name
    initial_preview_path = all_image_paths[initial_index]
    initial_score = all_scores[initial_index]

    current_data_state_tuple = (all_image_paths, all_scores, initial_index)

    return all_image_paths, "图片加载完成。", initial_image_name, initial_preview_path, initial_score, current_data_state_tuple


# --- UI交互辅助函数 (训练数据管理Tab) ---

def _update_managed_display_from_index(index, all_image_paths, all_scores):
    if not all_image_paths or not (0 <= index < len(all_image_paths)):
        return "", None, None, ""

    preview_path = all_image_paths[index]
    score = all_scores[index]
    image_name = Path(preview_path).name
    return image_name, preview_path, score, image_name


def _select_managed_image_for_edit(evt: gr.SelectData, current_data_state):
    all_image_paths, all_scores, _ = current_data_state
    if all_image_paths and 0 <= evt.index < len(all_image_paths):
        current_data_state = (all_image_paths, all_scores, evt.index)
        selected_name, selected_path, selected_score, delete_filename = _update_managed_display_from_index(evt.index,
                                                                                                           all_image_paths,
                                                                                                           all_scores)
        return selected_name, selected_path, selected_score, delete_filename, current_data_state
    return "", None, None, "", current_data_state


def _navigate_managed_image(direction, current_data_state):
    all_image_paths, all_scores, current_index = current_data_state
    if not all_image_paths:
        return "", None, None, "", current_data_state
    num_images = len(all_image_paths)
    if num_images == 0:
        return "", None, None, "", current_data_state
    new_index = current_index + direction
    if new_index < 0:
        new_index = num_images - 1
    elif new_index >= num_images:
        new_index = 0
    current_data_state = (all_image_paths, all_scores, new_index)
    selected_name, selected_path, selected_score, delete_filename = _update_managed_display_from_index(new_index,
                                                                                                       all_image_paths,
                                                                                                       all_scores)
    return selected_name, selected_path, selected_score, delete_filename, current_data_state


def _process_managed_single_score_edit(entered_score, current_data_state):
    all_image_paths, all_scores, selected_index = current_data_state
    if entered_score is not None:
        final_score = max(0, min(100, round(entered_score)))
    else:
        final_score = 50
    if selected_index != -1 and 0 <= selected_index < len(all_scores):
        all_scores[selected_index] = final_score
        current_data_state = (all_image_paths, all_scores, selected_index)

        status_msg, new_managed_state = save_data_to_data_dir(current_data_state)

        dataframe_data = pd.DataFrame(
            [[Path(img_path_str).name, score] for img_path_str, score in
             zip(new_managed_state[0], new_managed_state[1])],
            columns=["文件名", "分数"])

        return status_msg, dataframe_data, new_managed_state[0], new_managed_state[1], new_managed_state

    dataframe_data_current = pd.DataFrame()
    if all_image_paths and all_scores:
        dataframe_data_current = pd.DataFrame([[Path(p).name, s] for p, s in zip(all_image_paths, all_scores)],
                                              columns=["文件名", "分数"])
    return "无图片选中或数据无效。", dataframe_data_current, [], [], current_data_state


def load_training_data_for_management():
    image_paths = []
    scores = []

    score_file_path = Path(DATA_DIR) / SCORE_FILE_NAME
    if not score_file_path.exists():
        return [], "数据文件不存在，请先导入并保存数据。", ([], [], -1), [], []

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
        return [], f"错误: 读取分数文件 {score_file_path} 失败: {e}", ([], [], -1), [], []

    if not image_paths:
        return [], "没有找到有效的训练数据。", ([], [], -1), [], []

    df_data = [[Path(p).name, s] for p, s in zip(image_paths, scores)]
    status_msg = f"加载完成，共 {len(image_paths)} 张图片。"

    return df_data, status_msg, (image_paths, scores, 0), image_paths, scores


def save_data_to_data_dir(current_data_state):
    all_image_paths, all_scores, _ = current_data_state

    if not all_image_paths:
        return "没有要保存的图片和分数。", ([], [], -1)

    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

    score_file_path = Path(DATA_DIR) / SCORE_FILE_NAME

    updated_image_paths_in_data_dir = []
    try:
        with open(score_file_path, 'w') as f:
            for i, original_img_path_str in enumerate(all_image_paths):
                original_filename = Path(original_img_path_str).name
                dest_path = Path(DATA_DIR) / original_filename
                score = all_scores[i]

                if not dest_path.exists() or not Path(original_img_path_str).samefile(dest_path):
                    try:
                        shutil.copy2(original_img_path_str, dest_path)
                        print(f"复制文件: {original_img_path_str} -> {dest_path}")
                    except shutil.SameFileError:
                        pass

                f.write(f"{original_filename},{score}\n")
                updated_image_paths_in_data_dir.append(str(dest_path))

        return f"数据已保存到 {DATA_DIR}，共 {len(all_image_paths)} 条。", (
        updated_image_paths_in_data_dir, all_scores, -1)

    except Exception as e:
        return f"保存数据失败: {e}", current_data_state


def add_new_image_entry(new_image_file, new_image_name, new_score, current_data_state):
    all_image_paths, all_scores, _ = current_data_state

    if new_image_file is None and not new_image_name:
        return "请提供图片文件或图片名称。", None, None, [], [], current_data_state

    source_path_str = None
    if new_image_file:
        if isinstance(new_image_file, dict) and 'name' in new_image_file:
            source_path_str = new_image_file['name']
        elif isinstance(new_image_file, str):
            source_path_str = new_image_file

        if source_path_str:
            source_path = Path(source_path_str)
        else:
            return "无效的图片文件上传。", None, None, [], [], current_data_state

    final_image_name = new_image_name.strip() if new_image_name and new_image_name.strip() else None

    if source_path_str:
        if not final_image_name:
            final_image_name = source_path.name

        dest_path = Path(DATA_DIR) / final_image_name

        try:
            shutil.copy2(source_path, dest_path)
            new_image_path = str(dest_path)
        except Exception as e:
            return f"复制图片文件失败: {e}", None, None, [], [], current_data_state
    elif final_image_name:
        new_image_path = str(Path(DATA_DIR) / final_image_name)
        if not Path(new_image_path).exists():
            print(f"警告: 图片文件 {final_image_name} 在 data 目录中不存在，但已添加记录。")
    else:
        return "请提供图片文件或图片名称。", None, None, [], [], current_data_state

    existing_filenames = {Path(p).name for p in all_image_paths}
    if final_image_name in existing_filenames:
        return f"错误: 图片 {final_image_name} 已存在。", None, None, [], [], current_data_state

    score_to_add = max(0, min(100, round(new_score))) if new_score is not None else 50

    all_image_paths.append(new_image_path)
    all_scores.append(score_to_add)

    updated_df_data = [[Path(p).name, s] for p, s in zip(all_image_paths, all_scores)]

    status_msg, new_managed_state = save_data_to_data_dir((all_image_paths, all_scores, -1))

    return status_msg, updated_df_data, None, new_managed_state[0], new_managed_state[1], new_managed_state


def delete_image_entry(selected_filename, current_data_state):
    all_image_paths, all_scores, _ = current_data_state

    if not selected_filename:
        return "请选择要删除的图片。", None, None, [], [], current_data_state

    idx_to_delete = -1
    for i, p in enumerate(all_image_paths):
        if Path(p).name == selected_filename:
            idx_to_delete = i
            break

    if idx_to_delete == -1:
        return f"错误: 未找到图片 {selected_filename}。", None, None, [], [], current_data_state

    file_to_delete_path = Path(all_image_paths[idx_to_delete])
    try:
        if file_to_delete_path.exists():
            os.remove(file_to_delete_path)
            print(f"物理删除文件: {file_to_delete_path}")
    except Exception as e:
        print(f"删除文件 {selected_filename} 失败: {e}", e)
        return f"删除文件 {selected_filename} 失败: {e}", None, None, [], [], current_data_state

    del all_image_paths[idx_to_delete]
    del all_scores[idx_to_delete]

    updated_df_data = [[Path(p).name, s] for p, s in zip(all_image_paths, all_scores)]

    status_msg, new_managed_state = save_data_to_data_dir((all_image_paths, all_scores, -1))

    return status_msg, updated_df_data, None, new_managed_state[0], new_managed_state[1], new_managed_state


def update_data_from_management_dataframe(dataframe_data, current_data_state):
    if dataframe_data.empty:
        new_data_state = ([], [], -1)
        save_data_to_data_dir(new_data_state)
        return new_data_state[0], new_data_state[1], new_data_state

    updated_image_paths_in_data_dir = []
    updated_all_scores = []

    for _, row in dataframe_data.iterrows():
        filename = row["文件名"]
        score = max(0, min(100, round(row["分数"])))
        full_path = str(Path(DATA_DIR) / filename)

        if Path(full_path).exists():
            updated_image_paths_in_data_dir.append(full_path)
            updated_all_scores.append(score)
        else:
            print(f"警告: 文件 {filename} 在磁盘上不存在，已跳过此条更新。")

    new_data_state = (updated_image_paths_in_data_dir, updated_all_scores, -1)

    status_msg, final_saved_state = save_data_to_data_dir(new_data_state)
    print(status_msg)

    return final_saved_state[0], final_saved_state[1], final_saved_state


def get_image_size_by_model_name(model_name):
    if model_name == "inception_v3":
        return 299
    return 224
