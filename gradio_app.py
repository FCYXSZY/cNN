import gradio as gr
import pandas as pd
from pathlib import Path
import os
import matplotlib.pyplot as plt

# 导入所有模块中的核心函数和配置
from utils import load_images_from_folder_for_import, save_data_to_data_dir, \
    load_training_data_for_management, \
    add_new_image_entry, delete_image_entry, update_data_from_management_dataframe, \
    _update_managed_display_from_index, _select_managed_image_for_edit, _navigate_managed_image, \
    _process_managed_single_score_edit

from train_predict import TrainingAndPredictionEngine
from config import DEFAULT_EPOCHS, DEFAULT_LR, MODEL_SAVE_BASE_PATH, DATA_DIR


# --- UI辅助函数 (原始数据导入 Tab 专用) ---
# 这些函数现在只服务于“原始数据导入”Tab，所以在这里重新定义
def _update_main_display_from_index(index, all_image_paths, all_scores):
    if not all_image_paths or not (0 <= index < len(all_image_paths)):
        return "", None, None

    preview_path = all_image_paths[index]
    score = all_scores[index]
    image_name = Path(preview_path).name
    return image_name, preview_path, score


def _select_image_for_edit(evt: gr.SelectData, current_data_state):
    all_image_paths, all_scores, _ = current_data_state
    if all_image_paths and 0 <= evt.index < len(all_image_paths):
        current_data_state = (all_image_paths, all_scores, evt.index)
        selected_name, selected_path, selected_score = _update_main_display_from_index(evt.index, all_image_paths,
                                                                                       all_scores)
        return selected_name, selected_path, selected_score, current_data_state
    return "", None, None, current_data_state


def _navigate_image(direction, current_data_state):
    all_image_paths, all_scores, current_index = current_data_state
    if not all_image_paths:
        return "", None, None, current_data_state
    num_images = len(all_image_paths)
    if num_images == 0:
        return "", None, None, current_data_state
    new_index = current_index + direction
    if new_index < 0:
        new_index = num_images - 1
    elif new_index >= num_images:
        new_index = 0
    current_data_state = (all_image_paths, all_scores, new_index)
    selected_name, selected_path, selected_score = _update_main_display_from_index(new_index, all_image_paths,
                                                                                   all_scores)
    return selected_name, selected_path, selected_score, current_data_state


def _process_single_score_edit(entered_score, current_data_state):
    all_image_paths, all_scores, selected_index = current_data_state
    if entered_score is not None:
        final_score = max(0, min(100, round(entered_score)))
    else:
        final_score = 50
    if selected_index != -1 and 0 <= selected_index < len(all_scores):
        all_scores[selected_index] = final_score
        current_data_state = (all_image_paths, all_scores, selected_index)
        dataframe_data = pd.DataFrame(
            [[Path(img_path_str).name, score] for img_path_str, score in zip(all_image_paths, all_scores)],
            columns=["文件名", "分数"])
        return dataframe_data, current_data_state
    dataframe_data_current = pd.DataFrame()
    if all_image_paths and all_scores:
        dataframe_data_current = pd.DataFrame([[Path(p).name, s] for p, s in zip(all_image_paths, all_scores)],
                                              columns=["文件名", "分数"])
    return dataframe_data_current, current_data_state


def _create_static_error_plot(message):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, message,
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, fontsize=12, color='red')
    ax.set_title("训练失败")
    ax.axis('off')
    plt.tight_layout()
    return fig


# --- 全局状态和初始化 ---
training_engine = TrainingAndPredictionEngine()

# 可选的基础CNN模型列表
BASE_CNN_MODELS = ["resnet50", "resnet18", "vgg16", "densenet121", "efficientnet_b0", "inception_v3"]

# --- Gradio UI界面定义 ---
with gr.Blocks(title="图片分数编辑与模型训练") as demo:
    # 用于“原始数据导入”Tab的状态
    current_import_data_state = gr.State(([], [], -1))
    # 用于“训练数据管理”Tab的状态，这个状态会直接关联 data/scores.txt 的内容
    current_managed_data_state = gr.State(([], [], -1))

    gr.Markdown("## 图片分数编辑与模型训练系统")
    gr.Markdown("### 使用流程：")
    gr.Markdown(
        "1. **原始数据导入**: 从外部文件夹加载图片，解析文件名中的分数进行初始标注，并可预览、编辑。点击“保存数据”将图片复制到 `data` 目录并生成 `scores.txt`。")
    gr.Markdown(
        "2. **训练数据管理**: 浏览、编辑、添加、删除已保存到 `data` 目录的训练图片和分数。此处的更改会立即同步到文件。")
    gr.Markdown("3. **训练模型**: 选择模型类型和基础CNN模型，调整参数，开始训练。")
    gr.Markdown("4. **预测图片**: 上传图片，选择训练好的模型进行预测。")

    with gr.Tab("原始数据导入"):
        gr.Markdown("### 从外部文件夹导入图片并进行初始分数编辑")
        with gr.Row():
            folder_input = gr.Textbox(label="输入图片文件夹路径", placeholder="请输入绝对路径，例如 D:\\images",
                                      value="")
            load_btn = gr.Button("加载图片", variant="primary", scale=0)

        import_status_text = gr.Textbox(label="操作状态", interactive=False, max_lines=2)

        with gr.Column():
            gallery = gr.Gallery(
                label="图片列表", show_label=True, elem_id="gallery", height=600,
                preview=True, columns=5, rows=2, object_fit="contain"
            )

            with gr.Group():
                gr.Markdown("#### 当前图片信息与分数编辑")
                with gr.Row():
                    current_image_name_display = gr.Textbox(
                        label="当前图片名称", interactive=False, show_label=True,
                        placeholder="图片名称将在此处显示...", scale=3
                    )
                    current_score_input = gr.Number(
                        label="当前分数 (0-100)",
                        scale=1
                    )
                    confirm_score_btn = gr.Button("确定", variant="secondary", scale=0)

            with gr.Row():
                prev_btn = gr.Button("⬅️ 上一张", scale=1)
                next_btn = gr.Button("下一张 ➡️", scale=1)

            current_image_preview = gr.Image(
                label="当前选中图片（预览）", height=300, show_download_button=False,
                container=False, interactive=False
            )

        import_score_dataframe = gr.Dataframe(
            headers=["文件名", "分数"], datatype=["str", "number"], col_count=(2, "fixed"),
            interactive=True, label="所有图片分数表格 (可直接编辑)", value=[]
        )

        save_import_data_btn = gr.Button("保存数据到训练目录", variant="secondary")

    with gr.Tab("训练数据管理"):
        gr.Markdown("### 管理已保存的训练图片和分数")
        gr.Markdown(f"当前训练数据目录: `{DATA_DIR}`")
        load_managed_data_btn = gr.Button("加载训练数据", variant="primary")
        managed_data_status_text = gr.Textbox(label="管理状态", interactive=False, max_lines=2)

        with gr.Column():
            managed_gallery = gr.Gallery(
                label="已保存图片列表", show_label=True, elem_id="managed_gallery", height=400,
                preview=True, columns=5, rows=2, object_fit="contain", value=[]
            )

            with gr.Group():
                gr.Markdown("#### 当前图片信息与分数编辑 (训练数据)")
                with gr.Row():
                    managed_image_name_display = gr.Textbox(
                        label="当前图片名称", interactive=False, show_label=True,
                        placeholder="图片名称将在此处显示...", scale=3
                    )
                    managed_score_input = gr.Number(
                        label="当前分数 (0-100)",
                        scale=1
                    )
                    confirm_managed_score_btn = gr.Button("确定", variant="secondary", scale=0)

            with gr.Row():
                prev_managed_btn = gr.Button("⬅️ 上一张", scale=1)
                next_managed_btn = gr.Button("下一张 ➡️", scale=1)

            managed_image_preview = gr.Image(
                label="当前选中图片（预览）", height=200, show_download_button=False,
                container=False, interactive=False
            )

        managed_score_dataframe = gr.Dataframe(
            headers=["文件名", "分数"], datatype=["str", "number"], col_count=(2, "fixed"),
            interactive=True, label="所有图片分数表格 (可直接编辑)",  value=[]
        )

        with gr.Accordion("添加/删除图片条目", open=False):
            with gr.Row():
                new_image_file_input = gr.File(label="上传新图片文件", type="file")
                new_image_name_input = gr.Textbox(label="或输入文件名 (例如：85.jpg)", placeholder="若上传文件可留空",
                                                  scale=1)
                new_score_input = gr.Number(label="新分数 (0-100)", value=50)
                add_entry_btn = gr.Button("添加条目", variant="secondary", scale=0)
            with gr.Row():
                delete_filename_input = gr.Textbox(label="要删除的文件名", placeholder="请输入完整文件名，例如 85.jpg")
                delete_entry_btn = gr.Button("删除条目", variant="stop", scale=0)

    with gr.Tab("训练模型"):
        gr.Markdown("## 模型训练")
        gr.Markdown(
            "1. **重要**: 确保在“原始数据导入”或“训练数据管理”标签页已加载并**保存**了数据。模型将使用 `data` 文件夹中的图片和分数。")
        gr.Markdown("2. 选择模型类型和基础CNN模型，调整训练参数。")
        gr.Markdown("3. 点击“开始训练”按钮。训练过程中的损失曲线会实时显示。")

        with gr.Row():
            model_type_selector = gr.Dropdown(
                ["深度学习", "端到端深度学习", "随机森林", "支持向量回归", "梯度提升回归", "堆叠回归"],
                label="选择模型类型", value="深度学习", interactive=True
            )
            epochs_input = gr.Slider(1, 100, DEFAULT_EPOCHS, label="训练轮次", step=1)
            lr_input = gr.Number(DEFAULT_LR, label="学习率", precision=4)

        base_cnn_selector_train = gr.Dropdown(
            BASE_CNN_MODELS,
            label="选择基础CNN模型 (深度学习模式)",
            value="resnet50",
            interactive=True,
            visible=True
        )

        train_start_btn = gr.Button("开始训练", variant="primary")
        train_status_text = gr.Textbox(label="训练状态", interactive=False)
        loss_plot_output = gr.Plot(label="训练损失曲线")

    with gr.Tab("预测图片"):
        gr.Markdown("## 图片预测")
        gr.Markdown("1. 选择之前训练过的模型类型。")
        gr.Markdown("2. 上传您要预测的图片。")
        gr.Markdown("3. 点击“预测”按钮获取分数。")

        with gr.Row():
            predict_model_type_selector = gr.Dropdown(
                ["深度学习", "端到端深度学习", "随机森林", "支持向量回归", "梯度提升回归", "堆叠回归"],
                label="选择预测模型类型", value="深度学习", interactive=True
            )
            image_for_predict = gr.Image(type="filepath", label="上传图片进行预测")
            predict_btn = gr.Button("预测", variant="primary")
            predicted_score_output = gr.Label(label="预测结果")

        base_cnn_selector_predict = gr.Dropdown(
            BASE_CNN_MODELS,
            label="选择基础CNN模型 (深度学习模式)",
            value="resnet50",
            interactive=True,
            visible=True
        )

    # --- 事件绑定 ---

    # 1. 原始数据导入 Tab 的事件
    load_btn.click(
        fn=load_images_from_folder_for_import,
        inputs=[folder_input],
        outputs=[
            gallery,
            import_status_text,
            current_image_name_display,
            current_image_preview,
            current_score_input,
            current_import_data_state
        ]
    ).success(
        fn=lambda state_tuple_from_load: pd.DataFrame(
            [[Path(p).name, s] for p, s in zip(state_tuple_from_load[0], state_tuple_from_load[1])],
            columns=["文件名", "分数"]
        ),
        inputs=[current_import_data_state],
        outputs=import_score_dataframe
    )

    gallery.select(
        fn=_select_image_for_edit,  # 使用内部定义的 _select_image_for_edit
        inputs=[current_import_data_state],
        outputs=[current_image_name_display, current_image_preview, current_score_input, current_import_data_state]
    )

    prev_btn.click(
        fn=_navigate_image,  # 使用内部定义的 _navigate_image
        inputs=[gr.State(-1), current_import_data_state],  # -1 for previous
        outputs=[current_image_name_display, current_image_preview, current_score_input, current_import_data_state]
    )
    next_btn.click(
        fn=_navigate_image,  # 使用内部定义的 _navigate_image
        inputs=[gr.State(1), current_import_data_state],  # 1 for next
        outputs=[current_image_name_display, current_image_preview, current_score_input, current_import_data_state]
    )

    confirm_score_btn.click(
        fn=_process_single_score_edit,  # 使用内部定义的 _process_single_score_edit
        inputs=[current_score_input, current_import_data_state],
        outputs=[import_score_dataframe, current_import_data_state]
    )

    current_score_input.submit(
        fn=_process_single_score_edit,  # 使用内部定义的 _process_single_score_edit
        inputs=[current_score_input, current_import_data_state],
        outputs=[import_score_dataframe, current_import_data_state]
    )

    # DataFrame 内容改变时，更新 state
    # 注意：这里只更新 import_data_state，不自动保存到文件
    import_score_dataframe.change(
        fn=lambda dataframe_data, state: (state[0], [row["分数"] for _, row in dataframe_data.iterrows()], state[2]),
        inputs=[import_score_dataframe, current_import_data_state],
        outputs=[current_import_data_state]
    )

    save_import_data_btn.click(
        fn=save_data_to_data_dir,
        inputs=[current_import_data_state],
        outputs=[import_status_text, current_managed_data_state]
    ).then(
        fn=load_training_data_for_management,
        inputs=[],
        # 这里的输出顺序要与 load_training_data_for_management 的返回值严格匹配
        outputs=[managed_score_dataframe, managed_data_status_text, current_managed_data_state, managed_gallery,
                 gr.State(current_managed_data_state.value[1])]
    ).then(  # 再次触发显示第一张图片
        fn=lambda all_image_paths, all_scores, index: _update_managed_display_from_index(index, all_image_paths,
                                                                                         all_scores),
        inputs=[gr.State(current_managed_data_state.value[0]), gr.State(current_managed_data_state.value[1]),
                gr.State(current_managed_data_state.value[2])],
        outputs=[managed_image_name_display, managed_image_preview, managed_score_input, delete_filename_input]
    )

    # 2. 训练数据管理 Tab 的事件
    load_managed_data_btn.click(
        fn=load_training_data_for_management,
        inputs=[],
        outputs=[managed_score_dataframe, managed_data_status_text, current_managed_data_state, managed_gallery,
                 gr.State(current_managed_data_state.value[1])]
    ).then(  # 触发显示第一张图片
        fn=lambda all_image_paths, all_scores, index: _update_managed_display_from_index(index, all_image_paths,
                                                                                         all_scores),
        inputs=[gr.State(current_managed_data_state.value[0]), gr.State(current_managed_data_state.value[1]),
                gr.State(current_managed_data_state.value[2])],
        outputs=[managed_image_name_display, managed_image_preview, managed_score_input, delete_filename_input]
    )

    managed_gallery.select(
        fn=_select_managed_image_for_edit,
        inputs=[current_managed_data_state],
        outputs=[managed_image_name_display, managed_image_preview, managed_score_input, delete_filename_input,
                 current_managed_data_state]
    )

    prev_managed_btn.click(
        fn=_navigate_managed_image,
        inputs=[gr.State(-1), current_managed_data_state],
        outputs=[managed_image_name_display, managed_image_preview, managed_score_input, delete_filename_input,
                 current_managed_data_state]
    )
    next_managed_btn.click(
        fn=_navigate_managed_image,
        inputs=[gr.State(1), current_managed_data_state],
        outputs=[managed_image_name_display, managed_image_preview, managed_score_input, delete_filename_input,
                 current_managed_data_state]
    )

    confirm_managed_score_btn.click(
        fn=_process_managed_single_score_edit,
        inputs=[managed_score_input, current_managed_data_state],
        outputs=[managed_data_status_text, managed_score_dataframe, managed_gallery,
                 gr.State(current_managed_data_state.value[1]), current_managed_data_state]
    )
    managed_score_input.submit(
        fn=_process_managed_single_score_edit,
        inputs=[managed_score_input, current_managed_data_state],
        outputs=[managed_data_status_text, managed_score_dataframe, managed_gallery,
                 gr.State(current_managed_data_state.value[1]), current_managed_data_state]
    )

    # Dataframe直接编辑更新
    managed_score_dataframe.change(
        fn=update_data_from_management_dataframe,
        inputs=[managed_score_dataframe, current_managed_data_state],
        outputs=[managed_gallery, gr.State(current_managed_data_state.value[1]), current_managed_data_state]
    ).success(
        fn=lambda: "表格更新并保存成功！",  # 这里不再需要dataframe_data和state参数，因为update_data_from_management_dataframe已经处理了
        inputs=[],  # 确保没有多余的inputs
        outputs=[managed_data_status_text]
    )

    # 添加条目
    add_entry_btn.click(
        fn=add_new_image_entry,
        inputs=[new_image_file_input, new_image_name_input, new_score_input, current_managed_data_state],
        outputs=[managed_data_status_text, managed_score_dataframe, new_image_file_input, managed_gallery,
                 gr.State(current_managed_data_state.value[1]), current_managed_data_state]
    ).then(  # 添加后刷新画廊显示
        fn=lambda all_image_paths, all_scores, index: _update_managed_display_from_index(index, all_image_paths,
                                                                                         all_scores),
        inputs=[gr.State(current_managed_data_state.value[0]), gr.State(current_managed_data_state.value[1]),
                gr.State(current_managed_data_state.value[2])],
        outputs=[managed_image_name_display, managed_image_preview, managed_score_input, delete_filename_input]
    )

    # 删除条目
    delete_entry_btn.click(
        fn=delete_image_entry,
        inputs=[delete_filename_input, current_managed_data_state],
        outputs=[managed_data_status_text, managed_score_dataframe, delete_filename_input, managed_gallery,
                 gr.State(current_managed_data_state.value[1]), current_managed_data_state]
    ).then(  # 删除后刷新画廊显示
        fn=lambda all_image_paths, all_scores, index: _update_managed_display_from_index(index, all_image_paths,
                                                                                         all_scores),
        inputs=[gr.State(current_managed_data_state.value[0]), gr.State(current_managed_data_state.value[1]),
                gr.State(current_managed_data_state.value[2])],
        outputs=[managed_image_name_display, managed_image_preview, managed_score_input, delete_filename_input]
    )

    # 3. 训练模型 Tab 的事件
    model_type_selector.change(
        lambda model_type: gr.update(visible=model_type in ["深度学习", "端到端深度学习"]),
        inputs=[model_type_selector],
        outputs=[base_cnn_selector_train]
    )

    data_prep_success_flag_state = gr.State(False)

    train_start_btn.click(
        fn=training_engine.switch_model_type,
        inputs=[model_type_selector, base_cnn_selector_train],
        outputs=[train_status_text]
    ).then(
        fn=training_engine.prepare_data_for_training,
        inputs=[],
        outputs=[data_prep_success_flag_state, train_status_text]
    ).then(
        fn=lambda success_flag, epochs, lr: (
            training_engine.train_model(epochs, lr)
            if success_flag
            else _create_static_error_plot("数据准备失败或无数据，无法训练")
        ),
        inputs=[data_prep_success_flag_state, epochs_input, lr_input],
        outputs=[loss_plot_output]
    )

    # 4. 预测图片 Tab 的事件
    predict_model_type_selector.change(
        lambda model_type: gr.update(visible=model_type in ["深度学习", "端到端深度学习"]),
        inputs=[predict_model_type_selector],
        outputs=[base_cnn_selector_predict]
    )

    predict_btn.click(
        fn=training_engine.predict_score,
        inputs=[image_for_predict, predict_model_type_selector, base_cnn_selector_predict],
        outputs=predicted_score_output
    )

# 启动 Gradio 应用
if __name__ == "__main__":
    Path(DATA_DIR).mkdir(exist_ok=True, parents=True)
    Path(MODEL_SAVE_BASE_PATH).parent.mkdir(exist_ok=True, parents=True)

    demo.launch(
        share=False,
        server_port=7860,
        show_error=True,
        debug=True
    )

