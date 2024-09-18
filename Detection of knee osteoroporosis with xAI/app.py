import os
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from configs.config import BaseConfig
from configs.constants import ACTIVATIONS, CAM_METHODS, CONFIGS, MODELS
from model import build_model
from PIL import Image
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.transforms import v2
from utils import utils


def on_selected_model_change() -> None:
    global prev_selected_model
    if selected_model != prev_selected_model:
        prev_selected_model = selected_model
        init_model.clear()


@st.cache_resource
def init_model(
    selected_config: str | None, selected_weights: str | None
) -> tuple[nn.Module, BaseConfig]:
    if selected_config is None or selected_weights is None:
        st.error("The selected config or weights cannot be empty", icon="🚨")
        st.stop()

    config = CONFIGS[selected_model].from_yaml_file(file=selected_config)
    class_count = len(config.label_map_legend)

    utils.seed_everything(config.seed)

    model = build_model(
        class_count=class_count,
        model=MODELS[selected_model],
        hidden_size=config.hidden_size,
        dropout=config.dropout,
        activation=ACTIVATIONS[config.activation],
        trainable_model=config.trainable_model,
    )
    model.load_state_dict(
        torch.load(f"./weights/{selected_model}/best_model.pth", weights_only=True)
    )

    return model, config


@st.cache_data(ttl=120)
def predict(
    _model: nn.Module,
    _image: Image.Image,
    _transform: v2.Compose,
    *,
    device: torch.device,
    xai_method: BaseCAM | None = None,
) -> torch.Tensor | tuple[torch.Tensor, Image.Image]:
    model = _model.to(device)

    rgb_img = np.float32(_image.convert("RGB")) / 255
    tensor_image = _transform(_image)
    tensor_image = tensor_image.unsqueeze(0)
    tensor_image = tensor_image.to(device)

    if xai_method:
        target_layers = utils.get_target_layers(model, model.name)

        with CAM_METHODS[xai_method](
            model=model,
            target_layers=target_layers,
        ) as cam:
            grayscale_cam = cam(
                input_tensor=tensor_image, targets=None, aug_smooth=True
            )
            grayscale_cam = grayscale_cam[0, :]

            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        return torch.softmax(cam.outputs, dim=1), Image.fromarray(
            cam_image.astype("uint8"), "RGB"
        )

    output = model(tensor_image)

    return torch.softmax(output, dim=1)


@st.fragment
def download_image(byte_img: bytes):
    st.download_button(
        label="Download image",
        data=byte_img,
        file_name="xray_with_xai.png",
        mime="image/png",
    )


if __name__ == "__main__":
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    prev_selected_model = None
    predictions = None
    config = None

    st.set_page_config(layout="wide", page_title="Detection of knee osteoporosis")

    hide_img_fs = """
    <style>
    button[title="View fullscreen"]{
        visibility: hidden;}
    </style>
    """

    st.markdown(hide_img_fs, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.title("Detection of knee osteoporosis")

        selected_model = st.selectbox(
            label="Select a model",
            options=MODELS,
            index=3,
            on_change=on_selected_model_change,
        )

        selected_config = st.selectbox(
            label="Select a config",
            options=Path(f"./configs/{selected_model}").resolve().rglob("*.yaml"),
            format_func=lambda path: path.name,
            index=0,
            help="Select the config to use for the model. "
            + "The config should be in the '.yaml' format in the 'configs/{model_name}' directory.",
        )

        selected_weights = st.selectbox(
            label="Select weights",
            options=Path(f"./weights/{selected_model}").resolve().rglob("*.pth"),
            format_func=lambda path: path.name,
            index=0,
            help="Select the weights to use for the model. "
            + "The weights should be in the '.pth' format in the 'weights/{model_name}' directory.",
        )

        if os.path.exists("./dataset/knee-osteoarthritis/"):
            use_test_data = st.checkbox(
                "Image from test dataset",
                value=False,
                help="Use a random image from the test dataset to predict the presence of osteoporosis. "
                + "The image will be selected randomly from the 'dataset/knee-osteoarthritis/test' directory.",
            )
        else:
            use_test_data = False

        if not use_test_data:
            uploaded_img = st.file_uploader(
                label="Upload an image of a knee X-ray to predict the presence of osteoporosis",
                type=["jpg", "jpeg", "png"],
                disabled=use_test_data,
                label_visibility="collapsed",
            )
        else:
            uploaded_img = utils.get_image_from_test_dataset()

        st.subheader("Options")

        use_gpu_mps = st.checkbox(
            "Use GPU/MPS",
            help="Use the GPU or MPS (Apple Silicon) to speed up the prediction process (if you have a compatible device).",
        )

        show_confidence = st.checkbox(
            "Show confidence",
            value=True,
            help="Show the confidence of the model in the prediction as a bar chart.",
        )

        use_xai = st.checkbox(
            "Use xAI",
            help="Use explainable AI to explain the prediction. "
            + "This will slow down the prediction process, but it will provide insights into the model's decision-making process. ",
        )

        if use_xai:
            xai_method = st.selectbox(
                label="Select xAI method",
                options=CAM_METHODS,
                index=len(CAM_METHODS) - 1,
            )
            st.info(
                "The xAI feature is experimental and may not work as expected. "
                + "Currently, only the ConvNeXt model is not supported for xAI. ",
                icon="ℹ️",
            )
        else:
            xai_method = None

        if xai_method in ["AblationCAM", "ScoreCAM"] and not use_gpu_mps:
            st.warning(
                f"Without using the GPU/MPS, the xAI methods '{xai_method}' may take a long time to process. "
                + "Consider enabling the 'Use GPU/MPS' option to speed up the process. ",
                icon="⚠️",
            )

        if uploaded_img:
            image = Image.open(uploaded_img)

        device = utils.get_device() if use_gpu_mps else torch.device("cpu")

        if st.button("Predict", disabled=not uploaded_img):
            model, config = init_model(selected_config, selected_weights)
            model.eval()

            if use_xai:
                # print(model)
                predictions, image = predict(
                    model, image, transform, device=device, xai_method=xai_method
                )
            else:
                predictions = predict(model, image, transform, device=device)

    with col2:
        if predictions is not None:
            st.title("Model prediction")

            predictions = dict(
                zip(
                    config.label_map_legend.values(),
                    (predictions * 100).squeeze().cpu().detach().numpy().round(2),
                )
            )

            actual_class = (
                config.label_map_legend[
                    config.label_map.get(int(uploaded_img.parts[-2]), 0)
                ]
                if use_test_data
                else "Unknown"
            )

            st.info(
                f"Predicted class: {max(predictions, key=predictions.get)} | "
                + f"Actual class: {actual_class}",
                icon="🔍",
            )

            if show_confidence:
                st.bar_chart(
                    predictions,
                    stack=False,
                    horizontal=True,
                    height=300,
                )

            with st.columns(3)[1]:
                st.subheader("Knee X-ray")

                st.image(
                    image,
                    width=300,
                )

                if use_xai:
                    buf = BytesIO()
                    image.save(buf, format="PNG")
                    byte_img = buf.getvalue()

                    download_image(byte_img)
