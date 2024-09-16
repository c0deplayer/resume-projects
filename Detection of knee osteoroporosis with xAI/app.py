from io import BytesIO
import os
from pathlib import Path
import random

import streamlit as st
import torch
import torch.nn as nn
from configs.config import BaseConfig
from configs.constants import ACTIVATIONS, CONFIGS, MODELS
from model import build_model
from PIL import Image
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
    # device = utils.get_device()

    utils.seed_everything(config.seed)

    model, _ = build_model(
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


def predict(
    model: nn.Module, image: Image.Image, transform: v2.Compose, *, device: torch.device
) -> torch.Tensor:
    model.eval()
    model = model.to(device)

    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)

    output = model(image)

    return torch.softmax(output, dim=1)


@st.fragment
def download_image(byte_img: bytes):
    st.download_button(
        label="Download image",
        data=byte_img,
        file_name="xray_with_xai.png",
        mime="image/png",
    )


def get_image_from_test_dataset() -> Path:
    test_dataset_path = Path("./dataset/knee-osteoarthritis/test").resolve()

    return random.choice(list(test_dataset_path.rglob("*")))


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
            uploaded_img = get_image_from_test_dataset()

        st.subheader("Options")

        use_xai = st.checkbox(
            "Use xAI (WIP)",
            help="Use explainable AI to explain the prediction. "
            + "This will slow down the prediction process, but it will provide insights into the model's decision-making process. "
            + "This feature is a work in progress and do nothing at the moment.",
        )

        use_gpu_mps = st.checkbox(
            "Use GPU/MPS",
            help="Use the GPU or MPS (Apple Silicon) to speed up the prediction process.",
        )

        show_confidence = st.checkbox(
            "Show confidence",
            value=True,
            help="Show the confidence of the model in the prediction as a bar chart.",
        )

        if uploaded_img:
            image = Image.open(uploaded_img)

        device = utils.get_device() if use_gpu_mps else torch.device("cpu")

        if st.button("Predict", disabled=not uploaded_img):
            model, config = init_model(selected_config, selected_weights)
            model.eval()

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
