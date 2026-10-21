"""Streamlit demo: upload a blood-smear image -> annotated detections + cell counts.

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bloodcell.predict import count_cells  # noqa: E402
from bloodcell.preprocess import stage_visualisation  # noqa: E402

st.set_page_config(page_title="Blood Cell Detection & Counting", page_icon="🩸",
                   layout="wide")

st.title("🩸 Blood Cell Detection & Counting")
st.caption("YOLOv8 detector + OpenCV preprocessing · BCCD dataset · "
           "RBC / WBC / Platelets")

with st.sidebar:
    st.header("Settings")
    weights = st.text_input("Weights (.pt)", value="runs/detect/train/weights/best.pt")
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    use_pre = st.checkbox("Apply OpenCV preprocessing", value=True)
    st.markdown("---")
    st.markdown("Upload a stained peripheral-blood smear image "
                "(jpg/png). The model localises and counts each cell type.")


@st.cache_resource
def get_model(w: str):
    from ultralytics import YOLO

    return YOLO(w)


uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded is not None:
    data = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input")
        st.image(img[:, :, ::-1], use_container_width=True)

    if not os.path.exists(weights):
        st.warning(f"Weights not found at `{weights}`. Train first or point the "
                   "sidebar at a valid checkpoint.")
    else:
        model = get_model(weights)
        annotated, counts = count_cells(model, img, conf=conf, preprocess=use_pre)
        with col2:
            st.subheader("Detections")
            st.image(annotated[:, :, ::-1], use_container_width=True)

        st.subheader("Cell counts")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RBC", counts["RBC"])
        c2.metric("WBC", counts["WBC"])
        c3.metric("Platelets", counts["Platelets"])
        c4.metric("Total", sum(counts.values()))

    with st.expander("Preprocessing stages"):
        stages = stage_visualisation(img)
        cols = st.columns(len(stages))
        for col, (name, im) in zip(cols, stages.items()):
            col.image(im[:, :, ::-1], caption=name, use_container_width=True)
else:
    st.info("Upload an image to begin.")
