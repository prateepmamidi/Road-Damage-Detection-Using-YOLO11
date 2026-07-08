
# Streamlit app for LoveDA SegFormer
import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
import plotly.express as px
from transformers import SegformerForSemanticSegmentation

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH="segformer_best.pth"
IMAGE_SIZE=512
NUM_CLASSES=8
CLASS_NAMES=["Background","Building","Road","Water","Barren","Forest","Agriculture","Unknown"]
COLORS=np.array([[0,0,0],[255,0,0],[255,255,0],[0,0,255],[139,69,19],[0,255,0],[255,165,0],[255,255,255]],dtype=np.uint8)
MEAN=np.array([0.485,0.456,0.406],dtype=np.float32)
STD=np.array([0.229,0.224,0.225],dtype=np.float32)

@st.cache_resource
def load_model():
    m=SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True)
    m.load_state_dict(torch.load(MODEL_PATH,map_location=DEVICE))
    m.to(DEVICE).eval()
    return m

model=load_model()

def preprocess(img):
    img=img.convert("RGB").resize((IMAGE_SIZE,IMAGE_SIZE))
    x=np.asarray(img).astype(np.float32)/255.0
    x=(x-MEAN)/STD
    x=x.transpose(2,0,1)
    return torch.from_numpy(x).unsqueeze(0).to(DEVICE)

@torch.no_grad()
def predict(img):
    out=model(preprocess(img)).logits
    out=F.interpolate(out,size=(IMAGE_SIZE,IMAGE_SIZE),mode="bilinear",align_corners=False)
    return out.argmax(1).squeeze().cpu().numpy()

def to_png(arr):
    bio=BytesIO()
    Image.fromarray(arr).save(bio,format="PNG")
    return bio.getvalue()

st.set_page_config(page_title="LoveDA SegFormer",layout="wide")
st.title("🌍 LoveDA Land Cover Segmentation")
up=st.file_uploader("Upload image",type=["png","jpg","jpeg","tif","tiff"])
if up:
    img=Image.open(up)
    pred=predict(img)
    mask=COLORS[pred]
    base=np.asarray(img.resize((IMAGE_SIZE,IMAGE_SIZE)))
    over=((0.6*base)+(0.4*mask)).astype(np.uint8)
    c1,c2,c3=st.columns(3)
    c1.image(img,caption="Original",use_container_width=True)
    c2.image(mask,caption="Segmentation",use_container_width=True)
    c3.image(over,caption="Overlay",use_container_width=True)
    total=pred.size
    df=pd.DataFrame({
        "Class":CLASS_NAMES,
        "Coverage (%)":[round((pred==i).sum()*100/total,2) for i in range(NUM_CLASSES)]
    })
    st.dataframe(df,use_container_width=True)
    st.plotly_chart(px.pie(df,names="Class",values="Coverage (%)"),use_container_width=True)
    st.download_button("Download Mask",to_png(mask),"segmentation_mask.png","image/png")
    st.download_button("Download Overlay",to_png(over),"overlay.png","image/png")
