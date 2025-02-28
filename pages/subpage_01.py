import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

st.title("subpage_01")

# 각 탭에 주요 Feature이라 판단된 것들이 들어가며 이를 시각화, 모델의 F1-score / Accuracy / Confusion Matrix를 보여줌
tab1, tab2, tab3 = st.tabs(["Feature1", "Feature2", "Models"])

with tab1:
    st.header("Feature1")

with tab2:
    st.header("Feature2")

with tab3:
    st.header("Models")