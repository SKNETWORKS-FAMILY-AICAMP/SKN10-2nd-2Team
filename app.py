import streamlit as st
import pandas as pd

# 세팅
st.set_page_config(
    page_title = "가입 고객 이탈 예측",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

def main():
  st.title('메인 페이지 제목')
  st.divider()

if __name__ == '__main__':
  main()