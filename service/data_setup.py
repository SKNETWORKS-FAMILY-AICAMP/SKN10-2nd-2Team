import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def data_loader():
  '''
  입력변수 : root, filename
  root에서 csv파일을 불러와서 DataFrame형태로 return하는 함수
  '''
  pass

def data_preprocessing():
  '''
  입력변수 : data(DataFrame)
  1. data를 입력받아 전처리해줌
  2. return data
  '''
  pass

def input_mode():
  '''
  입력변수 : data(DataFrame), train_mode(boolean)
  1. train_mode가 True일 경우
    1) X(feature), y(target)을 나누어 줌
    2) train_test_split을 이용하여 7:3으로 나누어줌
    3) 이후 data_preprocessing을 실행하여 전처리해줌
    4) return X_train, X_test, y_train, y_test 
  2. train_mode가 False일 경우 data_preprocessing 실행
    - 사용처 : streamlit에서 입력받았을 때 사용
  '''