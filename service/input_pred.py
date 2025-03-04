'''
streamlit 페이지에서 데이터를 받아 저장된 모델을 이용하여 이탈 예측하기 위한 모듈
1. input으로 받아오는 것 : 임시로 dictioray 형태로 input을 받아오는거로 해두었고, input이 어떻게 되는가에 따라 수정할 것것
  - 연령(int)
  - 휴대폰 가입 여부(str : 'Yes' / 'No')
  - 인터넷 가입 여부(str : 'DSL', 'Fiber optic', 'No')
  - 휴대폰 및 인터넷 관련 서비스 (boolean)
2. 통계량(평균, 중위수)와 같은 값으로 기입해야할 것
  - MonthlyCharges : 계약기간/서비스 가입에 따른 평균값으로 대체
3. Feature Engineering
4. 입력 받아오고 tenure를 int형으로 0~60까지 넣어서 크기를 데이터 수를 71개로 만들어주고 예측
  - 이후 바뀌는 이탈자 발생하는 시점을 return해줌
추가사항
  - PaymentMethod : 입력으로 받아야할듯
  - PaperlessBilling : 입력으로 받아야할듯
'''
import numpy as np
import pandas as pd
from pathlib import Path

from service.data_setup import data_loader, data_preprocessing
from service.models import data_pred

df = pd.read_csv('../.data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

def __MonthlyCharges_Contract(data:pd.DataFrame = data_loader(), **kwargs):
  service_list = [
      'MultipleLines',
      'InternetService',
      'OnlineSecurity',
      'OnlineBackup',
      'DeviceProtection',
      'TechSupport',
      'StreamingTV',
      'StreamingMovies']
  # for col in service_list:
  #   data[col].fillna(0, inplace = True)
  # MonthlyCharges | Contract별 평균
  MonthlyCharges_df = pd.pivot_table(
    data = data,
    values = 'MonthlyCharges',
    index = service_list,
    columns = 'Contract',
    aggfunc = 'mean',
    ).mean(axis = 1)
  
  return MonthlyCharges_df

def __input_make_dataframe(input:dict) -> pd.DataFrame:
  # input(dict : 임시)을 통해 데이터 df의 각 column에 대한 값 채워넣을 것
  df = pd.DataFrame({
    'customID' : [0],
    'gender' : [0],
    'SeniorCitizen' : [0],
    'Partner' : [0],
    'Dependents' : [0],
    'tenure' : [0],
    'PhoneService' : [0],
    'MultipleLines' : [0],
    'InternetService' : [0],
    'OnlineSecurity' : [0],
    'OnlineBackup' : [0],
    'DeviceProtection' : [0],
    'TechSupport' : [0],
    'StreamingTV' : [0],
    'StreamingMovies' : [0],
    'Contract' : [0],
    'PaperlessBilling' : [0],
    'MonthlyCharges' : [0],
    'TotalCharges' : [0]
  })
  service_list = [
      'MultipleLines',
      'InternetService',
      'OnlineSecurity',
      'OnlineBackup',
      'DeviceProtection',
      'TechSupport',
      'StreamingTV',
      'StreamingMovies']
  df.loc[0, 'MonthlyCharges'] = __MonthlyCharges_Contract()[list(df.loc[0, service_list].values)]

  for r in range(0, 61):
    df.iloc[r, :] = df.iloc[0]
    df.loc[r, 'tenure'] = r
    df.loc[r, 'TotalCharges'] = r * df.loc[r, 'MonthlyCharges']

  return df

def __input_preprocessing(input:dict) -> pd.DataFrame:
  df = data_preprocessing(__input_make_dataframe(input = input))
  transpose_column = ['SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
      'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
      'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
      'MonthlyCharges', 'TotalCharges', 'tenure_group', 'ChargeChange',
      'Family', 'AutoPayment', 'InternetService_Fiber optic',
      'InternetService_No', 'Contract_One year', 'Contract_Two year',
      'PaymentMethod_Credit card (automatic)',
      'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
      'TotalServices']
  for col in transpose_column:
    if col not in df.columns:
      df[col] = 0
  df = df[transpose_column]
  return df

def tenure_predict(input:dict, root:Path = Path('/models'), model_name = 'randomforest'):
  pred = data_pred(data = __input_preprocessing(input = input), root = root, model_name = model_name)
  for month in range(len(pred) - 1):
    if pred[month] != pred[month + 1]:
      break
    else:
      month = 999
  if month != 999:
    return f'가입 시 {month}개월 뒤 이탈 예측'
  else:
    return f'가입 시 5년 내에는 이탈 예정 없음'