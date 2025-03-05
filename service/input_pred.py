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
import os
from pathlib import Path
import joblib

from service.data_setup import data_loader, data_preprocessing
from service.models import data_pred

# 서비스 및 계약 기간에 따른 월 요금 평균 계산하는 함수
def MonthlyCharges_Contract(data:pd.DataFrame = data_loader()) -> pd.pivot_table:
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

def input_preprocessing(data:pd.DataFrame) -> pd.DataFrame:
  service_list = [
      'MultipleLines',
      'InternetService',
      'OnlineSecurity',
      'OnlineBackup',
      'DeviceProtection',
      'TechSupport',
      'StreamingTV',
      'StreamingMovies']
  
  data['MonthlyCharges'] = MonthlyCharges_Contract()[tuple(data.loc[0, service_list].values)]
  data['TotalCharges'] = data['MonthlyCharges'] * data['tenure']

  return data
''' 수정
  # 입력 데이터에서 값 설정 (입력된 값만 덮어씀)
  # 추가 필드 설정
  if 'payment_method' in input:
    df.loc[0, 'PaymentMethod'] = str(input['payment_method'])
  
  if 'paperless_billing' in input:
    # Yes/No 형식으로 변환
    paperless_value = str(input['paperless_billing']).strip()
    if paperless_value.lower() in ['yes', 'true', '1', 'y']:
      df.loc[0, 'PaperlessBilling'] = 'Yes'
    elif paperless_value.lower() in ['no', 'false', '0', 'n']:
      df.loc[0, 'PaperlessBilling'] = 'No'
  
  # 나이를 SeniorCitizen으로 변환 (65세 이상이면 1, 아니면 0)
  if 'age' in input:
    try:
      # 문자열이나 다른 타입으로 들어온 경우 정수로 변환
      age_value = int(input['age'])
      df.loc[0, 'SeniorCitizen'] = 1 if age_value >= 65 else 0
    except (ValueError, TypeError):
      # 변환할 수 없는 경우 기본값 사용
      print(f"Warning: Invalid age value '{input['age']}', using default value")
      df.loc[0, 'SeniorCitizen'] = 0
  
  # 전화 서비스 설정
  if 'phone_subscription' in input:
    # Boolean 또는 문자열을 Yes/No로 변환
    phone_value = input['phone_subscription']
    if isinstance(phone_value, bool):
      df.loc[0, 'PhoneService'] = 'Yes' if phone_value else 'No'
    else:
      # 문자열이나 다른 값인 경우
      phone_str = str(phone_value).strip().lower()
      if phone_str in ['yes', 'true', '1', 'y']:
        df.loc[0, 'PhoneService'] = 'Yes'
      elif phone_str in ['no', 'false', '0', 'n']:
        df.loc[0, 'PhoneService'] = 'No'
  
  # 인터넷 서비스 설정
  if 'internet_type' in input:
    df.loc[0, 'InternetService'] = str(input['internet_type'])
  
  # 멀티라인 설정
  if 'multiple_lines' in input:
    df.loc[0, 'MultipleLines'] = str(input['multiple_lines'])
  
  # 서비스 설정
  if 'services' in input:
    # 문자열인 경우 리스트로 변환 (쉼표로 구분된 문자열 처리)
    services = input['services']
    if isinstance(services, str):
      services = [s.strip() for s in services.split(',')]
    
    if isinstance(services, list):
      for service in services:
        service_str = str(service).strip()
        if service_str in service_list:
          df.loc[0, service_str] = 'Yes'
  
  # Then calculate MonthlyCharges using the service values
  # Create a tuple of service values to use as an index
  service_values = tuple(df.loc[0, service_list].values)
  
  # Get the MonthlyCharges from the pivot table
  monthly_charges_data = MonthlyCharges_Contract()
  
  # Use a try-except block to handle potential KeyError
  try:
    df.loc[0, 'MonthlyCharges'] = monthly_charges_data.loc[service_values]
  except KeyError:
    # Fallback to a default value or mean if the exact combination doesn't exist
    df.loc[0, 'MonthlyCharges'] = monthly_charges_data.mean()

  # 먼저 DataFrame을 61개 행으로 확장합니다
  # 기존 DataFrame을 복사하여 61개 행으로 확장
  df_expanded = pd.DataFrame([df.iloc[0].tolist()] * 61, columns=df.columns)
  
  # 확장된 DataFrame에 tenure와 TotalCharges 값을 설정
  for r in range(0, 61):
    df_expanded.loc[r, 'tenure'] = r
    df_expanded.loc[r, 'TotalCharges'] = r * df_expanded.loc[r, 'MonthlyCharges']

  return df_expanded
  '''

  # transpose_column = ['SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
  #     'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
  #     'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
  #     'MonthlyCharges', 'TotalCharges', 'tenure_group', 'ChargeChange',
  #     'Family', 'AutoPayment', 'InternetService_Fiber optic',
  #     'InternetService_No', 'Contract_One year', 'Contract_Two year',
  #     'PaymentMethod_Credit card (automatic)',
  #     'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
  #     'TotalServices']

def fit_columns(data:pd.DataFrame, root:Path = Path('models'), model_name = 'rf.pkl'):
  model = joblib.load(root / model_name)
  model_columns = model.feature_names_in_
  for col in model_columns:
    if col not in data.columns:
      data[col] = 0
  data = data[model_columns]
  
  return data

def tenure_predict(data:pd.DataFrame, root:Path = Path('./models'), model_name = 'rf.pkl'):
  pred = data_pred(input = data, 
                  root = root, model_name = model_name)
  for month in range(len(pred) - 1):
    if pred[month] != pred[month + 1]:
      break
    else:
      month = 999
  if month != 999:
    return f'가입 시 {month}개월 뒤 이탈 예측'
  else:
    return f'가입 시 5년 내에는 이탈 예정 없음'