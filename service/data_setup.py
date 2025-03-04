import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def data_loader(root: str, filename: str) -> pd.DataFrame:
  '''
  입력변수 : root, filename
  root에서 csv파일을 불러와서 DataFrame형태로 return하는 함수
  '''
  file_path = os.path.join(root, filename)
  data = pd.read_csv(file_path, encoding='utf-8')
  return data

def data_preprocessing(data: pd.DataFrame) -> pd.DataFrame:
  '''
  입력변수 : data(DataFrame)
  1. data를 입력받아 전처리해줌
  2. return data
  '''
  # 1) 불필요한 컬럼 제거
  drop_cols = ['customerID', 'gender']
  for col in drop_cols:
    if col in data.columns:
      data.drop(columns=col, inplace=True)

  # 2) 결측치 처리 (TotalCharges -> numeric 변환 후 중앙값 대체)
  data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
  data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)

  # 3) tenure 그룹화
  data['tenure_group'] = pd.cut(
    data['tenure'],
    bins=[0, 12, 24, 48, 72, 1000],
    labels=[1, 2, 3, 4, 5],
    include_lowest=True
    )
  # 범위 밖(결측치 등)을 5로 채우기
  data['tenure_group'] = data['tenure_group'].astype(int).fillna('5')

  # 4) 요금 관련 변수 추가
  data['ChargesRatio'] = data['TotalCharges'] / (data['MonthlyCharges'] + 1)
  data['AverageMonthlyCharge'] = data['TotalCharges'] / (data['tenure'] + 1)
  data['ChargeChange'] = data['MonthlyCharges'] - data['AverageMonthlyCharge']

  # 5) 가족 여부 (Partner 또는 Dependents가 'Yes'이면 1)
  data['Family'] = ((data['Partner'] == 'Yes') | (data['Dependents'] == 'Yes')).astype(int)

  # 6) 자동 결제 여부 (결제 방식에 'check'가 들어있으면 0, 아니면 1)
  data['AutoPayment'] = data['PaymentMethod'].apply(lambda x: 0 if 'check' in x.lower() else 1)

  # 7) 범주형 변수 변환 (One Hot Encoding)
  data = pd.get_dummies(data, columns=['InternetService', 'Contract', 'PaymentMethod'], drop_first=True)

  # 8) 이진 변수(Yes/No) 처리
  binary_cols = [
    "Partner", "MultipleLines", "Dependents", "PhoneService",
    "PaperlessBilling", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
  for col in binary_cols:
    if col in data.columns:
      data[col] = data[col].map({'Yes': 1, 'No': 0})
      data[col].fillna(0, inplace=True)

  # 9) 서비스 관련 변수 합산 (MultipleLines, OnlineSecurity 등)
  # internet_service_cols = [
  #   "OnlineSecurity", "OnlineBackup",
  #   "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
  #   ]
  # data['TotalServices'] = data[internet_service_cols].sum(axis=1)

  # 10) 타겟 변수 변환 (Churn -> 0/1)
  if 'Churn' in data.columns:
    data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

  # 11) 사용 후 불필요한 컬럼 제거 (ChargesRatio, AverageMonthlyCharge)
  drop_cols_2 = ['ChargesRatio', 'AverageMonthlyCharge']
  for col in drop_cols_2:
    if col in data.columns:
      data.drop(columns=col, inplace=True)

    return data

def input_mode(data: pd.DataFrame, train_mode: bool = True, target = None):
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
  if train_mode:
        # 예: 'target' 컬럼을 타겟으로 가정 (실제 프로젝트에 맞게 수정)
        if target not in data.columns:
            raise ValueError(f"DataFrame에 {target} 컬럼이 존재하지 않습니다. 타겟 컬럼명을 확인하세요.")

        # X, y 분리
        X = data.drop(columns=[target])
        y = data[target]

        # 7:3으로 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y, shuffle=True
        )

        # 전처리(예: 결측치 처리 등)
        X_train = data_preprocessing(X_train)
        X_test = data_preprocessing(X_test)

        # (선택) 학습 데이터에 대해서만 오버샘플링 적용
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        return X_train_res, X_test, y_train_res, y_test

  else:
    # train_mode가 False면 전체 data에 대해 전처리만 수행 후 반환
    data_processed = data_preprocessing(data)
    return data_processed