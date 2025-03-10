# 통신사 고객 이탈 예측 모델 개발 


## 프로젝트 개요
통신사 고객 데이터를 활용하여 고객 이탈을 예측하고, 이탈 가능성이 높은 고객을 조기에 식별하여 선제적인 대응을 가능하게 하는 예측 모델 개발

## 프로젝트 목표
- 고객 이탈 예측 모델 개발 및 성능 최적화
- 주요 이탈 요인 분석 및 인사이트 도출
- 실시간 이탈 예측이 가능한 웹 애플리케이션 개발


# 

## 팀 소개
<table>
  <tr>
    <th>김현수</th>
    <th>박현준</th>
    <th>정소열</th>
    <th>신민주</th>
    <th>조현정</th>
    <th>전서빈</th>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/30860b88-995a-4320-a88e-4371632b9334" width="175" height="175"></td>
    <td><img src="https://github.com/user-attachments/assets/27188fe4-37ae-4287-a1cf-f3b5cff6ab87" width="175" height="175"></td>
    <td><img src="https://github.com/user-attachments/assets/534cc407-97e6-4999-ac57-86d7f7d3c699" width="175" height="175"></td>
    <td><img src="https://github.com/user-attachments/assets/3ac03c06-6074-47b4-bc72-d8f82fe2dab4" width="175" height="175"></td>
    <td><img src="https://github.com/user-attachments/assets/e714bc2a-27fe-4128-8bf9-d48976374374" width="175" height="175"></td>
    <td><img src="https://github.com/user-attachments/assets/51354b21-4396-4dfa-91e9-04deb17f9792" width="175" height="175"></td>

  </tr>
  <tr>
    <th>팀장</th>
    <th>팀원</th>
    <th>팀원</th>
    <th>팀원</th>
    <th>팀원</th>
    <th>팀원</th>
  </tr>
  <tr>
    <th> 
      <b>프로젝트 총괄</b><br>
      <b>데이터 분석</b><br>
      <b>모델 평가</b>
    </th>
    <th>
      <b>RF 모델 개발</b><br>
      <b>화면 개발</b><br>
      <b>GitHub 업데이트</b>
    </th>
    <th>
      <b>XGBoost 모델 개발</b><br>
      <b>성능 분석</b><br>
      <b>화면 설계</b>
    </th>
    <th>
      <b>ExtraTree 모델 개발</b><br>
      <b>성능 분석</b><br>
      <b>모듈화</b>
    </th>
    <th>
      <b>LightGBM 모델 개발</b><br>
      <b>성능 분석</b><br>
      <b>화면 개발</b>
    </th>
    <th>
      <b>SVM 모델 개발</b><br>
      <b>성능 분석</b><br>
      <b>화면 개발</b>
    </th>
  </tr>
</table>


---



## 목차
1. 데이터 분석 및 전처리
2. 특성 엔지니어링
3. 모델 개발 과정
4. 결과 및 시사점

---

## 1. 데이터 분석 및 전처리

### 1.1 데이터셋 개요
| 항목 | 내용 |
|------|------|
| 총 고객 수 | 7,043명 |
| 이탈률 | 26.54% |
| 고객 유지 기간 | 0-72개월 |
| 결측치 | TotalCharges 11건 (0.16%) |
| 특성 수 | 원본 21개 → 최종 27개 |


### 1.2 주요 이탈 요인 분석
| 요인 | 세부 내용 | 이탈률 | 전체 비중 |
|------|-----------|---------|------------|
| 계약 유형 | Month-to-month<br>One-year<br>Two-year | 42.7%<br>11.3%<br>2.8% | 55.0%<br>20.9%<br>24.1% |
| 서비스 유형 | Fiber optic<br>DSL<br>미사용 | 41.9%<br>19.0%<br>7.4% | 44.0%<br>34.4%<br>21.6% |
| 결제 방식 | Electronic check<br>Mailed check<br>Bank transfer<br>Credit card | 45.3%<br>19.1%<br>16.7%<br>15.2% | 33.6%<br>23.2%<br>21.8%<br>21.4% |

✅ 계약 유형(Contract)에 따른 이탈률
- 월 단위 계약 고객의 이탈률이 매우 높음
- 1년, 2년 계약 고객은 이탈 가능성이 낮음
➡ 장기 계약 유도 필요

✅ 인터넷 서비스 유형(InternetService)
- Fiber optic(광 인터넷) 사용자가 DSL보다 이탈률이 높음

✅ 결제 방식(PaymentMethod)과 이탈률
- 전자수표(Electronic check) 사용자의 이탈률이 가장 높음
- 자동이체(Credit Card, Bank Transfer) 사용자들은 이탈률 낮음
➡ 자동이체 가입 유도 필요
#### 인구통계학적 특성 별 이탈 현황
![image](https://github.com/user-attachments/assets/f3702d32-cc05-49c1-b22d-ba7c9fa9b001)
- SeniorCitizen (고령자): (고령일수록 이탈 가능성 조금 높음)
#### 서비스 별 이탈 현황
![image](https://github.com/user-attachments/assets/b46f8c03-a29b-4ead-b4fd-15f8ee27f7b9)

#### 주요 요인별 이탈 현황 비교 및 분석
![image](https://github.com/user-attachments/assets/d65a54b5-efd6-4293-9c60-ef03c9ff2c32)
✅ 가입 기간(tenure)
- 가입 기간이 짧을수록 이탈률이 높음
- 10개월 미만 가입자 중 상당수가 이탈
- **장기 계약자(2년)**의 이탈률이 낮음

✅ 월 청구 요금(MonthlyCharges)과 총 청구 금액(TotalCharges)
- 월 요금이 높을수록 이탈 가능성 증가
- 하지만 총 청구 금액이 많을수록(오랜 가입자일수록) 이탈률 감소
### 1.3 데이터 전처리
- 불필요한 컬럼 제거 (customerID, gender)
- 결측치 처리 (TotalCharges - 중앙값 대체)
- tenure 그룹화
- 요금 관련 변수 추가(ChargesRatio, AverageMonthlyCharge, ChargeChange)
- 가족 여부 추가
- 자동 결제 여부 추가
- One Hot Encoidng
- 이진 변수 처리
- 서비스 관련 변수 합산
- 타겟 변수 변환
~~~python
TotalCharges 결측치 처리 -> 중앙값으로 대체
median_total_charges = df['TotalCharges'].median()
df['TotalCharges'].fillna(median_total_charges, inplace=True)
~~~


---

## 2. 특성 엔지니어링

<img width="1190" alt="Screenshot 2025-03-04 at 5 14 15 PM" src="https://github.com/user-attachments/assets/8cf5e836-2b0d-41ee-9169-33d17b943ffe" />

> Random Forest 기반 특성 중요도 시각화

---

## 3. 모델 개발 과정

### 3.1 모델 별 성능 비교
| 모델 | 정확도 | ROC-AUC | 이탈고객<br>Recall | 이탈고객<br>Precision | F1-score |
|---------------|--------|--------|--------|--------|--------|
| Random Forest | 0.7818 | 0.8303 | 0.6328 | 0.5820 | 0.6063 |
| XGBoost | 0.7766 | 0.8228 | 0.5330 | 0.5874 | 0.5589 | 
| LightGBM | 0.7738 | 0.8185| 0.5740 | 0.5740 | 0.5740 | 
| SVM | 0.74 | 0.8017 | 0.64 | 0.51 | 0.57 | 0.0934 |
| Extra Tree | 0.7733 | 0.8252 | 0.6327 | 0.5652 | 0.5971 | 
| CatBoost | 0.7841 | 0.7170 | 0.5739 | 0.5974 | 0.5854 |

![image](https://github.com/user-attachments/assets/882746cc-75d7-4428-9af2-9d0093d14bad)
![image](https://github.com/user-attachments/assets/7333af5c-b1d9-4760-acf2-c64487fef859)


### 3.2 모델 발전 과정

### Random Forest 모델 버전별 성능 비교

| 버전 | 정확도 | ROC-AUC | F1-score | 주요 변경사항 |
|------|---------|----------|-----------|--------------|
| base | 0.7903 | 0.6911 | 0.5484 | 기본 모델 |
| ver1 | 0.7747 | 0.6958 | 0.5543 | SMOTE를 이용한 데이터 불균형 해결 |
| ver2 | 0.7814 | 0.7032 | 0.5658 | Label Encoding → One-Hot Encoding 변경 |
| ver3 | 0.7804 | 0.7003 | 0.5614 | 가입기간 그룹화 및 자동 결제 여부 추가 |
| ver4 | 0.7776 | 0.6995 | 0.5599 | 가족여부 추가 |
| ver5 | 0.7785 | 0.6967 | 0.5560 | 전화, 인터넷 서비스 가입 수 추가 |
| ver6 | 0.7747 | 0.6936 | 0.5509 | 인터넷 서비스 가입 수로 변경 |
| ver7 | 0.7785 | 0.7041 | 0.5667 | 전화, 인터넷 서비스 가입 수로 되돌리기 및 요금 관련 변수 추가 |


![image](https://github.com/user-attachments/assets/176d0982-9e4e-4517-86d7-924d493dd37d)

---


### 모델을 이용한 이탈 시점 예측

### 인자값 받기
<img width="1317" alt="Screenshot 2025-03-05 at 9 53 16 AM" src="https://github.com/user-attachments/assets/618ef5a9-5c4d-4e0d-a74e-0d229cd06f40"/>

### 모델을 활용한 이탈예측
<img width="1316" alt="Screenshot 2025-03-05 at 9 53 33 AM" 
     src="https://github.com/user-attachments/assets/d95c5152-a2d4-43b4-baae-f3b648cf5896"  
     style="border: 3px solid black; border-radius: 5px;">

<table>
  <tr>
    <th>개월 수</th>
    <th>0개월</th>
    <th>1개월</th>
    <th>2개월</th>
    <th>3개월</th>
    <th>4개월</th>
    <th>5개월</th>
  </tr>
   <tr>
    <th>모델 예측 값</th>
    <th>유지</th>
    <th>유지</th>
    <th>유지</th>
    <th>유지</th>
    <th>유지</th>
    <th>이탈</th>
  </tr>
<table/>

### 이탈 시점 예측

<img width="1315" alt="Screenshot 2025-03-05 at 10 01 36 AM" 
     src="https://github.com/user-attachments/assets/8709ff0c-bf41-4a16-8af0-a4d7d86faa39"  
     style="border: 3px solid black; border-radius: 5px;">

---


## 4. 결과 및 시사점


## 모델 개발 결과 및 결론

### 1. 모델 성능 요약
- **최고 성능 모델**: Random Forest (v7)
  - 정확도: 0.7785
  - ROC-AUC: 0.7041
  - F1-score: 0.5667

### 5. 최종 결론
- ver7 모델 채택 (ROC-AUC와 F1-score 균형)
- 실제 적용 시 임계값 조정으로 정밀도/재현율 trade-off 고려

---


## 한 줄 회고
- 김현수: 제 의사소통 능력의 부족함으로 머릿속에 있는 내용에 대해서 팀원들에게 명확하게 전달하지 못하여 많은 혼선이 있었으나 팀원분들이 어떻게든 처리해주어서 완성할 수 있었습니다. 팀원분들 감사합니다.
- 박현준: 여러 머신러닝 모델 및 기법을 접목하여 고객 이탈률을 예측하여 보았습니다. 좋은 팀원들과 함께하며 많이 배웠습니다.
- 전서빈: 아직 부족한 지식으로 데이터 전처리와 모델을 돌리다보니 쉽지 않았지만 설계적인 부분에서 지식을 성장시켰습니다. 화면구현 또한 익숙하지 않은 작업이었지만 나중에 사용할 기술이라 생각하고 즐겁게 잘하였습니다. 팀원분들과 다같이 열심히하여 이번 프로젝트를 잘 마무리 할 수 있게 되었습니다.
- 정소열: 고객 이탈 예측 모델을 만들면서 EDA의 중요성을 알았습니다. 확실히 실제 프로젝트로 적용해보는 것이 직접적으로 머신러닝에 대해 더 자세히 알 수 있는 경험이었습니다. 팀원분들이 많이 배려해주시고 도와주셔서 잘 마칠 수 있었습니다.
- 조현정: 머신러닝에 대해 다양한 시도를 하면서 전처리와 특성 공학에 대해 알아갔던 시간 이었습니다. 화면 구현에 대해서도 조금 장벽이 내려간 것 같아 나름의 의미 있는 시간이었습니다. 팀원간 의사소통하는 법도 조금은 더 익숙해졌고 앞으로 어떻게 애자일을 해야하는 지에 대해 공부해볼 계기가 된 것 같습니다. 
- 신민주:  배우기만 했던 전처리부터 모델 학습까지 직접 응용해 볼 수 있어서 좋은 경험이었습니다. 생각처럼 진행되지 않아서 어려운 부분도 있었지만 팀원분들이 도와주셔서 잘 진행할 수 있었습니다.
