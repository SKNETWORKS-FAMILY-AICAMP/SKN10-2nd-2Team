# SKN10-2nd-2Team
# 통신사 고객 이탈 예측 모델 개발 
<img width="1066" alt="Screenshot 2025-03-04 at 2 13 13 PM" src="https://github.com/user-attachments/assets/a9ea819c-1998-4d74-bb36-6b3cc5f977f9" />
<img width="1067" alt="Screenshot 2025-03-04 at 2 13 59 PM" src="https://github.com/user-attachments/assets/b3fda6ad-b002-4844-b92c-5b6de9bba23d" />

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
    <td><img src="https://github.com/user-attachments/assets/1d56cc60-e0d5-401b-b365-3f38f25bed43" width="175" height="175"></td>
    <td><img src="https://github.com/user-attachments/assets/1d56cc60-e0d5-401b-b365-3f38f25bed43" width="175" height="175"></td>
    <td><img src="https://github.com/user-attachments/assets/1d56cc60-e0d5-401b-b365-3f38f25bed43" width="175" height="175"></td>
    <td><img src="https://github.com/user-attachments/assets/1d56cc60-e0d5-401b-b365-3f38f25bed43" width="175" height="175"></td>
    <td><img src="https://github.com/user-attachments/assets/1d56cc60-e0d5-401b-b365-3f38f25bed43" width="175" height="175"></td>
    <td><img src="https://github.com/user-attachments/assets/1d56cc60-e0d5-401b-b365-3f38f25bed43" width="175" height="175"></td>


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
5. 기술 스택 및 향후 계획

---

## 1. 데이터 분석 및 전처리

### 1.1 데이터셋 개요
| 항목 | 내용 |
|------|------|
| 총 고객 수 | 7,043명 |
| 이탈률 | 26.54% |
| 고객 유지 기간 | 0-72개월 |
| 결측치 | TotalCharges 11건 (0.16%) |
| 특성 수 | 원본 21개 → 최종 6개 |


### 1.2 주요 이탈 요인 분석
| 요인 | 세부 내용 | 이탈률 | 전체 비중 |
|------|-----------|---------|------------|
| 계약 유형 | Month-to-month<br>One-year<br>Two-year | 42.7%<br>11.3%<br>2.8% | 55.0%<br>20.9%<br>24.1% |
| 서비스 유형 | Fiber optic<br>DSL<br>미사용 | 41.9%<br>19.0%<br>7.4% | 44.0%<br>34.4%<br>21.6% |
| 결제 방식 | Electronic check<br>Mailed check<br>Bank transfer<br>Credit card | 45.3%<br>19.1%<br>16.7%<br>15.2% | 33.6%<br>23.2%<br>21.8%<br>21.4% |

## 인구통계학적 특성 별 이탈율
![image](https://github.com/user-attachments/assets/f3702d32-cc05-49c1-b22d-ba7c9fa9b001)

## 서비스 별 이탈율
![image](https://github.com/user-attachments/assets/b46f8c03-a29b-4ead-b4fd-15f8ee27f7b9)

## 주요 요인별 이탈률 비교 및 분석
![image](https://github.com/user-attachments/assets/d65a54b5-efd6-4293-9c60-ef03c9ff2c32)

### 1.3 결측치 처리

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
| ver5 | 0.7785 | 0.6967 | 0.5560 | 인터넷 서비스 가입 수 추가 |
| ver6 | 0.7747 | 0.6936 | 0.5509 | 전화, 인터넷 서비스 가입 수로 변경 |
| ver7 | 0.7785 | 0.7041 | 0.5667 | 서비스 가입 수 제거 및 요금 관련 변수 추가 |


![image](https://github.com/user-attachments/assets/176d0982-9e4e-4517-86d7-924d493dd37d)

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


















## 5. 기술 스택 및 향후 계획

### 5.1 개발 환경
| 분야 | 도구 | 버전 | 용도 |
|------|------|------|------|
| 언어 | Python | 3.8+ | 전체 개발 |
| ML | scikit-learn | 0.24.2 | 모델 개발 |
| | imbalanced-learn | 0.8.1 | SMOTE 적용 |
| 데이터 처리 | pandas | 1.3.3 | 데이터 전처리 |
| | numpy | 1.21.2 | 수치 연산 |
| 시각화 | matplotlib | 3.4.3 | 기본 시각화 |
| | seaborn | 0.11.2 | 고급 시각화 |

![개발 파이프라인](./images/development_pipeline.png)
> 전체 개발 프로세스 및 도구 활용 흐름도

### 5.2 향후 개선 로드맵
| 단계 | 계획 | 우선순위 | 예상 효과 |
|------|------|----------|------------|
| 1단계 | 앙상블 모델 도입 | 높음 | 성능 향상 |
| | 시계열 특성 추가 | 높음 | 예측력 개선 |
| 2단계 | 실시간 예측 시스템 | 중간 | 운영 효율화 |
| | 자동 모니터링 | 중간 | 안정성 확보 |
| 3단계 | 딥러닝 모델 검토 | 낮음 | 잠재력 탐색 |

![개선 로드맵](./images/improvement_roadmap.png)
> 단계별 개선 계획 및 기대 효과

---


## 한 줄 회고
- 김현수:
- 박현준:
- 전서빈:
- 정소열:
- 조현정:
- 신민주: 좋은 팀을 만나서 해보고싶은 역할을 담당할 수 있었고 전처리부터 모델링 과정까지 경험할 수 있어서 좋았습니다.
