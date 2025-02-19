# SKN10-2nd-2Team
# [가입 고객 이탈 예측](https://www.kaggle.com/code/bbksjdd/telco-customer-churn)



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
    <td><img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN07-2nd-3Team/blob/main/images/%EC%9D%B8%ED%98%95.png" width="175" height="175"></td>
    <td><img src="https://i.namu.wiki/i/yHMdZs8LhKDP0D0XmvNkWe4NplRU5BDyXiZNDk5BTOd9ON5KtykFiDO_Q7SDpQLA-q9Q4fyFKfzM3apcZnPGtg.webp" width="175" height="175"></td>
    <td><img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN07-2nd-3Team/blob/main/images/%EC%8A%A4%EB%85%B8%EC%9A%B0%EB%A7%A8.png" width="175" height="175"></td>
    <td><img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN07-2nd-3Team/blob/main/images/baby.jpg" width="175" height="175"></td>
    <td><img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN07-2nd-3Team/blob/main/images/%EB%8F%99%EC%9D%B5.jpg" width="175" height="175"></td>
    <td><img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN07-2nd-3Team/blob/main/images/%EB%8F%99%EC%9D%B5.jpg" width="175" height="175"></td>


  </tr>
  <tr>
    <th>팀장</th>
    <th>팀원</th>
    <th>팀원</th>
    <th>팀원</th>
    <th>팀원</th>
    <th>팀원</th>
 
  </tr>
</table>

---

## 프로젝트 소개<BR>
고객 이탈률 예측은 고객 경험을 개선하고 경쟁력을 높이는 데 중요한 역할을 합니다. 예측 모델을 통해 고객의 불만이나 불편을 사전에 파악하고 이를 해결함으로써 고객 만족도를 향상시킬 수 있습니다. 또한, 이탈률을 예측함으로써 경쟁사보다 더 나은 서비스를 제공하고, 고객 충성도를 유지하며, 장기적인 성장을 이끌어낼 수 있습니다.<BR>
우리는 구독 기반 서비스 기업, 전자상거래 기업, 금융 서비스 기업 등 서비스 제공 기업을 위한 **고객 이탈률 예측 모델**을 구축했습니다. 이를 통해 기업은 이탈 대응 전략을 수립하고, 서비스 개선안을 도출하여 더 효율적인 고객 관리와 경쟁력 있는 시장 전략을 마련할 수 있습니다.



---
### 개요
#### 가입 고객 이탈 예측 모델 설계 및 구축
<br>

**1. 훈련·테스트 데이터 선정**<br>
&nbsp;&nbsp;&nbsp;&nbsp;- 결측치가 적고 불필요한 feature가 적은 데이터를 선정해 모델 학습<br>
&nbsp;&nbsp;&nbsp;&nbsp;- Kaggle 머신러닝 웹 커뮤니티의 **Telecom Churn Dataset** 채택<br>

**2. ML 성능 비교**<br>
&nbsp;&nbsp;&nbsp;&nbsp;- ML 알고리즘 5개를 선정해 동일한 데이터로 성능 비교 후, 가장 성능이 좋은 알고리즘을 채택 <br>

**3. 예측 모델 설계·구축**<br>
&nbsp;&nbsp;&nbsp;&nbsp;- 채택한 알고리즘으로 ML 모델 설계 및 구축하여 성능 검증<br>

### 목표
여러 머신러닝 모델의 비교, 분석을 통해 고객 이탈 예측에 가장 적합한 고성능 모델을 구축합니다.

---

## 주요 데이터 설명
<table> <tr> <th>Number</th> <th>Column</th> <th>Meaning</th> <th>Example</th> </tr> <tr> <th>1</th> <th>State</th> <th>고객이 거주하는 주</th> <th>CA" (California), "NY" (New York)</th> </tr> <tr> <th>2</th> <th>International plan</th> <th>국제 전화 요금제 가입 여부</th> <th>"Yes" (가입), "No" (미가입)</th> </tr> <tr> <th>3</th> <th>Voice mail plan</th> <th>음성사서함 요금제 가입 여부</th> <th>"Yes" (가입), "No" (미가입)</th> </tr> <tr> <th>4</th> <th>Total day minutes</th> <th>주간(낮 시간) 동안 사용한 총 통화 시간(분)</th> <th>265.1 → 265.1분 통화</th> </tr> <tr> <th>5</th> <th>Total eve minutes</th> <th>저녁 시간 동안 사용한 총 통화 시간(분)</th> <th>197.4 → 197.4분 통화</th> </tr> <tr> <th>6</th> <th>Total intl minutes</th> <th>국제 통화에 사용된 총 시간(분)</th> <th>10.0 → 10분 사용</th> </tr> <tr> <th>7</th> <th>Customer service calls</th> <th>고객 센터에 전화한 횟수</th> <th>1 → 1회 전화</th> </tr> <tr> <th>8</th> <th>Churn</th> <th>고객 이탈 여부</th> <th>True (이탈), False (유지)</th> </tr> </table>

---

## EDA
### Correlation Matrix
![download](https://github.com/user-attachments/assets/6f8277cb-c581-465d-b751-56b82aa42e2b)
- Total day minutes, Customer service calls 등과 Churn 간의 상관관계가 두드러지게 나타남. 주간 통화 시간이 길수록 이탈률이 높아지는 패턴이 확인됨

### 주간 통화 이용률
![download](https://github.com/user-attachments/assets/0506510c-8f6c-4107-ad54-341797303ecf)
- 낮 시간(Total day minutes) 동안 통화 시간이 많은 고객의 이탈률이 상대적으로 높게 나타남. 이는 통화량이 많을수록 불만족 요인이 발생할 가능성이 있음

### 야간 통화 이용률
![download](https://github.com/user-attachments/assets/f3801354-610a-4863-8835-173f94369d2d)
- 저녁 시간(Total eve minutes) 동안의 통화량은 이탈률과 상대적으로 낮은 상관관계를 보임

### 심야 통화 이용률
![download](https://github.com/user-attachments/assets/986f494c-42d6-4a46-9389-824cc258261a)
- 야간 시간(Total night minutes)의 통화량은 고객 이탈과의 상관관계가 미미함

### 고객센터 상담 비율
<table>
  <tr>
    <td align="center"><b>Distribution of Customer Service Calls</b></td>
    <td align="center"><b>Customer Service Calls별 Churn 비율</b></td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/1dac0981-380d-46f8-b8cc-7872f46c555c" alt="Distribution Graph" width="600">
    </td>
    <td>
      <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN07-2nd-3Team/blob/main/images/curstomer_service_calls.png" alt="Churn Rate Graph" width="400">
    </td>
  </tr>
</table>

- Customer service calls 횟수가 증가할수록 이탈 확률이 급격히 높아짐. 이는 고객 불만족과 문제 해결의 어려움을 반영함

### 국제전화 가입비율
<table>
  <tr>
    <td align="center"><b>국제전화 분포</b></td>
    <td align="center"><b>국제전화 가입별 Churn 비율</b></td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/1db52c47-3d11-4d62-b159-37f264a24ee5" alt="Distribution Graph" width="500">
    </td>
    <td>
      <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN07-2nd-3Team/blob/main/images/Subscribed.png" alt="Churn Rate Graph" width="500">
    </td>
  </tr>
</table>

- 국제전화 요금제에 가입한 고객의 이탈률이 더 높기 때문에, 요금제의 서비스 품질 개선 또는 고객 불만족 해소를 위한 전략이 필요함

### 음성사사함 가입비율
<table>
  <tr>
    <td align="center"><b>음성사서함 분포</b></td>
    <td align="center"><b>음성사서함 가입별 Churn 비율</b></td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/aa216f78-ad8d-4a0b-83d1-7b5d57e456bf" alt="Distribution Graph" width="500">
    </td>
    <td>
      <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN07-2nd-3Team/blob/main/images/voice_mail.png" alt="Churn Rate Graph" width="500">
    </td>
  </tr>
</table>

- Voice mail plan 미가입 고객의 이탈률이 상대적으로 높음. 음성사서함 요금제가 고객 유지에 기여할 가능성 존재

### 이탈률
![download](https://github.com/user-attachments/assets/aa611630-d6a6-473f-8955-8428e40611d2)
- 전체 고객 중 이탈 고객의 비율이 낮지만, 특정 조건(높은 통화량, 고객센터 이용량)에서 집중적으로 이탈이 발생함

### 고객센터 통화량에 따른 이탈률 & 이상치
![churn_vs_customer_service_calls](https://github.com/user-attachments/assets/e491b51b-ce10-41ea-b632-9ded739fc8e7)
- 고객센터에 5회 이상 전화를 한 고객은 이탈률이 현저히 높음. 이는 서비스 불만족이 주요 원인임을 나타냄

### 통화량이 높은 고객들 대상 이탈률
![churn_vs_total_day_minutes](https://github.com/user-attachments/assets/9618780c-ebd7-4a49-8a94-511026fcb1cf)
- 통화량이 비정상적으로 높은 고객군에서 이탈률이 급격히 상승함. 이러한 고객은 별도의 관리가 필요함

### 데이터 전처리
<pre>
<code>
columns_to_drop = ['State', 'Area code']
train_data = train_data.drop(columns=columns_to_drop, axis=1)
test_data = test_data.drop(columns=columns_to_drop, axis=1)
</code>
</pre>

---

## Machine Learning
1. KNN Model
2. Decision Tree Model
3. XGBoost Model
4. Random Forest Model

### Evaluation Metrics by Model
<table>
  <tr>
    <th>Model</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
    <th>ROC AUC</th>
  </tr>
  <tr>
    <th>KNN</th>
    <th>0.885</th>
    <th>0.828</th>
    <th>0.304</th>
    <th>0.444</th>
    <th>0.646</th>
  </tr>
  <tr>
    <th>Decision Tree</th>
    <th>0.913</th>
    <th>0.680</th>
    <th>0.737</th>
    <th>0.707</th>
    <th>0.840</th>
  </tr>
  <tr>
    <th>XGBoost</th>
    <th>0.957</th>
    <th>0.934</th>
    <th>0.747</th>
    <th>0.830</th>
    <th>0.911</th>
  </tr>
  <tr>
    <th>Random Forest</th>
    <th>0.958</th>
    <th>0.972</th>
    <th>0.737</th>
    <th>0.838</th>
    <th>0.924</th>
  </tr>
</table>

### 최종 모델 선정 과정
이번 프로젝트에서는 XGBoost와 Random Forest 모델이 가장 높은 성능을 보였습니다. 각 모델의 주요 성능 지표를 비교한 결과 다음과 같은 결론을 도출했습니다.

&nbsp;&nbsp;&nbsp;&nbsp;- Random Forest는 Precision에서 가장 높은 성능을 보였으나, Recall이 상대적으로 낮아 이탈 고객을 놓칠 가능성이 있었습니다.

&nbsp;&nbsp;&nbsp;&nbsp;- XGBoost는 Recall과 F1 Score에서 더 균형 잡힌 성능을 보였습니다. 특히 Recall이 Random Forest보다 높아 이탈 고객을 더 많이 예측하는 강점을 보였습니다.

### 최종 모델 선정 (XGBoost 선택)
1. 균형 잡힌 성능<br>
&nbsp;&nbsp;&nbsp;&nbsp;- Recall이 Random Forest보다 높아 이탈 고객을 더 많이 예측할 수 있습니다.<br>
&nbsp;&nbsp;&nbsp;&nbsp;- F1 Score도 높은 수준을 유지하며 정확도와 재현율 간 균형을 이룹니다.<br>
2. 과적합 방지<br>
&nbsp;&nbsp;&nbsp;&nbsp;- Random Forest는 Precision은 높았지만 과적합 경향이 보였습니다.<br>
&nbsp;&nbsp;&nbsp;&nbsp;- XGBoost는 안정적이면서도 일반화 성능이 뛰어났습니다.<br>
3. 해석 가능성<br>
&nbsp;&nbsp;&nbsp;&nbsp;- XGBoost를 통해 Feature Importance를 분석하면 이탈 요인을 명확히 파악할 수 있습니다.<br>

---

## Deep Learning
### RNN Model
1. RNN Model Create
   <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN07-2nd-3Team/blob/main/images/rnn_model_create.png" alt="rnn_model_create" width="900px">

2. Used to Prevent Overfitting
![ prevent_overfitting ](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN07-2nd-3Team/blob/main/images/early%20stopping.png)

3. Training Result
![ training_result ](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN07-2nd-3Team/blob/main/images/training_result.png)
  - Training during 34 epoch

4. Loss Curve
   <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN07-2nd-3Team/blob/main/images/RNN_model_loss.png" alt="loss_curve" width="1100px">

5. Learning Curve
   <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN07-2nd-3Team/blob/main/images/RNN_model_learning_curve.png" alt="learning_curve" width="1100px">

6. Evaluation
   <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN07-2nd-3Team/blob/main/images/RNN_model_evaluation.png" alt="rnn_model_evaluation" width="1100px">

### RNN 모델의 한계
1. 성능 부족<br>
&nbsp;&nbsp;&nbsp;&nbsp;- 정확도(Accuracy)가 88.9%로 나쁘지 않지만, Recall이 43.2%로 매우 낮습니다.<br>
&nbsp;&nbsp;&nbsp;&nbsp;- 이는 이탈 고객을 제대로 예측하지 못하는 한계가 있음을 의미합니다.<br>
2. F1 Score<br>
&nbsp;&nbsp;&nbsp;&nbsp;- F1 Score가 0.526으로 낮아 Precision과 Recall의 균형이 부족합니다.<br>
3. ROC-AUC<br>
&nbsp;&nbsp;&nbsp;&nbsp;- ROC-AUC 점수가 0.698로, 다른 머신러닝 모델(XGBoost, Random Forest)에 비해 식별 능력이 떨어집니다.<br>

### XGBoost와의 비교
&nbsp;&nbsp;&nbsp;&nbsp;- XGBoost는 모든 성능 지표에서 RNN보다 우수한 결과를 보였습니다.<br>
&nbsp;&nbsp;&nbsp;&nbsp;- 특히 Recall과 ROC-AUC에서 큰 차이를 보이며, 이탈 고객 예측에 더 효과적입니다.<br>

RNN은 딥러닝 모델로 적용되었지만, 성능과 효율성 면에서 부족했습니다.
따라서 XGBoost가 최종 모델로 선정되었습니다.

---

## 기술 스택

---

## 이탈 예측 모델 구현
사용자는 다양한 머신러닝 모델을 선택하고, CSV 파일을 업로드하여 이탈 예측 결과를 확인할 수 있습니다.

### 프로젝트 실행 과정
1. CSV 파일 업로드<br>
&nbsp;&nbsp;&nbsp;&nbsp;- 예측할 데이터셋을 업로드 합니다.<br>
2. 모델 선택<br>
&nbsp;&nbsp;&nbsp;&nbsp;- 사용자는 Decision Tree, Random Forest, XGBoost 등 다양한 모델 중 하나를 선택할 수 있습니다.<br>
3. 이탈 예측 결과 확인<br>
&nbsp;&nbsp;&nbsp;&nbsp;- 예측된 고객 이탈률과 이탈 예상 고객 수를 확인할 수 있습니다.<br>

<table> 
  <tr> <td align="center"><b>CSV 파일 업로드</b></td> <td align="center"><b>모델 선택</b></td> <td align="center"><b>이탈 예측 결과</b></td> </tr> <tr> <td> <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN07-2nd-3Team/blob/main/images/streamlit%2001.png" alt="Model Selection" width="300"> </td> <td> <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN07-2nd-3Team/blob/main/images/streamlit%2002.png" alt="CSV File Upload" width="300"> </td> <td> <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN07-2nd-3Team/blob/main/images/streamlit%2003.png" alt="Churn Prediction Results" width="300"> </td> </tr> 
</table>

--- 

## 한 줄 회고
- 김현수:
- 박현준:
- 전서빈:
- 정소열:
- 조현정:
- 신민주:
