XGBoost 이후로 나온 최신 부스팅 모델입니다.

이것은 리프 중심 트리 분할 방식을 사용합니다. 표로 정리된 데이터(tabular data)에서 Catboost, XGBoost와 함께 가장 좋은 성능을 보여주는 알고리즘입니다. 

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6ca7c64d-8446-4137-942d-6aadaa89f8cd/Untitled.png)

**장점**

- XGBoost보다도 빠르고 높은 정확도를 보여주는 경우가 많습니다.
- 예측에 영향을 미친 변수의 중요도를 확인할 수 있습니다.
- 변수 종류가 많고 데이터가 클수록 상대적으로 뛰어난 성능을 보여줍니다.

**단점**

- 복잡한 모델인 만큼, 해석에 어려움이 있습니다.
- 하이퍼파라미터 튜닝이 까다롭습니다.

**유용한 곳**

- 종속변수가 연속형 데이터인 경우나 범주형 데이터인 경우 모두 사용할 수 있습니다.
- 이미지나 자연어가 아닌 표로 정리된 데이터라면 거의 모든 상황에서 활용할 수 있습니다.

### Z-Score

Z 점수(Z-Score, z값, 표준값, 표준 점수라고도 표현합니다.)는 평균과 표준편차를 이용하여 특정값이 정규분포 범위에서 어느 수준에 위치하는지 나타냅니다.

$$
Z-Score = \frac{x(특정값)-\gamma(평균)}{\sigma(표준편차)}
$$

### 이진분류의 평가지표

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0993343c-6c70-4bf1-ae4c-b1fabd634be7/Untitled.png)

XGBoost와 LightGBM의 차이점은 ‘트리의 가지를 어떤 식으로 뻗어나가는가’로 구분할 수 있습니다. XGBoost는 균형분할(Level-wise tree growth)방식으로 각 노드에서 같은 깊이를 형성하도록 한층 한층 밑으로 내려옵니다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f5d6ec71-09c0-494a-89ff-4d39f7985447/Untitled.png)

반면 LightGBM은 이러한 전제를 거부하고, 특정 노드에서 뻗어나가는 가지가 모델의 개선에 더 도움이 되면 아래와 같은 형식으로 진행될 수 있습니다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/278d4e7b-bb3a-4f9c-91a0-b31ee18efa89/Untitled.png)

그래서 LightGBM은 속도가 더 빠르게 진행될 수 있고, 복잡성은 더 증가하며, 오버피팅 문제를 야기할 가능성도 더 높습니다.