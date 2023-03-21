import pandas as pd
from sklearn.preprocessing import OneHotEncoder as o


data_dic = {'label': ['Apple', 'Samsung', 'LG', 'Samsung']}
df = pd.DataFrame(data_dic)
oh = o()

sk_oh_encoded = oh.fit_transform(df)
pd_oh_encoded = pd.get_dummies(df['label'])
print(pd_oh_encoded)
print("="*100)
print(sk_oh_encoded)