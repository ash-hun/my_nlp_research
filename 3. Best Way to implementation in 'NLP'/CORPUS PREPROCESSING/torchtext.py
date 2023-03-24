from torchtext import data
from torchtext.data import TabularDataset, Iterator
import urllib.request
import pandas as pd


urllib.request.urlretrieve("https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv", filename="IMDb_Reviews.csv")
df = pd.read_csv('IMDb_Reviews.csv', encoding='latin1')

train_df = df[:25000]
test_df = df[25000:]

train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

# field Definition
TEXT = data.Field(sequential=True, use_vocab=True, tokenize=str.split, lower=True, batch_first=True, fix_length=20)
LABEL = data.Field(sequential=False, use_vocab=False, batch_first=False, is_target=True)

# sequential : 시퀀스 데이터 여부. (True가 기본값)
# use_vocab : 단어 집합을 만들 것인지 여부. (True가 기본값)
# tokenize : 어떤 토큰화 함수를 사용할 것인지 지정. (string.split이 기본값)
# lower : 영어 데이터를 전부 소문자화한다. (False가 기본값)
# batch_first : 미니 배치 차원을 맨 앞으로 하여 데이터를 불러올 것인지 여부. (False가 기본값)
# is_target : 레이블 데이터 여부. (False가 기본값)
# fix_length : 최대 허용 길이. 이 길이에 맞춰서 패딩 작업(Padding)이 진행된다.

# TabularDataset은 데이터를 불러오면서 필드에서 정의했던 토큰화 방법으로 토큰화를 수행합니다.
train_data, test_data = TabularDataset.splits(
    path=".", trainn='train_data.csv', test='test_data.csv', format='csv',
    fields=[('text', TEXT), ('label', LABEL)], skip_header=True)

# path : 파일이 위치한 경로.
# format : 데이터의 포맷.
# fields : 위에서 정의한 필드를 지정. 첫번째 원소는 데이터 셋 내에서 해당 필드를 호칭할 이름, 두번째 원소는 지정할 필드.
# skip_header : 데이터의 첫번째 줄은 무시.

TEXT.build_vocab(train_data, min_freq=10, max_size=10000)
# min_freq : 단어 집합에 추가 시 단어의 최소 등장 빈도 조건을 추가.
# max_size : 단어 집합의 최대 크기를 지정.


batch_size = 5
train_loader = Iterator(dataset=train_data, batch_size=batch_size)
test_loader = Iterator(dataset=test_data, batch_size=batch_size)

batch = next(iter(train_loader)) #첫번째 미니배치
