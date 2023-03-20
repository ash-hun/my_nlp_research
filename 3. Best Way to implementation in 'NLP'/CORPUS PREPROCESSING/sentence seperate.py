import sys, fileinput, re
from nltk.tokenize import sent_tokenize


def seperate_sentence():
    # - 적용 전 데이터 -
    # 자연어처리는 인공지능의 한 줄기 입니다. 시퀀스 투 시퀀스의 등장 이후로 딥러닝을 활용한 자연어처리는 새로운 전기를 맞이하게 되었습니다. 문장을 받아 단순히 수치로 나타내던 시절을 넘어, 원하는대로 문장을 만들어낼 수 있게 된 것입니다.
    for line in fileinput.input():
        if line.strip() != "":
            line = re.sub(r'([a-z])\.([A-Z])', r'\1.\2', line.strip())

            sentences = sent_tokenize(line.strip())

            print("="*100)

            for s in sentences:
                if s != "":
                    sys.stdout.write(s+"\n")

def combine_sentence():
    # - 적용 전 데이터 -
    # 자연어처리는 인공지능의 한 줄기 입니다.\n
    # 시퀀스 투 시퀀스의 등장 이후로 딥러닝을 활용한 자연어처리는 새로운 전기를 맞이하게 되었습니다. 문장을 \n
    # 받아 단순히 수치로 나타내던 시절을 넘어, 원하는대로 문장을 만들어낼 수 \n
    # 있게 된 것입니다.
    buf = []

    text = ["자연어처리는 인공지능의 한 줄기 입니다.\n",
    "시퀀스 투 시퀀스의 등장 이후로 딥러닝을 활용한 자연어처리는 새로운 전기를 맞이하게 되었습니다. 문장을\n",
    "받아 단순히 수치로 나타내던 시절을 넘어, 원하는대로 문장을 만들어낼 수\n",
    "있게 된 것입니다.\n"]

    for line in text:
        if line.strip() != "":
            buf += [line.strip()]
            sentences = sent_tokenize(" ".join(buf))

            if len(sentences) > 1:
                buf = sentences[1:]

                sys.stdout.write(sentences[0] + '\n')

    sys.stdout.write(" ".join(buf) + "\n")

if __name__=="__main__":
    combine_sentence()

    seperate_sentence()
