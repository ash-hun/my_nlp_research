import sys, fileinput, re
from nltk.tokenize import sent_tokenize


if __name__=="__main__":
    for line in fileinput.input():
        if line.strip() != "":
            line = re.sub(r'([a-z])\.([A-Z])', r'\1.\2', line.strip())

            sentences = sent_tokenize(line.strip())

            print("="*100)

            for s in sentences:
                if s != "":
                    sys.stdout.write(s+"\n")

# 자연어처리는 인공지능의 한 줄기 입니다. 시퀀스 투 시퀀스의 등장 이후로 딥러닝을 활용한 자연어처리는 새로운 전기를 맞이하게 되었습니다. 문장을 받아 단순히 수치로 나타내던 시절을 넘어, 원하는대로 문장을 만들어낼 수 있게 된 것입니다.