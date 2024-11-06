
# ipaToKorean

해당 파일은 127,996 문장의 IPA | 한글 발음 데이터를 수집하여 딥러닝 학습을 할 수 있도록 구성된 프로젝트입니다.

## 설치 방법

1. **아나콘다 설치 후 환경 활성화 및 Python 3.9 설정**
   ```bash
   conda create -n {name} python=3.9
   conda activate {name}
   ```

2. **필수 패키지 설치**
   ```bash
   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117
   pip install transformers
   ```

## 딥러닝 학습 방법

아래 명령어를 통해 모델 학습을 시작할 수 있습니다:

```bash
python enko_learn.py
```

모델은 에포크가 완료될 때마다 `result` 폴더에 저장됩니다.

## 추론 방법

아래 명령어를 통해 학습된 모델로 추론을 수행할 수 있습니다:

```bash
python enko_start.py
```

`enko_start.py` 파일 내의 코드에서 추론에 사용할 IPA 텍스트를 변경하여 원하는 텍스트로 결과를 확인할 수 있습니다.
