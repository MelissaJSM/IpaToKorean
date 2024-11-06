import torch
from transformers import BartTokenizer, BartForConditionalGeneration

# 모델 및 토크나이저 로드
model_path = "./results/bestOfbest"  # 학습한 모델이 저장된 경로
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained(model_path)

# 추론을 위해 GPU 설정 (RTX 3080 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 추론 함수 정의
def infer_ipa_to_korean(ipa_text):
    # 입력 텍스트 토큰화
    inputs = tokenizer(ipa_text, return_tensors="pt").to(device)
    # 모델 추론
    outputs = model.generate(
        inputs['input_ids'],
        max_length=50,  # 출력 최대 길이 (단어 단위이므로 적정값으로 설정)
        num_beams=5,  # 빔 서치 사용해 다양한 출력 탐색
        early_stopping=True
    )
    # 출력 텍스트 디코딩
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

# 예시 입력 및 추론 수행
example_ipa_texts = [
    "/hɛˈloʊ aɪm roʊˈboʊkoʊ sɑn ə ˈmɛmbɚ əv ˈhoʊloʊˌlaɪvz zɪroʊθ ˌʤɛnəˈreɪʃən. aɪ lʌv ˈstriːmɪŋ ɡeɪmz, əˈspɛʃəli ɛf pi ɛs ɡeɪmz, ənd aɪ ɛnˈʤɔɪ ˌɪntəˈrækʃən wɪð maɪ fænz, ðə ˈroʊboʊsɑz! aɪ ˈɔːlsoʊ lʌv ˈsɪŋɪŋ ənd pɑrˈtɪsɪˌpeɪtɪŋ ɪn ˈmjuzɪk ˈprɑʤɛkts. plʌs, aɪ hæv tu əˈdɔːrəbəl kæts, ˈhæpi ənd koʊˈroʊmoʊ. naɪs tə mit ju! lɛt mi noʊ ɪf ðɛrz ˈɛnɪθɪŋ ɛls jud laɪk tə noʊ!"
]

# 전체 문장 변환 수행
for ipa_text in example_ipa_texts:
    # 문장을 단어 단위로 분리
    ipa_words = ipa_text.split()
    # 각 단어에 대해 변환 수행 후 결과를 합침
    korean_pronunciation_words = [infer_ipa_to_korean(word) for word in ipa_words]
    korean_pronunciation_sentence = ' '.join(korean_pronunciation_words)
    print(f"IPA: {ipa_text} -> Korean Pronunciation: {korean_pronunciation_sentence}")

# GPU 메모리 정리
torch.cuda.empty_cache()
