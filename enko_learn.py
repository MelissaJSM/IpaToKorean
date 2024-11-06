import torch
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from torch.utils.data import Dataset, random_split
import os

# 데이터셋 파일 로드
data_path = "dataset.txt"
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 데이터 전처리
train_texts, train_labels = [], []
for line in lines:
    ipa, korean = line.strip().split('|')
    train_texts.append(ipa)
    train_labels.append(korean)

# 토크나이저 로드
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

# 데이터 토큰화
encodings = tokenizer(train_texts, padding=True, truncation=True, return_tensors="pt")
label_encodings = tokenizer(train_labels, padding=True, truncation=True, return_tensors="pt")

# 데이터셋 클래스 정의
class IPADataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])
        return item

    def __len__(self):
        return len(self.labels['input_ids'])

# 전체 데이터셋 준비
dataset = IPADataset(encodings, label_encodings)

# 학습 및 검증 데이터셋 분할
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 모델 준비
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

# 학습 설정
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=50,
    per_device_train_batch_size=16,  # 배치 크기 증가 (RTX A6000에 맞춤)
    gradient_accumulation_steps=2,  # 메모리 최적화를 위해 그라디언트 누적 사용
    fp16=True,  # Mixed Precision 사용으로 학습 속도 향상
    save_strategy="epoch",  # 에폭마다 모델 저장
    overwrite_output_dir=True,  # 출력 디렉토리를 덮어쓰지 않도록
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    eval_steps=10,  # 검증 단계를 주기적으로 수행
    eval_accumulation_steps=1  # 검증 데이터를 여러 번에 나눠서 평가
)

# 트레이너 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # 검증 데이터셋 추가
)

# 커스텀 콜백 추가 - 에폭마다 검증 손실 출력 및 최상의 모델 저장
from transformers import TrainerCallback

best_eval_loss = 10000.0  # 최상의 검증 손실을 추적하기 위한 변수

class EvalLossCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        global best_eval_loss
        if metrics:
            current_eval_loss = metrics.get('eval_loss')
            print(f"Validation Loss at Epoch {state.epoch}: {current_eval_loss}")
            # 최상의 검증 손실이면 모델 저장
            if current_eval_loss < best_eval_loss:
                print("최상의 모델을 찾았습니다.")
                best_eval_loss = current_eval_loss
                epoch_or_step = int(state.epoch) if state.epoch is not None else int(state.global_step)
                save_path = os.path.join(args.output_dir, f'model_epoch_{epoch_or_step}')  # 에포크/스텝별로 모델 저장
                trainer.save_model(save_path)
                print(f"Model saved at Epoch {state.epoch} with Validation Loss: {best_eval_loss}")
            else: 
                print(f"최상의 모델이 아닙니다. 현재 베스트값: {best_eval_loss}, 현재 진행 중인 값: {current_eval_loss}")

trainer.add_callback(EvalLossCallback)
# 검증 손실 출력
trainer.evaluate()
# 모델 학습 시작
trainer.train()
