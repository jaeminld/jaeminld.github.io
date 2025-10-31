---
layout: post
title: "Movie Sentiment Classification Competition Reflection"
slug: ai-tech-competition-reflection
date: 2025-10-31 10:32:20 +0300
description: Movie Review Sentiment Classification Competition Reflection"
img: domain_proj_aitech/head.png  # Add image post (optional)
fig-caption: # Add figcaption (optional)
tags: [BERT, Sentiment Analysis, NLP, Movie Reviews, HuggingFace, Competition, AI_TECH_8]
categories: [Competition]
published: True
---

네이버 부스트캠프 AI Tech에서 Week 7~8 동안 자연어 처리(한글)를 통해 "영화 리뷰 감성 분석 분류 예측" 대회를 진행했습니다. 이번에는 이번 대회를 진행하며 어떤 방식으로 어떻게 진행했는지를 Review 하겠습니다. 
Code refactoring -> Data preprocessing -> Model optimization 순으로 진행했습니다. Data preprocessing는 데이터를 직접 확인하며 tokenizer의 영향을 줄이기 위한 작업을 진행했고, Model의 경우 Huggingface에서 pretrained model을 이용해서 fine-tuning을 진행했습니다. 


## Code Refactoring

Baseline 코드가 ipynb로 되어 있어서 Shift + Enter 를 계속해야 하고, ipynb 특성상 깔끔하지 않아서 코드 재현성 측면에서 안정성이 떨어진다고 생각해서 모델 몇 번 돌려보고 바로 Refactoring을 진행했습니다. 연구 코드 같은 경우를 보면 model 정의부터 여러 utils를 이용하기 때문에 여러 폴더로 관리하는 걸 본 기억이 있는데, 대회에서 pretrained model을 이용했기 때문에 아래와 같이 단순하게 폴더를 구성했습니다

```
root/
├── configs/ - Json configs for each model
│   ├── kobert-base.yaml
│   ├── roberta-base.yaml
│   ├── koelectra-base-v3-discriminator.yaml
│   ├── rcbert-base.yaml
│   ├── bert-base.yaml
│   ├── lstm.yaml
│   
├── model/ - Model checkpoints
│
├── scripts/ 
│   ├── main.py - Execute entry
│   ├── ensemble.py
│   ├── helper.py - helper functions for train, test, prediction
│
├── src/
│   ├── datasets 
│       ├── custom_dataset.py
│       ├── dataset_factory.py - make function for dataset
│       ├── preprocessor.py
│   ├── models 
│       ├── custom_loss.py - Linear combination of Focal loss and Weighted loss
│       ├── custom_lstm.py 
│   ├── trainer 
│       ├── custom_scheduler.py
│       ├── custom_trainer.py
│       ├── trainer_factory.py - make function for Trainer
│   ├── utils 
│       ├── utils.py
```


HuggingFace에서 제공하는 `Trainer`를 이용해서 Model을 training시켰습니다. 이전에 Posting 했던 `pytorch-lightning`을 이용해볼려고 했는데, Pretrained model을 불러와서 사용하는데 어디선가 코드를 잘못 작성해서 학습이 원활히 안됐습니다...(~~여기에만 하루 날렸습니다~~) 

scripts 폴더 내의 py를 통해 src 폴더내의 함수들을 불러서 사용하는 방식으로 코드를 작성했습니다. 

configs 폴더 내에서는 yaml 파일로 각 모델의 Parameter, Training args 등 parameter 파일을 구성했습니다. 


## Data Preprocessing

일부 데이터를 확인 해본 결과, 굉장히 Raw한 데이터였습니다. 일부 욕설 같은 경우만 OO+ Masking 처리되어 있었고, 초성같은 것은 Masking이 안되어 있었습니다. 이모티콘같은 경우에도 "ㅡ.ㅡ" 같은 고전 이모티콘 뿐만 아니라 🔥같은 감정 혹은 상태를 나타내는 이모지도 다수 사용하는 것을 확인했습니다. 이외에도 "asdf" 나 "ㅁㄴㅇㄹ" 등과 같은 Noising 데이터도 있었습니다. 한국어 리뷰지만, 영어도 다수 섞여 있었습니다. 

위와 같은 경우가 대다수인 것도 있지만, 소수이면서도 Model이 해당 문장을 이해하는데 적지 않은 영향이 있을 것이라 생각했습니다. 따라서 다음과 같은 전략을 취했습니다. 

- 이모지나 이모티콘의 경우 다수 사용하는 이모티콘의 경우 종류를 나눠서 `SPECIAL_TOKEN`으로 취급했습니다. (😡 -> `[ANGRY]`, ㅠㅠ -> `[SAD]` 등)  
- 욕설 같은 경우, 데이터에서 다수 발견되어서 대표적인 단어나 초성은 `[ABUSE]`로 처리했습니다. 
- 영어가 섞여 있는 경우, 많지는 않지만 다수 발견되는 대표적인 단어 10개 정도를 직접 해석해서 대체시켜주었습니다. 또한 모두 소문자로 변환했습니다. 

위의 전략 이외에도 데이터를 약 1000개 정도 직접 보며 발견한 맞춤법 혹은 통일할 수 있는 내용들을 `SPECIAL_TOKEN`으로 취급하거나 손수 고쳐주었습니다.  
이때 필요한 것은 Regular Expression으로 한 번에 데이터를 처리하는 것이 였는데, 필요한 내용은 <a href="https://hamait.tistory.com/342" target="_blank">블로그</a>에서 많은 도움을 받았습니다. 

위와 같은 전략을 취한 이유는, 기존 한국어 데이터셋으로 Pretrained model의 경우에는 Noise가 포함되어 있지 않다고 생각했습니다. 이러한 Noise의 경우 성능에 Dramatic한 영향이 있지는 않을 수 있지만, 대회에서 진행하는 만큼 전처리하는 경우 소폭의 향상이라도 있을 것이라 생각했습니다. 

향상이 있을 것이라 생각한 이유는 Tokenizer의 OOV 문제 때문입니다. Bert tokenizer의 경우 WordPiece기반 tokenizer로, 자주 등장하는 단어의 경우는 하나의 token으로, 아니면 여러 개의 subword로 tokenize합니다. 소수의 단어가 model이 이해하기 어렵게 tokenize한다고 아래와 같은 코드를 통해 확인했고, OOV 문제가 발생한다고 판단했습니다. 

대표적으로 `klue/roberta-base` 모델의 Tokenizer 모델을 불러와서 text를 tokenizing 하는 코드를 보겠습니다. 


```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

texts = ["OST와 ost는 다른가요?", "오늘은 worst한 하루였어."]

for text in texts: 
    print(f"TEXT: {text}")
    print(f"bert-base-cased tokenizer result: {bert_tokenizer.tokenize(text)}")
    print(f"klue/roberta-base result {tokenizer.tokenize(text)}")
    print()
```

```
TEXT: OST와 ost는 다른가요?
bert-base-cased tokenizer result: ['[UNK]', '[UNK]', '[UNK]', '?']
klue/roberta-base result ['OST', '##와', 'o', '##st', '##는', '다른', '##가요', '?']

TEXT: 오늘은 worst한 하루였어.
bert-base-cased tokenizer result: ['[UNK]', 'worst', '##한', '[UNK]', '.']
klue/roberta-base result ['오늘', '##은', 'w', '##ors', '##t', '##한', '하루', '##였', '##어', '.']
```

위의 결과에서 볼 수 있듯이, 같은 OST를 나타내더라도, OST와 ost는 다른 단어를 의미하게 되고, worst의 경우에는 english tokenizer인 bert-base-cased tokenizer에서는 worst처럼 한 단어로 인식하지만, klue로 학습된 tokenizer의 경우 별개의 단어로 인식합니다. 전자의 경우 영어 단어는 모두 소문자화를 통해 의미를 통일하고자 했고, 후자의 경우에는 단어 빈도 수 기준 top 10~20 정도의 의미 있는 단어들만 한글로 naive하게 번역하여 대체했습니다. 

아래는 Tokenizer에 `SPECIAL_TOKEN`에 직접 token 추가 및 대체 예시 코드입니다. 

```python
# In preprocessor.py
###
...
text = re.sub(r"OTL", "좌절", text)
...
###

# In utils.py
...
tokenizer.add_special_tokens({"additional_special_tokens": preprocessor.new_tokens})
model.resize_token_embeddings(len(tokenizer))
...
```

이와 같은 과정을 거치며 OOV 문제, 특히 Subword 문제를 최소화하여 데이터에 있는 noise를 줄이고자 했습니다. 
AutoModel을 이용하여 Pretrained model을 load하면 다음과 같은 Folder에 다운로드됩니다. 모델마다 `snapshots/` folder의 구조가 다를 수 있습니다. 

```
root/
│
├── .cache/ 
│   ├── huggingface/
│       ├── hub/
│           ├── models--kykim--bert-kor-base
│               ├── snapshots/snapshot_ids
│                   ├── config.json
│                   ├── pytorch_model.bin
│                   ├── tokenizer_config.json
│                   ├── vocab.txt
```

json 파일들에서 현재 tokenizer의 special token을 알 수 있고, vocab파일에서 tokenizer의 vocab을 알 수 있습니다. 

데이터 전처리를 통해 Baseline model의 성능이 점차 향상되는 것을 확인했고, 데이터 전처리를 다한 후에 Model 선정, 앙상블 전략, hyperparameter 최적화를 진행했습니다. 


## Model

대회에서 사용가능한 모델은 `klue/roberta-base`, `klue/bert-base`, `kykim/bert-kor-base`, `beomi/kcbert-base`, `monologg/koelectra-base-v3-discriminator` 였고, 모두 활용하고자 했습니다. 모든 model마다 최적화된 hyperparameter를 찾기에는 앞선 Code refactoring과 Data preprocessing에 시간을 너무 많이 써서 baseline에서 가장 성능이 좋았던 `kykim/bert-for-base`를 base로 heuristic하게 모델 공통의 hyperparameter를 찾고자 했습니다. 

처음에는 모델의 결과를 계속해서 제출하며 Public 점수를 관찰했었는데, 그러다 보니 버전 관리하기가 어려웠습니다. 그래서 강의에서 권장했던 `wandb`를 활용했습니다. 


### Weight & Bias

wandb는 설치하고, terminal에 `wandb login` 으로 사용할 수 있었습니다. Trainer에는 아래 코드와 같이 사용할 시에 wandb에 자동으로 logging을 지원합니다. 

```Python
TrainingArguments(..., report_to="wandb", run_name="run_name", ...)
```

이렇게 하면 `run_name`에 따라 각 model의 Train loss, learning rate 등의 logging을 wandb에서 그래프로 확인할 수 있습니다. 아래는 제가 확인한 그래프입니다. 

<img src="{{site.baseurl}}/assets/img/domain_proj_aitech/wandb.png" alt="wandb">

각각의 process 이름으로 `model_name-data_ver-model_ver` 이라는 규칙으로 `run_name`에서 각기 다르게 설정했습니다. 

위에서 보면 다양한 모델 그래프를 볼 수 있는데, Metric은 accuracy 기반 평가이기 때문에, 주로 accuracy를 보고, 불균형을 잘 처리하는지 확인하기 위해서는 F1 score도 확인했습니다. 


### Ensemble
앙상블을 하면 통상적으로 각 모델이 가지는 특성을 특정 비율로 Voting을 하는 것이기에 성능 향상이 통상적으로 있다고 알려져 있습니다. 따라서 공통된 hyperparameter를 이용하여 각 model을 학습한 후에는 Soft-voting을 이용해서 성능을 높였습니다. 단일 모델 최고 성능이 0.8273였다면, soft-voting을 통해서 0.8389까지 성능을 높일 수 있었습니다.  


## Failed trial
위의 방법 외에도 많은 방법을 시도했었는데, 잘못 적용했는지 아니면 위의 모델이나 데이터에 적합하지 않은 탓인지 좋은 정확도를 보이지 못했습니다. 
- 먼저 Data label의 분포가 좀 불균형하기에 적은 개수의 label을 back-translation을 통해 데이터 augmentation을 해보려고 했는데, 모델을 계속 부르다보니 시간이 엄청 오래 걸릴뿐만 아니라 번역 품질도 굉장히 좋지 않아서 철회했습니다. 

- 데이터 전처리 과정에서 영어 단어로만 이루어진 문장도 있고, 영어 단어도 많았기에 이를 google 번역 API를 이용해서 번역해주려고 했는데, 데이터셋이 너무 크다보니 시간이 너무 오래걸렸습니다.. 그래서 top 10~20의 단어들만 손수 번역하여 대체했습니다. 

- `wandb` 그래프로 확인했을 때, Learning rate가 너무 낮아지면 과적합되어서 5~6 epoch 쯤부터는 성능이 오히려 하락하는 것을 볼 수 있었습니다. 그래서 Custom Learning Scheduler, 특히 <a href="https://gaussian37.github.io/dl-pytorch-lr_scheduler/" target="_blank">블로그</a>에서 Custom scheduler를 사용하려고 했는데, `Trainer`에 이를 변형하는 것이 생각보다 복잡했습니다. 단순히 optimizer랑 scheduler를 넣어주면 되는 logic인 줄 알았는데, base lr에서 올라가지 않고, step이 어떻게 계산되는지를 시간 부족으로 인해 해결하지 못했습니다. <a href="https://discuss.huggingface.co/t/how-do-use-lr-scheduler/4046" target="_blank">discuss</a>에서 처럼 Trainer를 상속받아서도 시도해보았는데, 결국엔 실패했습니다. 


- Code refactoring 폴더를 보면 Custom Loss를 정의한 부분이 있는데, 이는 Train 데이터의 불균형을 해소하기 위해 Weighted Cross entropy와 Focal loss를 Linear combination으로 적용했습니다. Weighted Cross Entropy의 경우 train data의 class weight를 그대로 적용하면 너무 성능이 떨어져서 살짝 Smoothning을 했습니다. 모든 모델을 이러한 Loss로 training 시키고, Base 5개 새로운 loss로 훈련시킨 모델 5개로 Soft-voting을 하려고 했었는데, 대회 종료 날짜를 착각하는 바람에... 제출하지 못했습니다. custom_lstm.py도 비슷한 맥락으로 못했습니다. 


## Review
이번 대회에 참여하면서, 자연어 처리를 하는데 방식에 대해 더 이해할 수 있었습니다. 자연어 데이터 전처리를 어떻게 해야 더 좋은 성능을 낼지, `wandb`를 이용한 모델 분석 등을 경험할 수 있어서 보람이 있었습니다. 아이디어가 더 많았는데, 시간 부족 + 날짜 착각이라는 이유로 못한 점이 아쉬운 것 같습니다. 다음에 대회를 진행한다면 code refactoring을 더 빨리, 그리고 일정 계산을 더 염두에 두고 진행해야겠다고 생각이 들었습니다.