---
layout: post
title: "pytorch-lightning Basic review"
slug: pytorch-lightning 
date: 2025-10-24 14:32:20 +0300
description: cVAE Description
img: ./pytorch-lightning/fig.svg   # Add image post (optional)
fig-caption: # Add figcaption (optional)
tags: [pytorch-lightning, AI_TECH_8]
categories: [Pytorch-framework]
published: True
---


이번에는 [pytorch-lightning](https://lightning.ai/docs/pytorch/stable/) framework 사용법을 익혀보려고 합니다. 먼저 framework에 어떤 기능들이 있고, 왜 쓰는지를 문서에서 Level-Up 카테고리의 Basic를 보며 정리했습니다. 

### Why use?

<div class="callout">
The lightweight PyTorch wrapper for high-performance AI research. Scale your models, not the boilerplate. 
</div>

위는 [PyPI](https://pypi.org/project/pytorch-lightning/)에서 언급한 소개입니다. 말 그대로 PyTorch 사용을 더 편하게 해주는 Wrapper Framerwork 입니다. 


Pytorch에서 `nn.Module`을 이용하여 Model을 만드는 것도 충분히 쉽지만, `pytorch-lightning`에서는 더 추상화해서 사용하기 더 쉽게 만들었습니다. Checkpoint 및 Log 관리도 편리하고, GPU 리소스 관리도 용이하게 해줍니다. 

개인적으로는 PyTorch도 충분히 High-level이라고 생각하는데, 레딧좀 찾아보니까 `pytorch-lightning`이 light하게 사용하기도 좋고 충분히 사용할만하다는 의견이 많네요. 저는 그전에 논문 코드에서 발견하고 PyTorch외에 추가적인 framework도 외워야하나.. 하고 좀 고민을 했는데, 이번 기회에 공부하면서 다른 사람이 이 framework를 사용한 경우에 구조랑 의도 파악하기 쉽도록 Key point 위주로 알아보겠습니다.  

이번 Posting에서는 기본적으로 어떻게 구성되는지와 Train, Test, Prediction, model save 를 알아보겠습니다. 

## Basic

### Train, Test, Predict

```python
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        # self.save_hyperparameters()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = F.mse_loss(x_hat, x)
        self.log("val_loss", val_loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

autoencoder = LitAutoEncoder(Encoder(), Decoder())
trainer = L.Trainer(callbacks=[EarlyStopping(monitor="val_loss", mode="min")], accelerator="gpu", devices="auto", max_epochs=5)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
```

위 코드가 가장 기본적인 예시입니다. `nn.Module` 대신 `L.LightningModule` 을 상속 받아서 구현해줍니다. 이때 `training_step`, `configure_optimizers` 함수는 필수적으로 구현해야하고, 나머지는 필요한 경우에 구현하면 됩니다. (~~대부분 구현하긴 해야겠지만요~~) 

그리고 단순하게 scikit-learn 모델처럼 `fit` 함수를 통해 학습을 시키면 끝입니다. 이전에는 loss도 계산에 optimizer까지 신경을 써야했다면 그러한 위에서는 과정이 추상화되었습니다. 

Checkpoint 저장 같은 경우 자동으로 checkpoint를 저장해줍니다. Default로는 가장 마지막 epoch의 모델만이 저장되는데, Intermediate 쪽에서 관련 내용이 나오니 밑에서 다루겠습니다. 저장되는 Directory를 지정하고 싶다면 `Trainer(default_root_dir="your_directory/path/")` 로 하면 됩니다. 



이후 저장된 checkpoint의 경우 `load_from_checkpoint(path)`를 통해 load 후 사용하면 됩니다. 

TODO : load_from_checkpoint 인자
```python
model = LitAutoEncoder.load_from_checkpoint(PATH, encoder=encoder, decoder=decoder)
```

하지만 위의 경우는 다소 불편할 수 있는데, 그 이유는 기존 `LitAutoEncoder`의 인자도 같이 줘야 하기 때문입니다. 해당 클래스의 경우에는 모델을 인자로 받기에, 해당 모델도 다시 정의해서 입력으로 넣어야합니다. (물론 encoder나 decoder의 state_dict는 checkpoint에 저장되어 있습니다.) 이런 불편한 case를 해소하려면 단순히 모델의 `init()`함수에 `self.save_hyperparameters()`를 작성하면 됩니다. (코드 주석 부분)

`self.save_hyperparameters()`은 `init()`의 입력으로 오는 인자를 저장합니다. 코드 예시는 다음과 같습니다. 

```python
from lightning.pytorch.core.mixins import HyperparametersMixin

class AutomaticArgsModel(HyperparametersMixin):
    def __init__(self, arg1, arg2, arg3):
        super().__init__()
        # equivalent automatic
        self.save_hyperparameters()
    def forward(self, *args, **kwargs):
        ...
        
model = AutomaticArgsModel(1, 'abc', 3.14)
print(model.hparams)
# "arg1": 1
# "arg2": abc
# "arg3": 3.14
```

checkpoint는 중간마다 저장된다고 했는데, 위의 `hparams`는 `checkpoint["hyper_parameters"]` 에서 확인할 수 있습니다. 아까 작성한 `LitAutoEncoder` class의 경우에는 `checkpoint["hyper_parameters"]`의 경우 다음과 같이 출력됩니다:

```python
{'encoder': Encoder(
   (l1): Sequential(
     (0): Linear(in_features=784, out_features=64, bias=True)
     (1): ReLU()
     (2): Linear(in_features=64, out_features=3, bias=True)
   )
 ),
 'decoder': Decoder(
   (l1): Sequential(
     (0): Linear(in_features=3, out_features=64, bias=True)
     (1): ReLU()
     (2): Linear(in_features=64, out_features=784, bias=True)
   )
 )}
```

이를 통해 `model = MyLightningModule.load_from_checkpoint("/path/to/checkpoint.ckpt")` 한 줄로 작성이 가능합니다.

마지막으로 `predict_step()`을 구현해서 모델의 predict를 편하게 했습니다. 여기에서 `test`와 `predict`의 차이가 무엇인지 궁금했는데, [link](https://github.com/Lightning-AI/pytorch-lightning/discussions/11455)에서 친절히 답변해주셨습니다. 즉, 이미 있는 Dataset에서 Train과 Test를 나누고, 여기서 다시 한번 Train과 Validation으로 나누어집니다. 즉 Test까지는 Label을 알고 있는 상황이고, 그저 unseen 데이터에 대해 얼마나 잘하는지 확인하기 위함입니다. 그래서 `predict`는 label도 없는 데이터를 예측할 때 사용합니다. 

```python
trainer.predict(model, data_loader)
```

즉 사용자는 `dataloader`를 for문을 통할 필요없이 인자로 넘겨줘서 코드를 깔끔하게 합니다. 이때, `Dropout` 같은 것을 사용한다면 `predict_loop` 함수 내에서 `self.dropout.train()`을 선언해줘야합니다. 


### Debugging

`pytorch-lightning`에서는 Debugging을 용이하게 하는 argument도 있습니다. 
- Model code가 잘 돌아가는지만 확인하기 위해  `Trainer(fast_dev_run=True)`로 5개의 Batch만 돌릴 수 있습니다. 
이외에도 `Trainer(limit_train_batches=0.1, limit_val_batches=0.01)`로 Batch의 일부만을 돌릴 수도 있습니다. 

- Train, Test 등 얼마나 시간이 걸렸는지를 알고 싶으면 `trainer = Trainer(profiler="simple")`를 통해 Profiling도 지원합니다. 

- 위에서는 `self.log()`를 통해 특정 loss만 기록했는데, `self.log_dict()`에 metrics 정보가 있는 dictionary를 넣어줌으로써 다른 metric도 logging이 가능합니다.  