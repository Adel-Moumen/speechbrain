import torch.nn as nn 
import torch
import transformers

from transformers import Wav2Vec2Model, HubertModel, WavLMModel
from transformers import Wav2Vec2Config, HubertConfig, WavLMConfig

from abc import ABC, abstractmethod


# HF_config = {
#     "wav2vec2": Wav2Vec2Config,
#     "hubert": HubertConfig,
#     "wavlm": WavLMConfig,
# }

class HuggingFaceModel(nn.Module, ABC):

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name 
        self.config = getattr(transformers, model_name + "Config", None)
        self.model = getattr(transformers, model_name + "Model", None)
        assert self.config is not None and self.model is not None 

    @abstractmethod
    def forward(self, **kargs):
        raise NotImplementedError

    

class HuggingFaceASR(HuggingFaceModel):

    def __init__(
        self,
        model_name
    ):
        super().__init__(model_name=model_name)


    def forward(self, x):
        return x

if __name__ == "__main__":
    
    # print(getattr(transformers, "Bert" + "Config", None)) 
    # print(getattr(transformers, "Hubert" + "Config", None)) # last value = default value if not found
    # print(getattr(transformers, "Wav2Vec2" + "Config", None)) 

    # # not the best way! 
    # # because we can avoid having a dictionary :) 
    # print("sb way:", HF_config.get("hubert"))
    # print("sb way:", HF_config.get("wav2vec2"))

    model = HuggingFaceASR("Hubert")
    model(torch.randn(1))
    
    print(getattr(transformers, "Hubert" + "FeatureExtractor", None)) 

