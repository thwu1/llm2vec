from train.modeling import BertModelCustom, BertForContrastiveLearning
from transformers import BertConfig
import torch

config = BertConfig.from_json_file("train/config.json")
model = BertModelCustom(config)
model.eval()

input = torch.randn(2, 7, 768)
input_permuted = torch.cat([input[:, 0:2, :], input[:, 5:7, :], input[:, 2:5, :]], dim=1)

print(model(input)[1])
print(model(input_permuted)[1])

assert torch.equal(model(input).pooler_output, model(input)[1])

model = BertForContrastiveLearning(config)
model.eval()

input = torch.randn(2, 7, 768)
input_permuted = torch.cat([input[:, 0:2, :], input[:, 5:7, :], input[:, 2:5, :]], dim=1)

print(model(input)[1])
print(model(input_permuted)[1])

print(torch.norm(model.encode(input, output_tensor=True), dim=1))

model.save_pretrained("checkpoint_0")

model = BertForContrastiveLearning.from_pretrained("checkpoint_0")
model.eval()
print(model(input)[1])
