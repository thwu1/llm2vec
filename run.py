from train.trainer import ContrastiveTrainer
from train.modeling import BertForContrastiveLearning
from train.dataset import ContrastiveDataset
from transformers import BertConfig
import json

config = json.load(open("config/config.json"))

dataset = ContrastiveDataset(dir = "evaluation/outputs/all-mpnet-base-v2")
label_name = dataset.get_label_name()
dataloader, test_dataloader = dataset.get_dataloader(batch_size=config["trainer_config"]["batch_size"], test_split=0.1)

model_config = BertConfig.from_dict(config["model_config"])
# print(model_config)
model = BertForContrastiveLearning(model_config).to("cuda")

trainer = ContrastiveTrainer(model, dataloader, test_dataloader, config=config, label_name=label_name)

trainer.train()