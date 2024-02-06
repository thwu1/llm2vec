from train.trainer import ContrastiveTrainer
from train.modeling import BertForContrastiveLearning
from train.dataset import ContrastiveDataset
from transformers import BertConfig

dataset = ContrastiveDataset(dir = "evaluation/outputs/all-mpnet-base-v2")
label_name = dataset.get_label_name()
dataloader, test_dataloader = dataset.get_dataloader(batch_size=8, test_split=0.1)

config = BertConfig.from_json_file("train/config.json")
model = BertForContrastiveLearning(config).to("cuda")

trainer = ContrastiveTrainer(model, dataloader, test_dataloader, optimizer_config=None, label_name=label_name)

trainer.train()