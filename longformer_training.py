from transformers import AutoTokenizer, LongformerModel
from transformers.models.longformer.modeling_longformer import LongformerPreTrainedModel, LongformerClassificationHead
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import argparse
import torch
from utils import preprocess, distribution_per_row


class LongformerForMultiLabelSequenceClassification(LongformerPreTrainedModel):
    """
    We instantiate a class of LongFormer adapted for a multilabel classification task.
    This instance takes the pooled output of the LongFormer based model and passes it through a classification head. We replace the traditional Cross Entropy loss with a BCE loss that generate probabilities for all the labels that we feed into the model.
    """

    def __init__(self, config):
        super(LongformerForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.longformer = LongformerModel(config)
        self.classifier = LongformerClassificationHead(config)
        self.softmax = torch.nn.Softmax(dim=1)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, global_attention_mask=None,
                token_type_ids=None, position_ids=None, inputs_embeds=None,
                labels=None):
        # create global attention on sequence, and a global attention token on the `s` token
        # the equivalent of the CLS token on BERT models. This is taken care of by HuggingFace
        # on the LongformerForSequenceClassification class
        if global_attention_mask is None:
            global_attention_mask = torch.zeros_like(input_ids)
            global_attention_mask[:, 0] = 1

        # pass arguments to longformer model
        outputs = self.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids)

        # if specified the model can return a dict where each key corresponds to the output of a
        # LongformerPooler output class. In this case we take the last hidden state of the sequence
        # which will have the shape (batch_size, sequence_length, hidden_size).
        sequence_output = outputs['last_hidden_state']

        # pass the hidden states through the classifier to obtain thee logits
        logits = self.classifier(sequence_output)
        # outputs = (logits,) + outputs[2:]
        outputs = self.softmax(logits)
        return outputs


class CustomDataset(Dataset):
    def __init__(self, texts, distributions, tokenizer):
        self.texts = texts
        self.distributions = distributions
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # tokenized_text = self.tokenizer(self.texts[idx],return_tensors = 'pt')
        label = torch.Tensor(self.distributions[idx])
        return self.texts[idx], label


class Datacollector():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        encoded_text = self.tokenizer.batch_encode_plus([x[0] for x in batch], padding=True, return_tensors='pt')
        labels = torch.stack([x[1] for x in batch])
        return encoded_text, labels


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', default=1e-3, type=float)
    parser.add_argument('-batch_size', default=4, type=int)
    parser.add_argument('-classes',default=5,type=int)
    return parser.parse_args()


def softcrossentropyloss(pred, real):
    temp = torch.log(pred) * real
    x = torch.sum(temp)
    return x


class Custom_Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        model_output = model(**inputs[0])
        labels = inputs[1]
        loss = softcrossentropyloss(model_output, labels)
        return loss


def train():
    args = parseargs()
    batch_size = args.batch_size
    lr = args.lr
    classes = args.classes
    args = TrainingArguments(output_dir="data", per_device_train_batch_size=batch_size)
    data = pd.read_csv('data/full_data.csv')
    data = preprocess(data)
    texts = data['text'].tolist()
    ann_cols = [col for col in data.columns if col not in ['text', 'example_id']]
    distributions = data[ann_cols].apply(lambda x: distribution_per_row(x), axis=1).tolist()
    model = LongformerForMultiLabelSequenceClassification.from_pretrained('allenai/longformer-base-4096',
                                                                          attention_window=512,
                                                                          num_labels=classes)
    tokenizer = AutoTokenizer.from_pretrained('allenai/longformer-base-4096')
    dataset = CustomDataset(texts, distributions, tokenizer)
    trainer = Custom_Trainer(model=model, args=args, data_collator=Datacollector(tokenizer), train_dataset=dataset,
                             eval_dataset=dataset, tokenizer=tokenizer)
    trainer.train()


train()
