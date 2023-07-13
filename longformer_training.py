from transformers import AutoTokenizer, LongformerModel
from transformers.models.longformer.modeling_longformer import LongformerPreTrainedModel, LongformerClassificationHead
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import argparse
import torch
from utils import preprocess, distribution_per_row
from sklearn.model_selection import train_test_split
from transformers import EvalPrediction
import optuna
import os




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
    def __init__(self, texts, distributions):
        """
        :param texts: list of strings
        :param distributions: list of numpy arrays size 5
        """
        self.texts = texts
        self.distributions = distributions

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        label = torch.Tensor(self.distributions[idx])
        return self.texts[idx], label


class Datacollector():
    """
    Custom data collector for datasets in trainer
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        encoded_text = self.tokenizer.batch_encode_plus([x[0] for x in batch], padding=True, return_tensors='pt')
        labels = torch.stack([x[1] for x in batch])
        return {'input_ids': encoded_text.input_ids, "attention_mask": encoded_text.attention_mask, "labels": labels}


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', default=5e-4, type=float)
    parser.add_argument('-train_batch_size', default=4, type=int)
    parser.add_argument('-eval_batch_size', default=-1, type=int)
    parser.add_argument('-classes', default=5, type=int)
    parser.add_argument('-train_percentage', type=float, default=0.6)
    parser.add_argument('-val_percentage', type=float, default=0.2)
    parser.add_argument('-evaluation_strategy', type=str, default='epoch')
    parser.add_argument('-eval_each_x_steps', type=int, default=1)
    parser.add_argument('-epochs', default=1, type=int)
    args = parser.parse_args()
    test_percentage = 1 - args.train_percentage - args.val_percentage
    assert test_percentage > 0
    args.test_percentage = test_percentage
    return args


def SoftCrossEntropyLoss(pred, real):
    """
    :param pred: tensor of predictions
    :param real: tensor of real labels
    :return: loss
    """
    entry_wise_entropy = torch.log(pred) * real
    loss_per_sample = -torch.sum(entry_wise_entropy, dim=1)
    return torch.mean(loss_per_sample, dim=0)

def SoftCrossEntropy(pred, real):
    """
    :param pred: numpy array of predictions
    :param real: numpy array of labels
    :return: cross entropy
    """
    entry_wise_entropy = np.log(pred) * real
    loss_per_sample = -np.sum(entry_wise_entropy, axis=1)
    return np.mean(loss_per_sample)

class Custom_Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        model_output = model(input_ids=inputs["input_ids"], attention_mask=inputs['attention_mask'])
        labels = inputs["labels"]
        loss = SoftCrossEntropyLoss(model_output, labels)
        return (loss, model_output) if return_outputs else loss


def process_data(path, args):
    data = pd.read_csv(path)
    data = preprocess(data)
    texts = data['text'].tolist()
    ann_cols = [col for col in data.columns if col not in ['text', 'example_id']]
    distributions = data[ann_cols].apply(lambda x: distribution_per_row(x), axis=1).tolist()
    #TODO: set train validation and test splits across all models
    train_val_texts, test_texts, train_val_distributions, test_distributions = train_test_split(texts, distributions,
                                                                                                test_size=args.test_percentage)
    train_texts, val_texts, train_distibutions, val_distributions = train_test_split(train_val_texts,
                                                                                     train_val_distributions,
                                                                                     test_size=args.val_percentage / (
                                                                                             args.train_percentage + args.val_percentage))
    train_dataset = CustomDataset(train_texts, train_distibutions)
    val_dataset = CustomDataset(val_texts, val_distributions)
    test_dataset = CustomDataset(test_texts, test_distributions)
    if args.eval_batch_size == -1:
        args.eval_batch_size = len(val_dataset)
    return train_dataset, val_dataset, test_dataset


def train():
    args = parseargs()
    os.environ["WANDB_DISABLED"] = "true"
    classes = args.classes
    path = 'data/full_data.csv'
    #TODO: check if we can play with attention window
    model = LongformerForMultiLabelSequenceClassification.from_pretrained('allenai/longformer-base-4096',
                                                                          attention_window=512,
                                                                          num_labels=classes)
    tokenizer = AutoTokenizer.from_pretrained('allenai/longformer-base-4096')
    train_dataset, val_dataset, test_dataset = process_data(path, args)
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    lr = args.lr
    evaluation_strategy = args.evaluation_strategy
    epochs = args.epochs
    #TODO: do checkpointing
    if evaluation_strategy == 'steps':
        eval_steps = args.eval_each_x_steps
        training_args = TrainingArguments(output_dir="data", per_device_train_batch_size=train_batch_size,
                                          per_device_eval_batch_size=eval_batch_size, learning_rate=lr,
                                          evaluation_strategy=evaluation_strategy, eval_steps=eval_steps, do_eval=True,
                                          do_train=True, num_train_epochs=epochs)
    else:
        training_args = TrainingArguments(output_dir="data", per_device_train_batch_size=train_batch_size,
                                          per_device_eval_batch_size=eval_batch_size, learning_rate=lr,
                                          evaluation_strategy=evaluation_strategy, do_eval=True, do_train=True,
                                          num_train_epochs=epochs)
    trainer = Custom_Trainer(model=model, args=training_args, data_collator=Datacollector(tokenizer),
                             train_dataset=train_dataset,
                             eval_dataset=val_dataset)
    trainer.train()


def optuna_train(trial):
    args = parseargs()
    os.environ["WANDB_DISABLED"] = "true"
    classes = args.classes
    path = 'data/full_data.csv'
    model = LongformerForMultiLabelSequenceClassification.from_pretrained('allenai/longformer-base-4096',
                                                                          attention_window=512,
                                                                          num_labels=classes)
    tokenizer = AutoTokenizer.from_pretrained('allenai/longformer-base-4096')
    train_dataset, val_dataset, test_dataset = process_data(path, args)
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    lr = trial.suggest_float("lr", 5e-5, 5e-3, log=True)
    evaluation_strategy = args.evaluation_strategy
    epochs = trial.suggest_int('epochs', 3, 30)
    if evaluation_strategy == 'steps':
        eval_steps = args.eval_each_x_steps
        training_args = TrainingArguments(output_dir="data", per_device_train_batch_size=train_batch_size,
                                          per_device_eval_batch_size=eval_batch_size, learning_rate=lr,
                                          evaluation_strategy=evaluation_strategy, eval_steps=eval_steps, do_eval=True,
                                          do_train=True, num_train_epochs=epochs)
    else:
        training_args = TrainingArguments(output_dir="data", per_device_train_batch_size=train_batch_size,
                                          per_device_eval_batch_size=eval_batch_size, learning_rate=lr,
                                          evaluation_strategy=evaluation_strategy, do_eval=True, do_train=True,
                                          num_train_epochs=epochs)
    trainer = Custom_Trainer(model=model, args=training_args, data_collator=Datacollector(tokenizer),
                             train_dataset=train_dataset,
                             eval_dataset=val_dataset)
    trainer.train()
    predictions = trainer.predict(val_dataset)

    return predictions.metrics['test_loss']


# train()
study = optuna.create_study(direction='minimize')
study.optimize(optuna_train, n_trials=100)
print("Best trial:", study.best_trial.number)
print("Best accuracy:", study.best_trial.value)
print("Best hyperparameters:", study.best_params)
with open('Best_hyperparameters.txt', 'w') as f:
    f.write('loss')
    f.write(str(study.best_trial.value))
    f.write("parameters")
    f.write(str(study.best_params))
