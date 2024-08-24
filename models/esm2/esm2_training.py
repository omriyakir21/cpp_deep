import torch
from transformers import AutoTokenizer, TrainingArguments
from datasets import Dataset
from setfit import SetFitModel, SetFitTrainer
from torch.optim import Adam

class ModelTrainer:
    def __init__(self, model_name, sentences, labels, output_dir, num_train_epochs, per_device_train_batch_size, per_device_eval_batch_size, warmup_steps, weight_decay, logging_dir, logging_steps):
        self.model_name = model_name
        self.sentences = sentences
        self.labels = labels
        self.output_dir = output_dir
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.logging_dir = logging_dir
        self.logging_steps = logging_steps
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = SetFitModel.from_pretrained(model_name, use_differentiable_head=True)

    def train(self):
        # Tokenize the input sentences
        encodings = self.tokenizer(self.sentences, truncation=True, padding=True, max_length=512)

        # Create a dataset object
        dataset = Dataset.from_dict({'input_ids': encodings['input_ids'], 'attention_mask': encodings['attention_mask'], 'labels': self.labels})

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,          # output directory
            num_train_epochs=self.num_train_epochs,              # number of training epochs
            per_device_train_batch_size=self.per_device_train_batch_size,   # batch size for training
            per_device_eval_batch_size=self.per_device_eval_batch_size,    # batch size for evaluation
            warmup_steps=self.warmup_steps,                # number of warmup steps for learning rate scheduler
            weight_decay=self.weight_decay,               # strength of weight decay
            logging_dir=self.logging_dir,            # directory for storing logs
            logging_steps=self.logging_steps,          # log every x steps
        )

        # Initialize the Adam optimizer
        optimizer = Adam(self.model.parameters(), lr=1e-4)

        # Initialize the SetFitTrainer
        trainer = SetFitTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset,  # Assuming you want to use the same dataset for evaluation
            optimizers=(optimizer, None)  # Pass the optimizer to the trainer
        )

        # Train the model
        trainer.train()

if __name__ == '__main__':
    sentences = ["Example sentence 1", "Example sentence 2"]
    labels = [0, 1]
    model_name = "bert-base-uncased"
    output_dir = './results'
    num_train_epochs = 3
    per_device_train_batch_size = 8
    per_device_eval_batch_size = 8
    warmup_steps = 500
    weight_decay = 0.01
    logging_dir = './logs'
    logging_steps = 10

    trainer = ModelTrainer(model_name, sentences, labels, output_dir,
                            num_train_epochs, per_device_train_batch_size,
                              per_device_eval_batch_size, warmup_steps, weight_decay, logging_dir, logging_steps)
    trainer.train()