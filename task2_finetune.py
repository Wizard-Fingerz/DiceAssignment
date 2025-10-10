"""
task2_finetune.py
Fine-tune DistilBERT for emotion/sentiment labels.
Requires: transformers, datasets, torch, sklearn
Run with GPU for reasonable speed.
"""

from datasets import load_dataset, ClassLabel, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./distil_finetuned_freshers"
BATCH_SIZE = 16
EPOCHS = 3

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "precision_macro": precision_score(labels, preds, average="macro"),
        "recall_macro": recall_score(labels, preds, average="macro"),
    }

def main():
    # Use the dair-ai/emotion dataset
    raw = load_dataset("dair-ai/emotion")
    # Limit each split to 2000 examples (or fewer if split is smaller)
    max_samples = 2000
    limited = {}
    for split in raw.keys():
        limited[split] = raw[split].select(range(min(len(raw[split]), max_samples)))
    # The dataset has splits: train, validation, test
    # The label column is already integer-encoded, and text is in "text"
    num_labels = len(limited["train"].features["label"].names)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    tokenized = {}
    for split in limited.keys():
        tokenized[split] = limited[split].map(tokenize, batched=True)
        tokenized[split] = tokenized[split].rename_column("label", "labels")
        tokenized[split].set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print("Training finished. Model saved to", OUTPUT_DIR)

if __name__ == "__main__":
    main()
