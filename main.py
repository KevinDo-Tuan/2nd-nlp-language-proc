import pandas as pd
from sklearn.preprocessing import LabelEncoder as lb
from datasets import Dataset as dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
# train,test,val,path
train_path = r"c:\Users\Do Pham Tuan\.cache\kagglehub\datasets\praveengovi\emotions-dataset-for-nlp\versions\1\train.txt"
test_path = r"C:\Users\Do Pham Tuan\.cache\kagglehub\datasets\praveengovi\emotions-dataset-for-nlp\versions\1\test.txt"
val_path = r"C:\Users\Do Pham Tuan\.cache\kagglehub\datasets\praveengovi\emotions-dataset-for-nlp\versions\1\val.txt"

# data
train_data = pd.read_csv(train_path, sep=';', header=None, names=['text', 'label'])
test_data = pd.read_csv(test_path, sep=';', header=None, names=['text', 'label'])
val_data = pd.read_csv(val_path, sep=';', header=None, names=['text', 'label'])


print (train_data)

# Encode the labels to train_data, test_data, and val_data
label = lb()
label.fit(train_data['label'])

train_data["id"] = lb.transform(label, train_data['label'])
label.fit (test_data['label'])

test_data["id"] = lb.transform(label, test_data['label'])
label.fit(val_data['label'])
val_data["id"] = lb.transform(label, val_data['label'])

# drop column labeled "label"
train_data = train_data.drop("label", axis=1)
test_data = test_data.drop("label", axis=1)
val_data = val_data.drop("label", axis=1)

# make into data for hugging face models
train_data = dataset.from_pandas(train_data)
test_data = dataset.from_pandas(test_data)
val_data = dataset.from_pandas(val_data)

token_model =BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_data(yup):
    return token_model(yup["text"], padding=True, truncation=True, max_length=512)
train_data = train_data.map(tokenize_data, batched=True)
print(train_data)

#training model:
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label.classes_))

#Training pre-models
pre_models = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="./logs"
)

#Trainer
trainer = Trainer(
    model=model,
    args=pre_models,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=token_model
)
trainer.train()

trainer.save_model("bert-sentiment-model")

print("\nEvaluating on test set...") #Evaluate on Test Set

results = trainer.evaluate(test_data)
print(f"Test Accuracy: {results['eval_accuracy']:.4f}")

example = "I'm feeling happy" # Change this!!
inputs = token_model(example, return_tensors="pt")

# Move inputs to the same device as the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
model.to(device)

outputs = model(**inputs)
pred_class = torch.argmax(outputs.logits).item()
print("\nPrediction for sample input:", example)






