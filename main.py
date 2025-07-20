import pandas as pd
from sklearn.preprocessing import LabelEncoder as lb

# train,test,val,path
train_path = r"c:\Users\Do Pham Tuan\.cache\kagglehub\datasets\praveengovi\emotions-dataset-for-nlp\versions\1\train.txt"
test_path = r"C:\Users\Do Pham Tuan\.cache\kagglehub\datasets\praveengovi\emotions-dataset-for-nlp\versions\1\test.txt"
val_path = r"C:\Users\Do Pham Tuan\.cache\kagglehub\datasets\praveengovi\emotions-dataset-for-nlp\versions\1\val.txt"

train_data = pd.read_csv(train_path, sep=';', header=None, names=['text', 'label'])


print (data)

train_data = 



