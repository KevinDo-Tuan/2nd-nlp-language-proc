import pandas as pd
from sklearn.preprocessing import LabelEncoder as lb
from io import StringIO

path = r"c:\Users\Do Pham Tuan\.cache\kagglehub\datasets\praveengovi\emotions-dataset-for-nlp\versions\1\train.txt"

data = pd.read_csv(path, column = ["text", "label"])

print (data)



