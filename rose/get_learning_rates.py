import json
import pandas as pd
import sys



file_path = sys.argv[1]

with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)


log_history = data['log_history']
df = pd.DataFrame(log_history)


bins = [0, 1, 2, 3, 4]  
labels = ['0-1', '1-2', '2-3', '3-4']  

df['epoch_group'] = pd.cut(df['epoch'], bins=bins, labels=labels, right=True)


grouped_df = df.groupby('epoch_group').mean()
learning_rate = grouped_df["learning_rate"].tolist()



for lr in learning_rate:
    print(lr)

