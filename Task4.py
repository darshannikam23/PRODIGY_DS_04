import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob 

tw = pd.read_csv("twitter_training.csv")
#print(tw)

col_names = ['ID', 'Entity', 'Sentiment', 'Content']
tw1 = pd.read_csv("twitter_training.csv", names = col_names)
print(tw1)

#print(tw1.shape)
#print(tw1.describe)

print(tw1.isnull().sum())
tw1.dropna(axis=0, inplace=True)
print(tw1.isnull().sum())

print(tw1.duplicated().sum())
tw1.drop_duplicates(inplace=True)
print(tw1.duplicated().sum())

sentiments_counts = tw1['Sentiment'].value_counts()
print(sentiments_counts)

plt.figure(figsize = (8,4))
sentiments_counts.plot(kind = 'bar', color = ['red', 'blue', 'orange', 'green'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.xticks(rotation = 0)
plt.show()

brand_data = tw1[tw1['Entity'].str.contains('Amazon', case=False)]
brand_sentiment_counts = brand_data['Sentiment'].value_counts()
print(brand_sentiment_counts)

plt.figure(figsize=(6, 6))
plt.pie(brand_sentiment_counts, labels=brand_sentiment_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Sentiment Distribution for Amazon')
plt.show()
