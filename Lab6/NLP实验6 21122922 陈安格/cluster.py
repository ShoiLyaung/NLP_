from sklearn.cluster import AffinityPropagation
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN



classify_num = 3
news = fetch_20newsgroups(subset='all')
texts = []
ans = [[] for i in range(20)]
j = 0
for i in range(1000):
    if (news.target[i] in [0, 7, 13]):
        texts.append(news.data[i])
        ans[news.target[i]].append(j)
        j += 1


# print(texts)

vectorizer= TfidfVectorizer()
X_test = vectorizer.fit_transform(texts)
clustering = KMeans(n_clusters=classify_num, random_state=0)
clustering.fit(X_test)

labels = clustering.labels_

ans_list = [[] for i in range(20)]
label_list = [[] for i in range(20)]
for i in range(len(labels)):
    ans_list[news['target'][i]].append(i)
    label_list[labels[i]].append(i)



# print(labels)
# print(news['target'][:1000])
for i in range(20):
    print(ans[i])

print('------------------------------------------------------------------')
for i in range(classify_num):
    print(label_list[i])
max_same = [0 for i in range(classify_num)]
for i in range(classify_num):
    max_same[i] = 0
    for j in range(20):
        max_same[i] = max(max_same[i], len(set(label_list[i]) & set(ans[j])))
print(sum(max_same) / sum([len(ans[i]) for i in range(20)]))

#
# from transformers import AutoTokenizer, AutoModel
# import torch
#
# model_name = 'bert-base-uncased'
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)
# texts = news.data[:100]
#
# input_ids = []
# attention_masks = []
# for text in texts:
#
#     encoded_dict = tokenizer.encode_plus(text,
#                                          add_special_tokens=True,
#                                          max_length=64,
#                                          padding='max_length',
#                                          return_attention_mask=True,
#                                          return_tensors='pt')
#
#     input_ids.append(encoded_dict['input_ids'])
#     attention_masks.append(encoded_dict['attention_mask'])
#
# input_ids = torch.cat(input_ids, dim=0)
# attention_masks = torch.cat(attention_masks, dim=0)
#
#
# with torch.no_grad():
#
#     outputs = model(input_ids, attention_masks)
#
#     features = outputs[0][:, 0, :].numpy()
#
# from sklearn.cluster import KMeans
#
#
# num_clusters = 20
#
# kmeans = KMeans(n_clusters=num_clusters, random_state=0)
# kmeans.fit(features)
