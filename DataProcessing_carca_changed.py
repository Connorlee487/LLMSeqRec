import gzip
from collections import defaultdict
from datetime import datetime
import array
import numpy as np
import pickle
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

def load_data(filename):
    try:
        with open(filename, "rb") as f:
            x= pickle.load(f)
    except:
        x = []
    return x

def save_data(data,filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

countU = defaultdict(lambda: 0)
countP = defaultdict(lambda: 0)
line = 0

dataset_name = 'Beauty'
f = open('reviews_' + dataset_name + '.txt', 'w')
for l in parse('reviews_' + dataset_name + '.json.gz'):
    line += 1
    f.write(" ".join([l['reviewerID'], l['asin'], str(l['overall']), str(l['unixReviewTime'])]) + ' \n')
    asin = l['asin']
    rev = l['reviewerID']
    time = l['unixReviewTime']
    countU[rev] += 1
    countP[asin] += 1
f.close()

usermap = dict()
usernum = 0
itemmap = dict()
itemnum = 0
User = dict()
for l in parse('reviews_' + dataset_name + '.json.gz'):
    line += 1
    asin = l['asin']
    rev = l['reviewerID']
    time = l['unixReviewTime']
    if countU[rev] < 5 or countP[asin] < 5:
        continue

    if rev in usermap:
        userid = usermap[rev]
    else:
        usernum += 1
        userid = usernum
        usermap[rev] = userid
        User[userid] = []
    if asin in itemmap:
        itemid = itemmap[asin]
    else:
        itemnum += 1
        itemid = itemnum
        itemmap[asin] = itemid
    User[userid].append([time, itemid])

# sort reviews in User according to time
for userid in User.keys():
    User[userid].sort(key=lambda x: x[0])

print(usernum, itemnum)

f = open(dataset_name + '_cxt.txt', 'w')
for user in User.keys():
    for i in User[user]:
        f.write('%d %d %s\n' % (user, i[1], i[0]))
f.close()

f = open(dataset_name + '.txt', 'w')
for user in User.keys():
    for i in User[user]:
        f.write('%d %d\n' % (user, i[1]))
f.close()

#### Reading and writing features
itemfeat_dict = {}
counter = 0
for l in parse('meta_' + dataset_name + '.json.gz'):
    line += 1
    asin = l['asin']

    title = ""
    if 'description' in l.keys():
        title = l['description']

    price = 0.0
    if 'price' in l.keys():
        price = float(l['price'])       

    brand = ""
    if 'brand' in l.keys():
        brand = l['brand']

    categories = l['categories'][0]
    if asin in itemmap.keys():
        itemid = itemmap[asin]
        itemfeat_dict[itemid] = [title, price, brand, categories]
        counter = counter + 1

features_list = list()
templist = ["", 0.0, "", []]
for item_id in range(1, itemnum+1):
    if item_id in itemfeat_dict.keys():
        features_list.append(itemfeat_dict[item_id])
    else:
        features_list.append(templist)

df = pd.DataFrame(features_list, columns=['title','price','brand','categories'])

# df.to_csv('features_list_carca.csv')

# model_name = "Qwen/Qwen3-Embedding-0.6B"
# model_name = "google-bert/bert-base-uncased"
model_name = "Qwen/Qwen3-Embedding-4B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
    return last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

embeddings = []
for idx, row in df.iterrows():
    parts = [
        str(row.get('title', '')),
        str(row.get('brand', '')),
        " ".join(map(str, row.get('categories', [])))
    ]
    text_input = " | ".join([p for p in parts if p.strip() != ""])
    embedding = get_embedding(text_input)
    embeddings.append(embedding)

# Convert embeddings to DataFrame
df = pd.DataFrame(embeddings)
print(df.shape)

# Save final feature matrix
save_data(df.values, dataset_name + '_feat_5.dat')
