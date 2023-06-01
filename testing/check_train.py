import json

path1 = '/root/CS570-Final-Project/datasets/imgnet1k_original/val.json'
path2 = '/root/CS570-Final-Project/datasets/imgnet1k_original/hyperparam_search/random_3_fixed_mc_dropout_majority_0.6.json'

with open(path1, "rb") as f:
    data1 = json.load(f)['data']
with open(path2, "rb") as f:
    data2 = json.load(f)['data']

for i in range(len(data2)):
    img_path = data2[i]['img_path']
    for j in range(len(data1)):
        if data1[j]['img_path'] == img_path:
            if data1[j]['labels'] != data2[i]['labels']:
                print(img_path)
                print(data1[j]['labels'])
                print(data2[i]['labels'])
                print('-----------------------')
            break
