import json

path1 = '/root/CS570-Final-Project/datasets/imgnet1k_our/train0.json'
path2 = '/root/CS570-Final-Project/datasets/imgnet1k_our/train1_naive.json'

with open(path1, "rb") as f:
    data1 = json.load(f)['data']
with open(path2, "rb") as f:
    data2 = json.load(f)['data']

# check whether the same
for i in range(len(data1)):
    if data1[i]['img_path'] != data2[i]['img_path']:
        print('different')
        break
    for x in data1[i]['labels']:
        if x not in data2[i]['labels']:
            print('different')
            break
    for x in data2[i]['labels']:
        if x not in data1[i]['labels']:
            print('different')
            break
print('same')