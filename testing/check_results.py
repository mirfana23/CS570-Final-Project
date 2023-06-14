import pickle

path = 'results4.pkl'
results = pickle.load(open(path, 'rb'))

def check_requirements(key, requirements):
    for idx, val in requirements:
        if key[idx] != val:
            return False
    return True

def get_best_results(results, metric, requirements, k=3):
    items = list(results.items())
    # remove the naive
    # items = [item for item in items if item[0][3] == 'naive']
    items = [item for item in items if check_requirements(item[0], requirements)]
    items.sort(key=lambda x: x[1][metric], reverse=True)
    return items[:k]

# print(results[('random', 3, 'range', 'mc_dropout', 'majority', 0.2)])

requirements = [
    (3, 'mc_perturbation'),
    # (3, 'mc_dropout'),
    # (3, 'naive'),
]

metrics = [
    'Average jaccard similarity',
    'Average precision',
    'Average recall',
    'Average f1 score',
]

for metric in metrics:
    print(metric)
    # for requirement in requirements:
    #     print(requirement)
    print(get_best_results(results, metric, requirements, k=1))
    print()

# print('best results for Average jaccard similarity: ', get_best_results(results, 'Average jaccard similarity'))
# print('best results for Average precision: ', get_best_results(results, 'Average precision'))
# print('best results for Average recall: ', get_best_results(results, 'Average recall'))
# print('best results for Average f1 score: ', get_best_results(results, 'Average f1 score'))

# update the keys
# new_results = {}
# for key, value in results.items():
#     key_list = list(key)
#     if key[3] == 'mc_dropout':
#         continue
#     new_results[tuple(key_list)] = value

# # save
# pickle.dump(new_results, open(path, 'wb'))