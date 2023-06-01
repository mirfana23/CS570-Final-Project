import pickle
import yaml
import os
from tqdm import tqdm

# default_config = {
#     'dataset': {
#         'json_dir': '/root/CS570-Final-Project/datasets/imgnet1k_original/val.json',
#         'num_workers': 0
#     },
#     'load_from_checkpoint': False,
#     'device': 'cuda',
#     'relabeled_dataset_save_path': '/root/CS570-Final-Project/datasets/imgnet1k_original/val_relabelled.json',

#     'crop_method': 'random',
#     'num_crops': 5,
#     'crop_size_method': 'fixed',

#     'confidence_measure': 'mc_dropout',
#     'dropout_rate': 0.05,
#     'post_proc_logits': 'average',
#     'threshold': 0.0
# }

relabeller_path = '/root/CS570-Final-Project/src/run/relabeller.py'
eval_path = '/root/CS570-Final-Project/testing/relabeller.py'

def get_results(crop_method, num_crops, crop_size_method, confidence_measure, post_proc_logits, threshold):
    # make a config yaml file
    # copy the default config
    # save the config
    os.system(f'python {eval_path} --pred_dataset_path /root/CS570-Final-Project/datasets/imgnet1k_original/hyperparam_search/{crop_method}_{num_crops}_{crop_size_method}_{confidence_measure}_{post_proc_logits}_{threshold}.json --gt_dataset_path /root/CS570-Final-Project/datasets/imgnet1k_ReaL/val.json --exclude_val /root/CS570-Final-Project/datasets/imgnet1k_original/val.json > /root/CS570-Final-Project/testing/eval_output.txt')
    output = open('/root/CS570-Final-Project/testing/eval_output.txt', 'r').read()

    items = output.split(',')
    keys, values = [], []
    for item in items:
        key, value = item.split(':')
        keys.append(key.strip())
        values.append(float(value.strip()))
    
    return dict(zip(keys, values))


options = {
    'crop_method': [
       'random', #'rpn'
    ],
    'num_crops': [
        3, 5, 10
    ],
    'crop_size_method': [
        'fixed', 'range',
    ],
    'confidence_measure': [
        'mc_perturbation', 'mc_dropout', 'naive'
        # 'mc_dropout'
    ],
    'post_proc_logits': [
        'average', 'majority'
    ],
    'threshold': [
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
    ]
}

results = {}

result_file_path = 'results5.pkl'
# if the file exists
if os.path.exists(result_file_path):
    results = pickle.load(open(result_file_path, 'rb'))

RUN_AGAIN = False

for crop_method in tqdm(options['crop_method']):
    for num_crops in tqdm(options['num_crops']):
        for crop_size_method in tqdm(options['crop_size_method']):
            for confidence_measure in tqdm(options['confidence_measure']):
                for post_proc_logits in tqdm(options['post_proc_logits']):
                    for threshold in tqdm(options['threshold']):
                        key = (crop_method, num_crops, crop_size_method, confidence_measure, post_proc_logits, threshold)
                        if key in results and not RUN_AGAIN:
                            print('skipping', key)
                            continue
                        print('running', key)
                        metrics = get_results(*key)
                        results[key] = metrics
                        pickle.dump(results, open(result_file_path, 'wb'))

