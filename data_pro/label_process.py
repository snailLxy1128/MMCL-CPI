import os
import numpy as np


def label_save(data_file):
    with open(data_file, 'r') as f:
        data_list = f.read().strip().split('\n')
    """Exclude data contains '.' in the SMILES format."""
    data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    N = len(data_list)
    print("label----dataset_size:", N)
    positive_num = 0
    negative_num = 0
    interactions = []
    for i, data in enumerate(data_list):
        interaction = data.strip().split(" ")[2]
        label = interaction.split('.')[0]
        interaction = float(interaction)
        interactions.append(np.array([interaction], dtype=np.float32))
        if int(label) == 1:
            positive_num += 1
        elif int(label) == 0:
            negative_num += 1
    print("positive nums:", positive_num)
    print("negative nums:", negative_num)
    label_name = data_root + "/input/" + dataset_name + "_" + data_file.split("_")[1].split(".")[0] + '_interactions'
    np.save(label_name, interactions)


if __name__ == '__main__':
    dataset_name = "BIOSNAP"
    print("dataset_name: " + dataset_name)
    data_root = "data/" + dataset_name
    train_file = data_root + "/" + dataset_name + "_train.txt"
    test_file = data_root + "/" + dataset_name + "_test.txt"
    val_file = data_root + "/" + dataset_name + "_val.txt"
    input_file = data_root + "/input/"
    if not os.path.exists(input_file):
        os.makedirs(input_file)
    label_save(train_file)
    label_save(test_file)
    label_save(val_file)
