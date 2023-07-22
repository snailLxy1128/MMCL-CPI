import os
from rdkit.Chem import Draw
from rdkit import Chem
import numpy as np

def smile2pic(file_path, file_data):
    with open(file_data, "r") as f:
        data_list = f.read().strip().split("\n")
    """Exclude data contains '.' in the SMILES format."""
    data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    smiles = []
    for i, data in enumerate(data_list):
        if i % 1000 == 0:
            print('/'.join(map(str, [i + 1, len(data_list)])))
        smile = data.strip().split(" ")[0]
        mol = Chem.MolFromSmiles(smile)
        img = Draw.MolToImage(mol, size=(pic_size, pic_size), wedgeBonds=False)
        number = str(i + 1)
        number = number.zfill(len(str(len(data_list))))
        smiles += smile
        save_name = file_path + "/" + number + ".png"
        img.save(save_name)

def pic_info(file_path):
    file_list = os.listdir(file_path)
    num = 0
    for pic in file_list:
        if ".png" in pic:
            num += 1
    str_len = len(str(num))
    print(str_len)
    print(file_path)
    with open(file_path + "/pic_inf_data", "w") as f:
        for i in range(num):
            number = str(i + 1)
            number = number.zfill(len(str(len(file_list))))
            if i == num - 1:
                f.write(file_path + "/" + number + ".png" + "\t" + number + ".png")
            else:
                f.write(file_path + "/" + number + '.png' + "\t" + number + '.png' + "\n")

def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    import re
    # pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|Na|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|Na|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return tokens

def make_vocab(vocab_path,train_file,test_file,val_file):

    vocab = set()
    with open(train_file, "r") as f:
        data_list = f.read().strip().split("\n")
    """Exclude data contains '.' in the SMILES format."""
    data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    for i, data in enumerate(data_list):
        # if i % 100 == 0:
        #     print('/'.join(map(str, [i + 1, len(data_list)])))
        smile = data.strip().split(" ")[0]
        smile_list = smi_tokenizer(smile)
        for item in smile_list:
            vocab.add(item)
    with open(test_file, "r") as f:
        data_list = f.read().strip().split("\n")
    """Exclude data contains '.' in the SMILES format."""
    data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    for i, data in enumerate(data_list):
        # if i % 100 == 0:
        #     print('/'.join(map(str, [i + 1, len(data_list)])))
        smile = data.strip().split(" ")[0]
        smile_list = smi_tokenizer(smile)
        for item in smile_list:
            vocab.add(item)
    with open(val_file, "r") as f:
        data_list = f.read().strip().split("\n")
    """Exclude data contains '.' in the SMILES format."""
    data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    for i, data in enumerate(data_list):
        # if i % 100 == 0:
        #     print('/'.join(map(str, [i + 1, len(data_list)])))
        smile = data.strip().split(" ")[0]
        smile_list = smi_tokenizer(smile)
        for item in smile_list:
            vocab.add(item)
    with open(vocab_path,"w+") as f:
        f.write("<PAD>"+"\n")
        for item in vocab:
            f.write(item+"\n")
    # print(vocab)
    print("smile dict size:"+str(len(vocab)))

def smiles_pro(dataset_name, data_file,vocab_path,type):
    # vocab_path =
    vocab = [token.strip() for token in open(vocab_path)]
    vocab_t2i = {vocab[i]: i for i in range(len(vocab))}
    vocab_i2t = {v: k for k, v in vocab_t2i.items()}

    # print(vocab_t2i)

    with open(data_file, "r") as f:
        data_list = f.read().strip().split("\n")
    """Exclude data contains '.' in the SMILES format."""
    data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    pad_len = 256
    smiles = []
    max_len = 0
    for i, data in enumerate(data_list):
        # if i % 100 == 0:
        #     print('/'.join(map(str, [i + 1, len(data_list)])))
        smile = data.strip().split(" ")[0]
        smile_list = list(smile)#smi_tokenizer(smile)
        # 在列表首添加<S>
        smile_list.insert(0, "<S>")
        if len(smile_list) > max_len:
            max_len = len(smile_list)
            max_smiles = "".join(smile_list)
        while (len(smile_list) < 256):
            smile_list.append("<PAD>")
        smile_list = smile_list[:256]
        # 在列表尾添加<CLS>
        smile_list.append("<CLS>")
        # 列表总长为257
        smile_index = [vocab_t2i[token] for token in smile_list]
        smiles.append(smile_index)
    smiles_file = "data/" + dataset_name + "/input/" + dataset_name + "_" + type + "_new_smiles2"

    np.save(smiles_file,smiles)
    print("finish!")

if __name__ == '__main__':
    dataset_name = "BindingDB"
    print("dataset_name: " + dataset_name)
    pic_size = 256
    data_root = "data/" + dataset_name
    train_file = data_root + "/" + dataset_name + "_train.txt"
    test_file = data_root + "/" + dataset_name + "_test.txt"
    val_file = data_root + "/" + dataset_name + "_val.txt"
    train_path = data_root + "/train/"
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    test_path = data_root + "/test/"
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    val_path = data_root + "/val/"
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    pic_train_path = train_path + "Pic_" + str(pic_size) + "_" + str(pic_size)
    if not os.path.exists(pic_train_path):
        os.makedirs(pic_train_path)
    pic_test_path = test_path + "Pic_" + str(pic_size) + "_" + str(pic_size)
    if not os.path.exists(pic_test_path):
        os.makedirs(pic_test_path)
    pic_val_path = val_path + "Pic_" + str(pic_size) + "_" + str(pic_size)
    if not os.path.exists(pic_val_path):
        os.makedirs(pic_val_path)
    smile2pic(pic_train_path, train_file)
    print("Train_Pic generated.size=", pic_size, "*", pic_size, "----")
    smile2pic(pic_test_path, test_file)
    print("Test_Pic generated.size=", pic_size, "*", pic_size, "----")
    smile2pic(pic_val_path, val_file)
    print("Val_Pic generated.size=", pic_size, "*", pic_size, "----")
    pic_info(pic_train_path)
    pic_info(pic_test_path)
    pic_info(pic_val_path)

    # -------------------------add ----------------------------------
    vocab_path = data_root + "/vocab.token"
    make_vocab(vocab_path, train_file, test_file, val_file)
    types = {"train", "val", "test"}

    vocab_path = "/data/lxy/Experiment/pretrain/data/cheml/vocab_coca.token"

    for type in types:
        data_file = "data/" + dataset_name + "/" + dataset_name + "_" + type + ".txt"
        smiles_pro(dataset_name, data_file, vocab_path, type=type)
