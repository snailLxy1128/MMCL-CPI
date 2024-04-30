from main import run


# IMG+SMILES	IMG, SMILES	  Frozen
datas = ["BindingDB"]

# lr_decay = [0.8,0.85,0.90,0.95]
lr_decay = [0.8]
drop_ration = [0.15]

froizen_list = [None]
# pre_model = ["all","img","smiles"]

# 哪些模块使用预训练参数: all, smiles, img
pre_model = ["all"]

# 模型中化合物的输入: way_img_smiles, way_img, way_smiles
# way_list = ["way_img_smiles","way_img","way_smiles","way_A","way_B","way_C","way_D"]
way_list = ["way_img_smiles"]

for pre_modelfile in ["/data/lxy/Experiment/pretrain/data/cheml/model/epoch:1-- dataset: Chem -- loss: 0.4343904047636595--  lr_decay: 0.8-- depth:1  batch_size: 512  device: cuda:0 pre_model.model"]:
    for data in datas:
        for lr in lr_decay:
            for froizen in froizen_list:
                for pre in pre_model:
                    for dr in drop_ration:
                        for way in way_list:
                            for i in range(2):
                                run(data, batch_size=128, lr_decay=lr, drop_ratio=dr, pic_size=256,froizen=froizen,pre_modelfile=pre_modelfile,depth=1,way = way,pre=pre)

