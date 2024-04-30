import os
import random

import torch
import argparse
from datetime import datetime
from utils2 import data_loader2, get_pic_path
# from CAT_model import CAT, Train_model, Tester
from CAT_model3 import CAT2, Train_model, Tester
from CAT_model_img import CAT_img
from CAT_model_smiles import CAT_smiles
import time

from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve, auc, average_precision_score

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # "cuda:0"
else:
    device = torch.device("cpu")


def model_run(args):
    # time
    global m_path
    ISOTIMEFORMAT = '%Y_%m%d_%H%M'
    run_time = datetime.now().strftime(ISOTIMEFORMAT)

    # Par
    # device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    decay_interval = 10

    print("数据集：", args.dataset_name, run_time)
    lr, lr_decay, weight_decay = map(float, [1e-3, args.lr_decay, 1e-6])

    resul_name = "result/" + args.dataset_name
    if not os.path.exists(resul_name):
        os.makedirs(resul_name)

    file_AUCs_test = resul_name + "/" + run_time + " " +args.way+" CAT_final vocab2 " + args.dataset_name + " premodel_ 冻结：" + str(args.froizen) + " load"+args.pre+" lr_decay" + str(
        args.lr_decay) + " Batchsize: " + str(args.batch_size) +" depth:"+str(args.depth)+" dr:"+ str(args.drop)+" NEW  .txt"
    # file_AUCs_test = resul_name + "/" + run_time + " " + args.way  + args.dataset_name + "冻结：" + str(
    #     args.froizen) + " Batchsize: " + str(args.batch_size) + " "+args.pre_modelfile +" .txt"

    # ********************************* Train_dataset *********************************
    train_pic_path = "data/" + args.dataset_name + "/train/" + "Pic_" + str(args.pic_size) + "_" + str(
        args.pic_size) + "/pic_inf_data"
    train_pic = get_pic_path(train_pic_path)

    train_protein_name = "data/" + args.dataset_name + "/input/" + args.dataset_name + "_train_proteins.npy"
    train_itr_name = "data/" + args.dataset_name + "/input/" + args.dataset_name + "_train_interactions.npy"
    train_smiles_name = "data/" + args.dataset_name + "/input/" + args.dataset_name + "_train_smiles2.npy"
    train_dataset, train_loader = data_loader2(args.batch_size, train_smiles_name, train_pic, train_protein_name,
                                               train_itr_name)


    print(train_dataset)

    # ********************************* Test_dataset *********************************
    test_pic_path = "data/" + args.dataset_name + "/test/" + "Pic_" + str(args.pic_size) + "_" + str(
        args.pic_size) + "/pic_inf_data"
    test_pic = get_pic_path(test_pic_path)

    test_protein_name = "data/" + args.dataset_name + "/input/" + args.dataset_name + "_test_proteins.npy"
    test_itr_name = "data/" + args.dataset_name + "/input/" + args.dataset_name + "_test_interactions.npy"
    test_smiles_name = "data/" + args.dataset_name + "/input/" + args.dataset_name + "_test_smiles2.npy"

    test_dataset, test_loader = data_loader2(args.batch_size, test_smiles_name, test_pic, test_protein_name,
                                             test_itr_name)

    # ********************************* Val_dataset *********************************
    val_pic_path = "data/" + args.dataset_name + "/val/" + "Pic_" + str(args.pic_size) + "_" + str(
        args.pic_size) + "/pic_inf_data"
    val_pic = get_pic_path(val_pic_path)

    val_protein_name = "data/" + args.dataset_name + "/input/" + args.dataset_name + "_val_proteins.npy"
    val_itr_name = "data/" + args.dataset_name + "/input/" + args.dataset_name + "_val_interactions.npy"
    val_smiles_name = "data/" + args.dataset_name + "/input/" + args.dataset_name + "_val_smiles2.npy"

    val_dataset, val_loader = data_loader2(args.batch_size, val_smiles_name, val_pic, val_protein_name, val_itr_name)


    torch.manual_seed(2)
    if args.pre == "all":
        model = CAT2(embed_dim=args.embed_dim,
                    depth=args.depth,
                    drop_ratio=args.drop,
                    usemlp=args.mlp_flag, dataset_name=args.dataset_name,way=args.way
                    ).to(device)
    # 加载img的参数
    elif args.pre == "img":
        model = CAT_img(embed_dim=args.embed_dim,
                     depth=args.depth,
                     drop_ratio=args.drop,
                     usemlp=args.mlp_flag, dataset_name=args.dataset_name, way=args.way
                     ).to(device)

    # 加载smiles的参数
    elif args.pre =="smiles":
        model = CAT_smiles(embed_dim=args.embed_dim,
                     depth=args.depth,
                     drop_ratio=args.drop,
                     usemlp=args.mlp_flag, dataset_name=args.dataset_name, way=args.way
                     ).to(device)

    if args.froizen != None:
        model_dict = model.state_dict()
        pre_model = torch.load(args.pre_modelfile)
        pre_model = pre_model.to(device)
        # for index, (name, param) in enumerate(pre_model.named_parameters()):
        #     print(str(index) + " " + name)
        pretrained_dict = pre_model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        with open(file_AUCs_test, "a+") as f:
            f.write(args.pre_modelfile+"\n")


        if args.froizen == True and args.pre == "all":
            for index, (name, param) in enumerate(model.named_parameters()):
                print(str(index) + " " + name)
            for i, p in enumerate(model.parameters()):
                if (i == 0 ) or( i ==2 ) or (i>=5 and i<=17) or (i>=31 and i<=51) :
                        p.requires_grad = False
        if args.froizen == True and args.pre == "img":
            for index, (name, param) in enumerate(model.named_parameters()):
                print(str(index) + " " + name)
            for i, p in enumerate(model.parameters()):
                if (i == 0 ) or (i>=5 and i<=17) or (i>=44 and i<=51) :
                        p.requires_grad = False
        if args.froizen == True and args.pre == "smiles":
            for index, (name, param) in enumerate(model.named_parameters()):
                print(str(index) + " " + name)
            for i, p in enumerate(model.parameters()):
                if (i == 2) or (i >= 31 and i <= 43):
                    p.requires_grad = False

    trainer = Train_model(model, lr, weight_decay)

    for epoch in range(1, args.epochs + 1):
        # 训练
        start_time = time.time()
        print("training  Epoch: " + str(epoch))
        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        total_loss = []

        for i, data_train in enumerate(train_loader):
            if data_train[0].shape[0] <= 1:
                break

            loss_train = trainer.train(data_train)  #
            total_loss.append(loss_train)
            if (i + 1) % 50 == 0:
                print(
                    "Training [Epoch %d/%d] [Batch %d/%d] [batch_size %d] [loss_train : %f]"
                    % (epoch, args.epochs, i, len(train_loader), data_train[0].shape[0], loss_train)
                )

        model_path = "/data/lxy/Experiment/CAT-CPI/" + args.dataset_name + "/output/model/"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        file_model = model_path + run_time + "_" + str(epoch) + ".model"
        trainer.save_model(model, file_model)
        print("平均loss", sum(total_loss) / len(train_loader))
        # print("模型地址：", file_model)
        with torch.no_grad():
            val(args, file_model, val_loader)
            test(args, file_model, test_dataset, test_loader, file_AUCs_test,start_time)


def val(args, file_model, val_loader):
    torch.manual_seed(2)
    model = torch.load(file_model)
    valer = Tester(model)
    Loss, y_label, y_pred, y_score = [], [], [], []
    for i, data_list in enumerate(val_loader):
        loss, correct_labels, predicted_labels, predicted_scores = valer.test(data_list)
        Loss.append(loss)
        for c_l in correct_labels:
            y_label.append(c_l)
        for p_l in predicted_labels:
            y_pred.append(p_l)
        for p_s in predicted_scores:
            y_score.append(p_s)

    loss_val = sum(Loss) / len(val_loader)
    AUC_val = roc_auc_score(y_label, y_score)
    fpr, tpr, thresholds = roc_curve(y_label, y_score)
    # AUC_test = auc(fpr, tpr)

    AUPRC = average_precision_score(y_label, y_score)

    precision_val = precision_score(y_label, y_pred)
    recall_val = recall_score(y_label, y_pred)
    f1_score = (2 * precision_val * recall_val) / (recall_val + precision_val + 0.0001)
    print(
        "Valing  batch_size %d  [loss : %.3f] [AUC : %.3f] [AUPRC : %.3f] [precision : %.3f] [recall : %.3f] [F1 : %.3f] "
        % (args.batch_size, loss_val, AUC_val, AUPRC, precision_val, recall_val, f1_score)
    )


def test(args, file_model, test_dataset, test_loader, file_AUCs_test, start_time):
    torch.manual_seed(2)
    model = torch.load(file_model)
    tester = Tester(model)
    Loss, y_label, y_pred, y_score = [], [], [], []
    for i, data_list in enumerate(test_loader):
        loss, correct_labels, predicted_labels, predicted_scores = tester.test(data_list)
        Loss.append(loss)
        for c_l in correct_labels:
            y_label.append(c_l)
        for p_l in predicted_labels:
            y_pred.append(p_l)
        for p_s in predicted_scores:
            y_score.append(p_s)

    loss_test = sum(Loss) / len(test_loader)
    AUC_test = roc_auc_score(y_label, y_score)
    fpr, tpr, thresholds = roc_curve(y_label, y_score)
    # AUC_test = auc(fpr, tpr)

    AUPRC = average_precision_score(y_label, y_score)

    precision_test = precision_score(y_label, y_pred)
    recall_test = recall_score(y_label, y_pred)
    f1_score = (2 * precision_test * recall_test) / (recall_test + precision_test + 0.0001)
    end_time = time.time()
    run_time = end_time - start_time
    print(
        "Testing  batch_size %d  [loss : %.3f] [AUC : %.3f] [AUPRC : %.3f] [precision : %.3f] [recall : %.3f] [F1 : %.3f] "
        % (args.batch_size, loss_test, AUC_test, AUPRC, precision_test, recall_test, f1_score)
    )
    print()
    AUCs = [len(test_dataset),
            len(test_loader),
            format(loss_test, '.3f'),
            format(AUC_test, '.3f'),
            format(precision_test, '.3f'),
            format(recall_test, '.3f'),
            format(f1_score, ".3f"),
            format(AUPRC, '.3f'),
            format(run_time, ".5f")]
    tester.save_AUCs(AUCs, file_AUCs_test)


def prepare_input(resolution):
    smiles = torch.LongTensor(128, 3, 128, 128).to(device)
    protein = torch.LongTensor(128, 64, 64).to(device)
    return dict(smiles=smiles, protein=protein)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, default="BindingDB")
    parser.add_argument('--pre_modelfile', type=str, default=None, help='pretrained model file')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--pic_size', type=int, default=256)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--drop', type=float, default=0.)
    parser.add_argument('--lr_decay',type=float,default=0.8)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--dim_compound', type=int, default=64)
    parser.add_argument('--dim_protein', type=int, default=64)
    parser.add_argument("--device", default='cuda:0')
    parser.add_argument("--mlp_flag", type=int, default=1)
    parser.add_argument('--froizen', type=bool, default=None,help='None means that the pre-training parameters are not loaded, False means that the pre-training parameters are used and fine-tuned, and True means that the pre-training parameters are frozen')

    parser.add_argument('--way', type=str, default="way_img_smiles")
    parser.add_argument('--pre', type=str, default="all")

    opt = parser.parse_args()
    model_run(opt)

