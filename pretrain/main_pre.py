import os
import shutil
import torch
import argparse
from datetime import datetime
from pretrain_model import MVQA
from Train_pre import Train_model, Tester
from utils_pre import data_loader, get_img_path

from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve, auc, average_precision_score




def model_run(args):
    # time
    global m_path
    ISOTIMEFORMAT = '%Y_%m%d_%H%M'
    run_time = datetime.now().strftime(ISOTIMEFORMAT)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"数据集：{args.dataset_name} 运行时间: {run_time}")

    # ********************************* Pre_dataset *********************************
    pre_img_path = "/data/lxy/Experiment/pretrain/data/cheml/pic/Img_256_256" + "/img_inf_data"
    pre_image = get_img_path(pre_img_path)
    pre_smile_name = "/data/lxy/Experiment/pretrain/data/cheml/smiles/chem_smiles.npy"
    train_dataset, train_loader = data_loader(batch_size=args.batch_size,
                                              imgs=pre_image,
                                              smile_name=pre_smile_name)

    # ********************************* create model *********************************

    torch.manual_seed(2)
    model = MVQA(depth_smiles=args.depth_e1,
                 depth_img=args.depth_e2,
                 depth_decoder=args.depth_decoder,
                 embed_dim=args.embed_dim,
                 protein_dim=args.protein_dim,
                 drop_ratio=args.drop_ratio,
                 backbone=args.backbone,
                 device=device,
                 ).to(device)

    # ********************************* training *********************************
    lr, lr_decay, weight_decay = map(float, [1e-3, args.lr_decay, 1e-8])

    # ConvNext
    # lr, weight_decay = map(float, [5e-4, 5e-2])

    trainer = Train_model(model, lr, weight_decay)

    print("开始训练....")
    min_loss = 999
    for epoch in range(1, args.epochs + 1):
        if epoch % 10 == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        total_loss = []

        for i, data_train in enumerate(train_loader):
            if data_train[0].shape[0] <= 1:
                break

            loss_train = trainer.train(data_train)
            total_loss.append(loss_train)
            if (i + 1) % 50 == 0:
                print(
                    f"Training "
                    f"[Epoch {epoch}/{args.epochs}] "
                    f"[Batch  {i}/{len(train_loader)}] "
                    f"[batch_size {data_train[0].shape[0]}] "
                    f"[loss_train : {loss_train}]")

        # model save
        model_path = "/data/lxy/Experiment/pretrain/data/cheml/model/"
        # if os.path.exists(model_path):
        #     shutil.rmtree(model_path)

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        model_file = model_path + "epoch:"+str(epoch) + "-- dataset: Chem " + "-- loss: " + str(
                sum(total_loss) / len(train_loader)) + "--  lr_decay: " + str(args.lr_decay) + "-- depth:" + str(
                args.depth_e1) + "  batch_size: " + str(args.batch_size) + "  device: " + str(
                args.device) + " pre_model.model"
        # if (epoch-1) % 10 ==0 and sum(total_loss) / len(train_loader) < min_loss:
        torch.save(model, model_file)
        print("model save :" + model_file)

        # if sum(total_loss) / len(train_loader) < min_loss:
        #     min_loss = total_loss
        #     torch.save(model, model_path + run_time +" dataset: MySmiles "+ "-- loss: " + str(
        #         sum(total_loss) / len(train_loader)) + "--  lr_decay: " + str(args.lr_decay) + "-- depth:" + str(
        #         args.depth_e1) + "  batch_size: " + str(args.batch_size) + "  device: " + str(
        #         args.device) + " pre_model.model")

        print("Epoch: ", epoch, " SUM(loss): ", sum(total_loss), " len(data): ", len(train_loader),
              "     Train平均loss: ",
              sum(total_loss) / len(train_loader))

        # print("模型地址：", model_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, default="cheml")
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--img_size', type=int, default=256)

    parser.add_argument('--k', type=int, default=1)

    parser.add_argument('--backbone', type=str, default="CNN")

    # dim
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--img_dim', type=int, default=2)
    parser.add_argument('--fingure_dim', type=int, default=64)
    parser.add_argument('--smile_dim', type=int, default=64)
    parser.add_argument('--protein_dim', type=int, default=256)

    # depth
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--depth_e1', type=int, default=1)
    parser.add_argument('--depth_e2', type=int, default=1)
    parser.add_argument('--depth_decoder', type=int, default=2)
    parser.add_argument('--lr_decay', type=float, default=0.8)
    parser.add_argument('--drop_ratio', type=float, default=0.)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument("--device", type=str, default='cuda:0')
    opt = parser.parse_args()
    model_run(opt)
