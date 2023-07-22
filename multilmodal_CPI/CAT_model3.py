import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')




def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)



class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 drop_ratio=0.,
                 ):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(drop_ratio)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop_ratio)


    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x




class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 act_layer=nn.GELU,
                 drop=0.):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 drop_ratio=0.
                 ):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, drop_ratio=drop_ratio)

        self.drop_path = DropPath(drop_ratio) if drop_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            drop=drop_ratio
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Cross_Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 drop_ratio=0.,
                 ):
        super(Cross_Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop_ratio)

    def forward(self, x, y):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv2 = self.qkv(y).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv2[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out



class Smiles_Image_Encoder(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 drop_ratio=0.
                 ):
        super(Smiles_Image_Encoder, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, drop_ratio=drop_ratio)
        self.cross_attn = Cross_Attention(dim, drop_ratio=drop_ratio)

        self.drop_path = DropPath(drop_ratio) if drop_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            drop=drop_ratio
        )

    def forward(self, x, y):

        y = y + self.drop_path(self.attn(self.norm1(y)))
        out = y + self.drop_path(self.cross_attn(x, self.norm1(y)))
        out = out + self.drop_path(self.mlp(self.norm2(y)))
        return out



class CAT2(nn.Module):
    def __init__(self,
                 depth=4,  # depth of transformer
                 mlp_ratio=4.0,  # ratio of mlp hidden dim to embedding dim
                 embed_dim=192,  # embedding dimension
                 drop_ratio=0.,  # dropout rate
                 usemlp=1,
                 dataset_name = "Davis",way="way_1"
                 ):
        super(CAT2, self).__init__()

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.pos_embed_img = nn.Parameter(torch.zeros(1, 256, 256))
        self.pos_embed_pro = nn.Parameter(torch.zeros(1, 256, 256))
        self.pos_embed_smiles = nn.Parameter(torch.zeros(1, 256, 256))
        self.pos_drop_img = nn.Dropout(p=drop_ratio)
        self.pos_drop_pro = nn.Dropout(p=drop_ratio)
        self.pos_drop_smiles = nn.Dropout(p=drop_ratio)

        self.rate1 = torch.nn.Parameter(torch.rand(1))
        self.rate2 = torch.nn.Parameter(torch.rand(1))
        # stochastic depth decay rule
        dpr_img = [x.item() for x in torch.linspace(0, drop_ratio, depth)]
        self.encoder_img = nn.Sequential(*[
            Block(dim=embed_dim,
                  mlp_ratio=mlp_ratio,
                  drop_ratio=dpr_img[i],
                  )
            for i in range(depth)
        ])
        self.norm_img = norm_layer(embed_dim)
        depth_pro = 1
        dpr_pro = [x.item() for x in torch.linspace(0, drop_ratio, depth_pro)]
        self.encoder_pro = nn.Sequential(*[
            Block(dim=embed_dim,
                  mlp_ratio=mlp_ratio,
                  drop_ratio=dpr_pro[i],
                  )
            for i in range(depth_pro)
        ])

        self.norm_pro = norm_layer(embed_dim)

        depth_smiles = 1
        dpr_smiles = [x.item() for x in torch.linspace(0, drop_ratio, depth_smiles)]
        self.encoder_smiles = nn.Sequential(*[
            Block(dim=embed_dim,
                  mlp_ratio=mlp_ratio,
                  drop_ratio=dpr_smiles[i],
                  )
            for i in range(depth_smiles)
        ])

        self.norm_smiles = norm_layer(embed_dim)



        if usemlp==1:
            self.mlp=nn.Identity()
        else:
            self.mlp = Mlp(
                in_features=992,
                hidden_features=2048,
                out_features=992,
                drop=0.2
            )

        self.apply(_init_vit_weights)

        # self.cnn = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(0.02, inplace=True),
        #
        #     nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.LeakyReLU(0.02, inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        # )
        self.img_backbone = nn.Sequential(  # 3*256*256
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),  # 64*128*128
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.02, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64*64*64

            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=2, padding=1),  # 128*32*32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.02, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128*16*16
        )

        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=128, kernel_size=3, stride=2, padding=1),  # 128*128*128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.02, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),  # 256*32*32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.02, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 256*16*16
        )

        self.conv2d_2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=2, padding=1),  # 128*128*128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.02, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),  # 256*32*32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.02, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 256*16*16
        )

        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv2d_3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=2, padding=1),  # 128*128*128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.02, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),  # 256*32*32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.02, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 256*16*16
        )
        self.conv1d_2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.way = way
        self.cp_depth = 2
        self.cppt_depth = 2
        self.cp_norm = norm_layer(embed_dim)
        self.cp_norm2 = norm_layer(embed_dim)
        self.cppt_norm = norm_layer(embed_dim)
        self.cppt_norm2 = norm_layer(embed_dim)
        self.Image_to_text_Encoder = Smiles_Image_Encoder(dim=embed_dim, mlp_ratio=4, drop_ratio=0., )
        self.Text_to_image_Encoder = Smiles_Image_Encoder(dim=embed_dim, mlp_ratio=4, drop_ratio=0., )
        self.Compound_to_protein_Encoder = Smiles_Image_Encoder(dim=embed_dim, mlp_ratio=4, drop_ratio=0., )
        self.Protein_to_compound_Encoder = Smiles_Image_Encoder(dim=embed_dim, mlp_ratio=4, drop_ratio=0., )
        self.fc_interaction = nn.Linear(992, 2)
        self.fc_interaction_C = nn.Linear(3 * 256 * 256, 2)
        self.fc_interaction_D = nn.Linear(2 * 992, 2)

        self.smiles_pad = 0
        if dataset_name == 'Human':
            self.pro_embeddings = nn.Embedding(23, 256)
            # self.smiles_embeddings = nn.Embedding(65, 256)
        elif dataset_name == 'Celegans':
            self.pro_embeddings = nn.Embedding(22, 256)
            # self.smiles_embeddings = nn.Embedding(65, 256)
        elif dataset_name =='Davis':
            self.pro_embeddings = nn.Embedding(22, 256)
            # self.smiles_embeddings = nn.Embedding(65, 256)
        elif dataset_name == "BindingDB":
            self.pro_embeddings = nn.Embedding(21, 256)
            # self.smiles_embeddings = nn.Embedding(65, 256)
        elif dataset_name == "BIOSNAP":
            self.pro_embeddings = nn.Embedding(23, 256)
            # self.smiles_embeddings = nn.Embedding(65, 256)
        self.smiles_embeddings = nn.Embedding(65, 256)


    def forward_img_features(self, x):
        B, _, _, _ = x.shape
        # x = self.cnn(x)
        x = self.img_backbone(x)
        x = x.reshape(B, 256, -1)
        x = self.pos_drop_img(x + self.pos_embed_img)
        x = self.encoder_img(x)
        x = self.norm_img(x)
        return x

    def forward_protein_features(self, x):
        x = self.pos_drop_pro(x + self.pos_embed_pro)
        x = self.encoder_pro(x)
        x = self.norm_pro(x)
        return x

    def forward_smiles_features(self, x):

        x = self.pos_drop_smiles(x + self.pos_embed_smiles)
        x = self.encoder_smiles(x)
        x = self.norm_smiles(x)
        return x


    def forward_compound_feature(self, x, y): # self.forward_compound_feature(cp_smiles_feature,cp_img_feature)
        B, _, _ = x.shape

        # cross1
        out1 = self.Image_to_text_Encoder(x, y)
        for i in range(self.cp_depth - 1):
            out1 = self.Image_to_text_Encoder(x, out1)
        out1 = self.cp_norm(out1)
        out1 = out1.reshape(B, 1, 256, -1)

        # cross2
        out2 = self.Text_to_image_Encoder(y, x)
        for i in range(self.cp_depth - 1):
            out2 = self.Text_to_image_Encoder(y, out2)
        out2 = self.cp_norm2(out2)
        out2 = out2.reshape(B, 1, 256, -1)
        out = torch.cat((out1, out2), 1)

        return out

    # def forward_features_decoder(self, x, y):
    #     B, _, _ = x.shape
    #     y = self.embeddings_decoder(y)
    #     y = self.pos_drop_decoder(y + self.pos_embed_decoder)
    #
    #     # cross1
    #     out1 = self.decoder(x, y)
    #     for i in range(self.depth_decoder - 1):
    #         out1 = self.decoder(x, out1)
    #     out1 = self.norm_decoder(out1)
    #     out1 = out1.reshape(B, 1, 256, -1)
    #
    #     # cross2
    #     out2 = self.decoder2(y, x)
    #     for i in range(self.depth_decoder - 1):
    #         out2 = self.decoder2(y, out2)
    #     out2 = self.norm_decoder2(out2)
    #     out2 = out2.reshape(B, 1, 256, -1)
    #     out = torch.cat((out1, out2), 1)
    #
    #     return out

    def forward_compound_feature2(self, x, y): # self.forward_compound_feature(cp_smiles_feature,cp_img_feature)
        B, _, _ = x.shape

        # cross1
        out1 = self.Image_to_text_Encoder(x, y)
        for i in range(self.cp_depth - 1):
            out1 = self.Image_to_text_Encoder(x, out1)
        out1 = self.cp_norm(out1)
        # out1 = out1.reshape(B, 1, 256, -1)

        # cross2
        out2 = self.Text_to_image_Encoder(y, x)
        for i in range(self.cp_depth - 1):
            out2 = self.Text_to_image_Encoder(y, out2)
        out2 = self.cp_norm2(out2)
        # out2 = out2.reshape(B, 1, 256, -1)
        # out = torch.cat((out1, out2), 1)

        return [out1,out2]

    def forward_compound_protein(self, x, y): # self.forward_compound_feature(cp_smiles_feature,cp_img_feature)
        B, _, _ = x.shape

        # cross1
        out1 = self.Compound_to_protein_Encoder(x, y)
        for i in range(self.cppt_depth - 1):
            out1 = self.Compound_to_protein_Encoder(x, out1)
        out1 = self.cppt_norm(out1)
        out1 = out1.reshape(B, 1, 256, -1)

        # cross2
        out2 = self.Protein_to_compound_Encoder(y, x)
        for i in range(self.cppt_depth - 1):
            out2 = self.Protein_to_compound_Encoder(y, out2)
        out2 = self.cppt_norm2(out2)
        out2 = out2.reshape(B, 1, 256, -1)
        out = torch.cat((out1, out2), 1)

        return out


    # def CSA_forward(self,inputs):
    #     smiles, compound, protein = inputs[0], inputs[1], inputs[2]
    #     B, _, _, _ = compound.shape
    #     smiles = smiles.to(device)
    #     compound = compound.to(device)
    #     protein = protein.to(device)
    #
    #     cp_img_feature = self.forward_img_features(compound)  # (B,256,256)
    #     # cp_img_feature = cp_img_feature.reshape(B, 1, 256, 256)  # (B,1,256,256)
    #
    #     cp_smiles_feature = self.smiles_embeddings(smiles)  # (B,256,256)
    #     cp_smiles_feature = self.forward_smiles_features(cp_smiles_feature)  # (B,256,256)
    #     # cp_smiles_feature = cp_smiles_feature.reshape(B, 1, 256, 256)  # (B,1,256,256)
    #
    #     cp_feature = self.forward_compound_feature(cp_smiles_feature,cp_img_feature)
    #
    #     pt_feature = self.pro_embeddings(protein)  # (B,256,256)
    #     pt_feature = self.forward_pro_features(pt_feature)  # (B,256,256)
    #     pt_feature = pt_feature.reshape(B, 1, 256, 256)  # (B,1,256,256)
    #
    #     feature = torch.cat([cp_feature, pt_feature], dim=1)  # (B,2,256,256)
    #     feature = self.conv2d(feature)  # (B,256,16,16)
    #     feature = feature.reshape(B, 256, -1)  # (B,256,256)
    #     feature = self.conv1d(feature)  # (B,32,31)
    #     feature = feature.reshape(B, -1)  # (64,992)
    #     feature = self.mlp(feature)
    #     interaction = self.fc_interaction(feature)
    #     return interaction



    def forward(self, inputs):
        smiles, img, protein = inputs[0], inputs[1], inputs[2]
        B, _, _, _ = img.shape
        smiles = smiles.to(device)
        img = img.to(device)
        protein = protein.to(device)

        # 原始CAT模型（仅使用化合物的image图像特征）
        if self.way == "way_img":
            cp_img_feature = self.forward_img_features(img) # (B, 256, 256)
            cp_img_feature = cp_img_feature.reshape(B, 1, 256, 256)

            pt_feature = self.pro_embeddings(protein)
            pt_feature = self.forward_protein_features(pt_feature)
            pt_feature = pt_feature.reshape(B,1,256, 256)

            z = torch.cat((cp_img_feature,pt_feature),1)
            z = self.conv2d(z)
            z = z.reshape(B, 256, -1)
            z = self.conv1d(z)
            z = z.reshape(B, -1)
            # z = self.mlp(z)
            interaction = self.fc_interaction(z)
            return interaction

        elif self.way == "way_smiles":
            cp_smiles_feature = self.smiles_embeddings(smiles)  # (B, 256, 256)
            cp_smiles_feature = self.forward_smiles_features(cp_smiles_feature)
            cp_smiles_feature = cp_smiles_feature.reshape(B, 1, 256, 256)

            pt_feature = self.pro_embeddings(protein)
            pt_feature = self.forward_protein_features(pt_feature)
            pt_feature = pt_feature.reshape(B, 1, 256, 256)

            z = torch.cat((cp_smiles_feature, pt_feature), 1)
            z = self.conv2d(z)
            z = z.reshape(B, 256, -1)
            z = self.conv1d(z)
            z = z.reshape(B, -1)
            # z = self.mlp(z)
            interaction = self.fc_interaction(z)
            return interaction

        # smiles和img不经过Cross Attention，reshape后在第一维拼接
        elif self.way == "way_img_smiles":
            cp_img_feature = self.forward_img_features(img)  # (B, 256, 256)
            # cp_img_feature = cp_img_feature.reshape(B, 1, 256, 256)

            cp_smiles_feature = self.smiles_embeddings(smiles)  # (B, 256, 256)
            cp_smiles_feature = self.forward_smiles_features(cp_smiles_feature)  # (B, 256, 256)
            # cp_smiles_feature = cp_smiles_feature.reshape(B,1,256,256)

            cp_img_feature = cp_img_feature.reshape(B, 1, 256, 256)
            cp_smiles_feature = cp_smiles_feature.reshape(B, 1, 256, 256)
            cp_feature = torch.cat([cp_img_feature, cp_smiles_feature], dim=1)
            # cp_feature = cp_feature.reshape(B, 1, 256, 256)

            pt_feature = self.pro_embeddings(protein)
            pt_feature = self.forward_protein_features(pt_feature)
            pt_feature = pt_feature.reshape(B, 1, 256, 256)

            # z = torch.cat((cp_img_feature, pt_feature), 1)
            z = torch.cat((cp_feature, pt_feature), 1)
            z = self.conv2d_2(z)
            z = z.reshape(B, 256, -1)
            z = self.conv1d(z)
            z = z.reshape(B, -1)
            # z = self.mlp(z)
            interaction = self.fc_interaction(z)
            return interaction


        elif self.way == "way_A":
            cp_img_feature = self.forward_img_features(img)  # (B, 256, 256)
            # cp_img_feature = cp_img_feature.reshape(B, 1, 256, 256)

            cp_smiles_feature = self.smiles_embeddings(smiles)  # (B, 256, 256)
            cp_smiles_feature = self.forward_smiles_features(cp_smiles_feature)  # (B, 256, 256)
            # cp_smiles_feature = cp_smiles_feature.reshape(B,1,256,256)

            cp_img_feature = cp_img_feature.reshape(B, 1, 256, 256)
            cp_smiles_feature = cp_smiles_feature.reshape(B, 1, 256, 256)
            cp_feature = cp_img_feature + cp_smiles_feature

            pt_feature = self.pro_embeddings(protein)
            pt_feature = self.forward_protein_features(pt_feature)
            pt_feature = pt_feature.reshape(B, 1, 256, 256)

            # z = torch.cat((cp_img_feature, pt_feature), 1)
            z = torch.cat((cp_feature, pt_feature), 1)
            z = self.conv2d(z)
            z = z.reshape(B, 256, -1)
            z = self.conv1d(z)
            z = z.reshape(B, -1)
            # z = self.mlp(z)
            interaction = self.fc_interaction(z)
            return interaction

        elif self.way == "way_B":
            cp_img_feature = self.forward_img_features(img)  # (B, 256, 256)
            # cp_img_feature = cp_img_feature.reshape(B, 1, 256, 256)

            cp_smiles_feature = self.smiles_embeddings(smiles)  # (B, 256, 256)
            cp_smiles_feature = self.forward_smiles_features(cp_smiles_feature)  # (B, 256, 256)
            # cp_smiles_feature = cp_smiles_feature.reshape(B,1,256,256)

            cp_img_feature = cp_img_feature.reshape(B, 1, 256, 256)
            cp_smiles_feature = cp_smiles_feature.reshape(B, 1, 256, 256)
            cp_feature = cp_img_feature * cp_smiles_feature

            pt_feature = self.pro_embeddings(protein)
            pt_feature = self.forward_protein_features(pt_feature)
            pt_feature = pt_feature.reshape(B, 1, 256, 256)

            # z = torch.cat((cp_img_feature, pt_feature), 1)
            z = torch.cat((cp_feature, pt_feature), 1)
            z = self.conv2d(z)
            z = z.reshape(B, 256, -1)
            z = self.conv1d(z)
            z = z.reshape(B, -1)
            # z = self.mlp(z)
            interaction = self.fc_interaction(z)
            return interaction

        elif self.way == "way_C":
            cp_img_feature = self.forward_img_features(img)  # (B, 256, 256)
            # cp_img_feature = cp_img_feature.reshape(B, 1, 256, 256)

            cp_smiles_feature = self.smiles_embeddings(smiles)  # (B, 256, 256)
            cp_smiles_feature = self.forward_smiles_features(cp_smiles_feature)  # (B, 256, 256)
            # cp_smiles_feature = cp_smiles_feature.reshape(B,1,256,256)

            cp_img_feature = cp_img_feature.reshape(B, 1, 256, 256)
            cp_smiles_feature = cp_smiles_feature.reshape(B, 1, 256, 256)
            cp_feature = torch.cat([cp_img_feature, cp_smiles_feature], dim=1)
            # cp_feature = cp_feature.reshape(B, 1, 256, 256)

            pt_feature = self.pro_embeddings(protein)
            pt_feature = self.forward_protein_features(pt_feature)
            pt_feature = pt_feature.reshape(B, 1, 256, 256)

            # z = torch.cat((cp_img_feature, pt_feature), 1)
            z = torch.cat((cp_feature, pt_feature), 1)

            # z = torch.cat((cp_img_feature, pt_feature), 1)
            z = torch.cat((cp_feature, pt_feature), 1)
            # z = self.conv2d(z)
            # z = z.reshape(B, 256, -1)
            # z = self.conv1d(z)
            z = z.reshape(B, -1)
            # z = self.mlp(z)
            interaction = self.fc_interaction_C(z)
            return interaction

        elif self.way == "way_D":
            cp_img_feature = self.forward_img_features(img)  # (B, 256, 256)
            # cp_img_feature = cp_img_feature.reshape(B, 1, 256, 256)

            cp_smiles_feature = self.smiles_embeddings(smiles)  # (B, 256, 256)
            cp_smiles_feature = self.forward_smiles_features(cp_smiles_feature)  # (B, 256, 256)
            # cp_smiles_feature = cp_smiles_feature.reshape(B,1,256,256)

            cp_img_feature = cp_img_feature.reshape(B, 1, 256, 256)
            cp_smiles_feature = cp_smiles_feature.reshape(B, 1, 256, 256)
            cp_feature = torch.cat([cp_img_feature, cp_smiles_feature], dim=1)
            # cp_feature = cp_feature.reshape(B, 1, 256, 256)

            pt_feature = self.pro_embeddings(protein)
            pt_feature = self.forward_protein_features(pt_feature)
            pt_feature = pt_feature.reshape(B, 1, 256, 256)

            z1 = self.conv2d(cp_feature)
            z1 = z1.reshape(B, 256, -1)
            z1 = self.conv1d(z1)
            z1 = z1.reshape(B, -1) # (1,992)
            z2 = self.conv2d_3(pt_feature)
            z2 = z2.reshape(B, 256, -1)
            z2 = self.conv1d_2(z2)
            z2 = z2.reshape(B, -1)
            # z = torch.cat((cp_img_feature, pt_feature), 1)
            z = torch.cat((z1, z2), 0)

            z = z.reshape(B, -1)
            # z = self.mlp(z)
            interaction = self.fc_interaction_D(z)
            return interaction

        # smiles和img不经过Cross Attention，reshape后在第一维拼接
        elif self.way == "way_img_smiles2":
            cp_img_feature = self.forward_img_features(img)  # (B, 256, 256)
            # cp_img_feature = cp_img_feature.reshape(B, 1, 256, 256)

            cp_smiles_feature = self.smiles_embeddings(smiles)  # (B, 256, 256)
            cp_smiles_feature = self.forward_smiles_features(cp_smiles_feature)  # (B, 256, 256)
            # cp_smiles_feature = cp_smiles_feature.reshape(B,1,256,256)

            cp_img_feature = cp_img_feature.reshape(B, 1, 256, 256)
            cp_smiles_feature = cp_smiles_feature.reshape(B, 1, 256, 256)
            cp_feature = cp_img_feature + cp_smiles_feature
            # cp_feature = cp_feature.reshape(B, 1, 256, 256)

            pt_feature = self.pro_embeddings(protein)
            pt_feature = self.forward_protein_features(pt_feature)
            pt_feature = pt_feature.reshape(B, 1, 256, 256)

            # z = torch.cat((cp_img_feature, pt_feature), 1)
            z = torch.cat((cp_feature, pt_feature), 1)
            z = self.conv2d(z)
            z = z.reshape(B, 256, -1)
            z = self.conv1d(z)
            z = z.reshape(B, -1)
            # z = self.mlp(z)
            interaction = self.fc_interaction(z)
            return interaction


        # smiles和img进行cross attention交互后，将两个模块的输出reshape后在第一维度拼接
        elif self.way == "way_2":
            cp_img_feature = self.forward_img_features(img)  # (B, 256, 256)
            # cp_img_feature = cp_img_feature.reshape(B, 1, 256, 256)

            cp_smiles_feature = self.smiles_embeddings(smiles)  # (B, 256, 256)
            cp_smiles_feature = self.forward_smiles_features(cp_smiles_feature)  # (B, 256, 256)
            # cp_smiles_feature = cp_smiles_feature.reshape(B,1,256,256)

            cp_feature = self.forward_compound_feature(cp_smiles_feature, cp_img_feature)

            pt_feature = self.pro_embeddings(protein)
            pt_feature = self.forward_protein_features(pt_feature)
            pt_feature = pt_feature.reshape(B, 1, 256, 256)

            # z = torch.cat((cp_img_feature, pt_feature), 1)
            z = torch.cat((cp_feature, pt_feature), 1)
            z = self.conv2d_2(z)
            z = z.reshape(B, 256, -1)
            z = self.conv1d(z)
            z = z.reshape(B, -1)
            # z = self.mlp(z)
            interaction = self.fc_interaction(z)
            return interaction

        # smiles和img进行cross attention交互后，取img交互后的输出
        elif self.way == "way_3":
            cp_img_feature = self.forward_img_features(img)  # (B, 256, 256)
            # cp_img_feature = cp_img_feature.reshape(B, 1, 256, 256)

            cp_smiles_feature = self.smiles_embeddings(smiles)  # (B, 256, 256)
            cp_smiles_feature = self.forward_smiles_features(cp_smiles_feature)  # (B, 256, 256)
            # cp_smiles_feature = cp_smiles_feature.reshape(B,1,256,256)

            cp_feature1,cp_feature2 = self.forward_compound_feature2(cp_smiles_feature, cp_img_feature)
            cp_feature1 = cp_feature1.reshape(B,1,256, 256)


            pt_feature = self.pro_embeddings(protein)
            pt_feature = self.forward_protein_features(pt_feature)
            pt_feature = pt_feature.reshape(B, 1, 256, 256)

            # z = torch.cat((cp_img_feature, pt_feature), 1)
            z = torch.cat((cp_feature1, pt_feature), 1)
            z = self.conv2d(z)
            z = z.reshape(B, 256, -1)
            z = self.conv1d(z)
            z = z.reshape(B, -1)
            # z = self.mlp(z)
            interaction = self.fc_interaction(z)
            return interaction


        # smiles和img进行cross attention交互后，取img交互后的输出
        elif self.way == "way_10":
            cp_img_feature = self.img_backbone(img)
            cp_img_feature = cp_img_feature.reshape(B, 256, -1)  # (B, 256, 256)
            cp_img_feature = self.pos_drop_img(cp_img_feature + self.pos_embed_img)
            cp_img_feature = self.norm_img(cp_img_feature)

            cp_smiles_feature = self.smiles_embeddings(smiles)  # (B, 256, 256)
            cp_smiles_feature = self.pos_drop_smiles(cp_smiles_feature + self.pos_embed_smiles)
            cp_smiles_feature = self.norm_smiles(cp_smiles_feature)

            cp_feature1, cp_feature2 = self.forward_compound_feature2(cp_smiles_feature, cp_img_feature)
            cp_feature1 = cp_feature1.reshape(B, 1, 256, 256)

            pt_feature = self.pro_embeddings(protein)
            pt_feature = self.forward_protein_features(pt_feature)
            pt_feature = pt_feature.reshape(B, 1, 256, 256)

            # z = torch.cat((cp_img_feature, pt_feature), 1)
            z = torch.cat((cp_feature1, pt_feature), 1)
            z = self.conv2d(z)
            z = z.reshape(B, 256, -1)
            z = self.conv1d(z)
            z = z.reshape(B, -1)
            # z = self.mlp(z)
            interaction = self.fc_interaction(z)
            return interaction

        # smiles和img经过Cross Attention后，按比例相加
        elif self.way == "way_12":

            cp_img_feature = self.img_backbone(img)
            cp_img_feature = cp_img_feature.reshape(B, 256, -1)  # (B, 256, 256)
            cp_img_feature = self.pos_drop_img(cp_img_feature + self.pos_embed_img)
            # cp_img_feature = self.norm_img(cp_img_feature)

            cp_smiles_feature = self.smiles_embeddings(smiles)  # (B, 256, 256)
            cp_smiles_feature = self.pos_drop_smiles(cp_smiles_feature + self.pos_embed_smiles)
            # cp_smiles_feature = self.norm_smiles(cp_smiles_feature)

            cp_feature1, cp_feature2 = self.forward_compound_feature2(cp_smiles_feature, cp_img_feature)

            cp_feature = self.rate1 * cp_feature1 + self.rate2 * cp_feature2
            cp_feature = cp_feature.reshape(B, 1, 256, 256)

            pt_feature = self.pro_embeddings(protein)
            pt_feature = self.forward_protein_features(pt_feature)
            pt_feature = pt_feature.reshape(B, 1, 256, 256)

            # z = torch.cat((cp_img_feature, pt_feature), 1)
            z = torch.cat((cp_feature, pt_feature), 1)
            z = self.conv2d(z)
            z = z.reshape(B, 256, -1)
            z = self.conv1d(z)
            z = z.reshape(B, -1)
            # z = self.mlp(z)
            interaction = self.fc_interaction(z)
            return interaction

        # smiles和img经过Cross Attention后，再reshape后拼接
        elif self.way == "way_16":

            cp_img_feature = self.img_backbone(img)
            cp_img_feature = cp_img_feature.reshape(B, 256, -1)  # (B, 256, 256)
            cp_img_feature = self.pos_drop_img(cp_img_feature + self.pos_embed_img)
            # cp_img_feature = self.norm_img(cp_img_feature)

            cp_smiles_feature = self.smiles_embeddings(smiles)  # (B, 256, 256)
            cp_smiles_feature = self.pos_drop_smiles(cp_smiles_feature + self.pos_embed_smiles)
            # cp_smiles_feature = self.norm_smiles(cp_smiles_feature)

            cp_feature1, cp_feature2 = self.forward_compound_feature2(cp_smiles_feature, cp_img_feature)

            cp_feature1 = cp_feature1.reshape(B,1, 256, 256)

            cp_feature2 = cp_feature2.reshape(B,1, 256, 256)

            cp_feature = torch.cat([cp_feature1, cp_feature2], dim=1)
            pt_feature = self.pro_embeddings(protein)
            pt_feature = self.forward_protein_features(pt_feature)
            pt_feature = pt_feature.reshape(B, 1, 256, 256)

            # z = torch.cat((cp_img_feature, pt_feature), 1)
            z = torch.cat((cp_feature, pt_feature), 1)
            z = self.conv2d_2(z)
            z = z.reshape(B, 256, -1)
            z = self.conv1d(z)
            z = z.reshape(B, -1)
            # z = self.mlp(z)
            interaction = self.fc_interaction(z)
            return interaction

        # # smiles和img进行cross attention交互后，取smiles交互后的输出
        # elif self.way == "way_4":
        #     cp_img_feature = self.forward_img_features(img)  # (B, 256, 256)
        #     # cp_img_feature = cp_img_feature.reshape(B, 1, 256, 256)
        #
        #     cp_smiles_feature = self.smiles_embeddings(smiles)  # (B, 256, 256)
        #     cp_smiles_feature = self.forward_smiles_features(cp_smiles_feature)  # (B, 256, 256)
        #     # cp_smiles_feature = cp_smiles_feature.reshape(B,1,256,256)
        #
        #     cp_feature1,cp_feature2 = self.forward_compound_feature2(cp_smiles_feature, cp_img_feature)
        #     cp_feature2 = cp_feature2.reshape(B,1,256, 256)
        #
        #
        #     pt_feature = self.pro_embeddings(protein)
        #     pt_feature = self.forward_protein_features(pt_feature)
        #     pt_feature = pt_feature.reshape(B, 1, 256, 256)
        #
        #     # z = torch.cat((cp_img_feature, pt_feature), 1)
        #     z = torch.cat((cp_feature2, pt_feature), 1)
        #     z = self.conv2d(z)
        #     z = z.reshape(B, 256, -1)
        #     z = self.conv1d(z)
        #     z = z.reshape(B, -1)
        #     # z = self.mlp(z)
        #     interaction = self.fc_interaction(z)
        #     return interaction
        #
        # # smiles和img经过Cross Attention后，按比例相加
        # elif self.way == "way_5":
        #     cp_img_feature = self.forward_img_features(img)  # (B, 256, 256)
        #     # cp_img_feature = cp_img_feature.reshape(B, 1, 256, 256)
        #
        #     cp_smiles_feature = self.smiles_embeddings(smiles)  # (B, 256, 256)
        #     cp_smiles_feature = self.forward_smiles_features(cp_smiles_feature)  # (B, 256, 256)
        #     # cp_smiles_feature = cp_smiles_feature.reshape(B,1,256,256)
        #
        #     cp_feature1,cp_feature2 = self.forward_compound_feature2(cp_smiles_feature, cp_img_feature)
        #
        #     cp_feature = self.rate1 * cp_feature1 + self.rate2 * cp_feature2
        #     cp_feature = cp_feature.reshape(B,1,256, 256)
        #
        #
        #     pt_feature = self.pro_embeddings(protein)
        #     pt_feature = self.forward_protein_features(pt_feature)
        #     pt_feature = pt_feature.reshape(B, 1, 256, 256)
        #
        #     # z = torch.cat((cp_img_feature, pt_feature), 1)
        #     z = torch.cat((cp_feature, pt_feature), 1)
        #     z = self.conv2d(z)
        #     z = z.reshape(B, 256, -1)
        #     z = self.conv1d(z)
        #     z = z.reshape(B, -1)
        #     # z = self.mlp(z)
        #     interaction = self.fc_interaction(z)
        #     return interaction
        #
        # # smiles和img不经过cross attention，按比例直接相加
        # elif self.way == "way_6":
        #     cp_img_feature = self.forward_img_features(img)  # (B, 256, 256)
        #     # cp_img_feature = cp_img_feature.reshape(B, 1, 256, 256)
        #
        #     cp_smiles_feature = self.smiles_embeddings(smiles)  # (B, 256, 256)
        #     cp_smiles_feature = self.forward_smiles_features(cp_smiles_feature)  # (B, 256, 256)
        #     # cp_smiles_feature = cp_smiles_feature.reshape(B,1,256,256)
        #
        #     cp_feature = self.rate1 * cp_smiles_feature + self.rate2 * cp_img_feature
        #     cp_feature = cp_feature.reshape(B,1,256, 256)
        #
        #
        #     pt_feature = self.pro_embeddings(protein)
        #     pt_feature = self.forward_protein_features(pt_feature)
        #     pt_feature = pt_feature.reshape(B, 1, 256, 256)
        #
        #     # z = torch.cat((cp_img_feature, pt_feature), 1)
        #     z = torch.cat((cp_feature, pt_feature), 1)
        #     z = self.conv2d(z)
        #     z = z.reshape(B, 256, -1)
        #     z = self.conv1d(z)
        #     z = z.reshape(B, -1)
        #     # z = self.mlp(z)
        #     interaction = self.fc_interaction(z)
        #     return interaction



        # # 只使用化合物的smiles特征，不用图像
        # elif self.way == "way_8":
        #     # cp_img_feature = self.forward_img_features(img)  # (B, 256, 256)
        #     # cp_img_feature = cp_img_feature.reshape(B, 1, 256, 256)
        #
        #     cp_smiles_feature = self.smiles_embeddings(smiles)  # (B, 256, 256)
        #     cp_smiles_feature = self.forward_smiles_features(cp_smiles_feature)  # (B, 256, 256)
        #     cp_smiles_feature = cp_smiles_feature.reshape(B,1,256,256)
        #
        #     # cp_img_feature = cp_img_feature.reshape(B, 1, 256, 256)
        #     # cp_smiles_feature = cp_smiles_feature.reshape(B, 1, 256, 256)
        #     pt_feature = self.pro_embeddings(protein)
        #     pt_feature = self.forward_protein_features(pt_feature)
        #     pt_feature = pt_feature.reshape(B, 1, 256, 256)
        #
        #     z = torch.cat((cp_smiles_feature,pt_feature),dim=1)
        #     z = self.conv2d(z)
        #     z = z.reshape(B, 256, -1)
        #     z = self.conv1d(z)
        #     z = z.reshape(B, -1)
        #     # z = self.mlp(z)
        #     interaction = self.fc_interaction(z)
        #     return interaction
        #
        # # MCL拼接化合物的图像和smiles（之前的way其实多了一层MSA）
        # elif self.way == "way_9":
        #     cp_img_feature = self.img_backbone(img)
        #     cp_img_feature = cp_img_feature.reshape(B,256,-1)# (B, 256, 256)
        #     cp_img_feature = self.pos_drop_img(cp_img_feature + self.pos_embed_img)
        #     cp_img_feature = self.norm_img(cp_img_feature)
        #
        #     cp_smiles_feature = self.smiles_embeddings(smiles)  # (B, 256, 256)
        #     cp_smiles_feature = self.pos_drop_smiles(cp_smiles_feature + self.pos_embed_smiles)
        #     cp_smiles_feature = self.norm_smiles(cp_smiles_feature)
        #
        #     cp_features = self.forward_compound_feature(cp_smiles_feature,cp_img_feature)
        #
        #     pt_feature = self.pro_embeddings(protein)
        #     pt_feature = self.forward_protein_features(pt_feature)
        #     pt_feature = pt_feature.reshape(B, 1, 256, 256)
        #
        #     z = torch.cat((cp_features,pt_feature),dim=1)
        #     z = self.conv2d_2(z)
        #     z = z.reshape(B, 256, -1)
        #     z = self.conv1d(z)
        #     z = z.reshape(B, -1)
        #     # z = self.mlp(z)
        #     interaction = self.fc_interaction(z)
        #     return interaction



        # # smiles和img进行cross attention交互后，取smiles交互后的输出
        # elif self.way == "way_11":
        #     cp_img_feature = self.img_backbone(img)
        #     cp_img_feature = cp_img_feature.reshape(B, 256, -1)  # (B, 256, 256)
        #     cp_img_feature = self.pos_drop_img(cp_img_feature + self.pos_embed_img)
        #     cp_img_feature = self.norm_img(cp_img_feature)
        #
        #     cp_smiles_feature = self.smiles_embeddings(smiles)  # (B, 256, 256)
        #     cp_smiles_feature = self.pos_drop_smiles(cp_smiles_feature + self.pos_embed_smiles)
        #     cp_smiles_feature = self.norm_smiles(cp_smiles_feature)
        #
        #     cp_feature1, cp_feature2 = self.forward_compound_feature2(cp_smiles_feature, cp_img_feature)
        #     cp_feature2 = cp_feature2.reshape(B, 1, 256, 256)
        #
        #     pt_feature = self.pro_embeddings(protein)
        #     pt_feature = self.forward_protein_features(pt_feature)
        #     pt_feature = pt_feature.reshape(B, 1, 256, 256)
        #
        #     # z = torch.cat((cp_img_feature, pt_feature), 1)
        #     z = torch.cat((cp_feature2, pt_feature), 1)
        #     z = self.conv2d(z)
        #     z = z.reshape(B, 256, -1)
        #     z = self.conv1d(z)
        #     z = z.reshape(B, -1)
        #     # z = self.mlp(z)
        #     interaction = self.fc_interaction(z)
        #     return interaction



        # # 化合物的image和smiles MCA后按比例相加再和蛋白质序列cross attention
        # elif self.way == "way_13":
        #     cp_img_feature = self.img_backbone(img)
        #     cp_img_feature = cp_img_feature.reshape(B, 256, -1)  # (B, 256, 256)
        #     cp_img_feature = self.pos_drop_img(cp_img_feature + self.pos_embed_img)
        #     cp_img_feature = self.norm_img(cp_img_feature)
        #
        #     cp_smiles_feature = self.smiles_embeddings(smiles)  # (B, 256, 256)
        #     cp_smiles_feature = self.pos_drop_smiles(cp_smiles_feature + self.pos_embed_smiles)
        #     cp_smiles_feature = self.norm_smiles(cp_smiles_feature)
        #
        #     cp_feature1, cp_feature2 = self.forward_compound_feature2(cp_smiles_feature, cp_img_feature)
        #     cp_feature = self.rate1 * cp_feature1 + self.rate2 * cp_feature2
        #
        #     pt_feature = self.pro_embeddings(protein)
        #     # pt_feature = self.forward_protein_features(pt_feature)
        #     pt_feature = self.pos_drop_pro(pt_feature + self.pos_embed_pro)
        #
        #     z = self.forward_compound_protein(cp_feature,pt_feature)
        #     z = self.conv2d(z)
        #     z = z.reshape(B, 256, -1)
        #     z = self.conv1d(z)
        #     z = z.reshape(B, -1)
        #     # z = self.mlp(z)
        #     interaction = self.fc_interaction(z)
        #     return interaction
        #
        #
        # # 化合物的image和smiles MCA后取img模块的输出再和蛋白质序列cross attention
        # elif self.way == "way_14":
        #     cp_img_feature = self.img_backbone(img)
        #     cp_img_feature = cp_img_feature.reshape(B, 256, -1)  # (B, 256, 256)
        #     cp_img_feature = self.pos_drop_img(cp_img_feature + self.pos_embed_img)
        #     cp_img_feature = self.norm_img(cp_img_feature)
        #
        #     cp_smiles_feature = self.smiles_embeddings(smiles)  # (B, 256, 256)
        #     cp_smiles_feature = self.pos_drop_smiles(cp_smiles_feature + self.pos_embed_smiles)
        #     cp_smiles_feature = self.norm_smiles(cp_smiles_feature)
        #
        #     cp_feature1, cp_feature2 = self.forward_compound_feature2(cp_smiles_feature, cp_img_feature)
        #
        #     pt_feature = self.pro_embeddings(protein)
        #     # pt_feature = self.forward_protein_features(pt_feature)
        #     pt_feature = self.pos_drop_pro(pt_feature + self.pos_embed_pro)
        #
        #     z = self.forward_compound_protein(cp_feature2, pt_feature)
        #     z = self.conv2d(z)
        #     z = z.reshape(B, 256, -1)
        #     z = self.conv1d(z)
        #     z = z.reshape(B, -1)
        #     # z = self.mlp(z)
        #     interaction = self.fc_interaction(z)
        #     return interaction
        #
        # # 原始的MCL论文，化合物的图像和smiles拼接后MCL
        # elif self.way == "way_15":
        #     cp_img_feature = self.forward_img_features(img)  # (B, 256, 256)
        #     # cp_img_feature = cp_img_feature.reshape(B, 1, 256, 256)
        #
        #     cp_smiles_feature = self.smiles_embeddings(smiles)  # (B, 256, 256)
        #     cp_smiles_feature = self.forward_smiles_features(cp_smiles_feature)  # (B, 256, 256)
        #     # cp_smiles_feature = cp_smiles_feature.reshape(B,1,256,256)
        #
        #     cp_feature = self.rate1 * cp_smiles_feature + self.rate2 * cp_img_feature
        #
        #     pt_feature = self.pro_embeddings(protein)
        #     pt_feature = self.pos_drop_pro(pt_feature + self.pos_embed_pro)
        #     pt_feature = self.norm_pro(pt_feature)
        #     # pt_feature = self.forward_protein_features(pt_feature)
        #
        #     z = self.forward_compound_protein(cp_feature, pt_feature)
        #     z = self.conv2d(z)
        #     z = z.reshape(B, 256, -1)
        #     z = self.conv1d(z)
        #     z = z.reshape(B, -1)
        #     # z = self.mlp(z)
        #     interaction = self.fc_interaction(z)
        #     return interaction
        #
        # elif self.way == "way_16":
        #     # smiles_mask = get_pad_mask(smiles,self.smiles_pad)
        #     #
        #     # src_img_mask = torch.ones(img.shape[0], 1, 256)
        #     # src_img_mask = src_img_mask.eq(1)
        #     # src_img_mask = src_img_mask.cuda()
        #
        #
        #     cp_smiles_feature = self.smiles_embeddings(smiles)  # (B, 256, 256)
        #     cp_smiles_feature = self.pos_drop_smiles(self.position_enc(cp_smiles_feature))
        #     cp_smiles_feature = self.norm_smiles(cp_smiles_feature)
        #
        #
        #     cp_img_feature = self.forward_img_features(img)
        #
        #     for enc_layer in self.layer_stack:
        #         smiles_enc_output, enc_slf_attn = enc_layer(cp_smiles_feature)
        #
        #     # ？？？？？？？没有写完？？？？？？？
        #     print()





        # B, _, _, _ = compound.shape
        # compound = compound.to(device)
        # protein = protein.to(device)
        # cp_feature = self.forward_features(compound)
        # cp_feature = cp_feature.reshape(B, 1, 256, 256)
        # pt_feature = self.embeddings(protein)
        # pt_feature = self.forward_features2(pt_feature)
        # pt_feature = pt_feature.reshape(B, 1, 256, 256)
        # z = torch.cat((cp_feature, pt_feature), 1)
        # z = self.conv2d(z)
        # z = z.reshape(B, 256, -1)
        # z = self.conv1d(z)
        # z = z.reshape(B, -1)
        # # z = self.mlp(z)
        # interaction = self.fc_interaction(z)
        # return interaction

    def __call__(self, data, train=True):
        inputs, correct_interaction = data[:-1], data[-1]
        predicted_interaction = self.forward(inputs)
        correct_interaction = torch.squeeze(correct_interaction)
        loss = F.cross_entropy(predicted_interaction, correct_interaction.to(device))
        correct_labels = correct_interaction.to('cpu').data.numpy()
        ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
        predicted_labels = list(map(lambda x: np.argmax(x), ys))
        predicted_scores = list(map(lambda x: x[1], ys))
        return loss, correct_labels, predicted_labels, predicted_scores


class Train_model(object):
    def __init__(self, model, lr, weight_decay):
        self.model = model
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    def train(self, dataset):
        loss, _, _, _ = self.model(dataset)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.to('cpu').data.numpy()

    def save_model(self, model, filename):
        torch.save(model, filename)


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, data):
        (loss, correct_labels, predicted_labels,
         predicted_scores) = self.model(data, train=False)
        return loss, correct_labels, predicted_labels, predicted_scores

    def save_AUCs(self, AUCs, file_name):
        with open(file_name, "a+") as f:
            line = "epoch: " + str(AUCs[0]) + \
                   "    Batch: " + str(AUCs[1]) + \
                   "    loss: " + str(AUCs[2]) + \
                   "    AUC: " + str(AUCs[3]) + \
                   "    AUPRC: " + str(AUCs[7]) + \
                   "    Precision: " + str(AUCs[4]) + \
                   "    Recall: " + str(AUCs[5]) + \
                   "    F1: " + str(AUCs[6]) + \
                   "    time: " + str(AUCs[8]) + \
                   "\n"
            f.write(line)

