import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_



def _init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


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


class Attention(nn.Module):
    def __init__(self,
                 dim,  # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,  # 生成qkv 是否使用偏置
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


class Decoder(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 drop_ratio=0.
                 ):
        super(Decoder, self).__init__()
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


class MVQA(nn.Module):
    def __init__(self,
                 depth_img=4,
                 depth_smiles=4,
                 depth_decoder=4,
                 embed_dim=256,
                 protein_dim=256,
                 drop_ratio=0.,
                 backbone="",
                 device=None,
                 ):
        super(MVQA, self).__init__()
        self.device=device

        self.depth_decoder = depth_decoder
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # if backbone == "ResNet18":
        #     self.img_backbone = ResNet(BasicBlock, [2, 2, 2, 2], img_dim=img_dim, include_top=True)
        # elif backbone == "ResNet34":
        #     self.img_backbone = ResNet(BasicBlock, [3, 4, 6, 3], img_dim=img_dim, include_top=True)
        # elif
        if backbone == "CNN":
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

        #  encoder 1
        self.norm_img = norm_layer(embed_dim)
        self.pos_drop_img = nn.Dropout(p=drop_ratio)
        self.pos_embed_img = nn.Parameter(torch.zeros(1, 256, embed_dim))
        dpr_img = [x.item() for x in torch.linspace(0, drop_ratio, depth_img)]
        self.encoder_img = nn.Sequential(*[
            Block(dim=embed_dim,
                  mlp_ratio=4,
                  drop_ratio=dpr_img[i],
                  )
            for i in range(depth_img)
        ])

        #  encoder 2

        self.smiles_embeddings = nn.Embedding(65, embed_dim)
        self.norm_smiles = norm_layer(embed_dim)
        self.pos_drop_smiles = nn.Dropout(p=drop_ratio)
        self.pos_embed_smiles = nn.Parameter(torch.zeros(1, 256, embed_dim))
        dpr_smiles = [x.item() for x in torch.linspace(0, drop_ratio, depth_smiles)]
        self.encoder_smiles = nn.Sequential(*[
            Block(dim=embed_dim,
                  mlp_ratio=4,
                  drop_ratio=dpr_smiles[i],
                  )
            for i in range(depth_smiles)
        ])

        # decoder


        # decoder2
        self.conv1d_img = nn.Sequential(
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
            nn.MaxPool1d(kernel_size=2)
        )

        self.conv1d_smile = nn.Sequential(
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
            nn.MaxPool1d(kernel_size=2)
        )

        self.fc = nn.Linear(992, 2)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward_img_features(self, x):
        B, _, _, _ = x.shape
        # x = self.cnn(x)
        x = self.img_backbone(x)
        x = x.reshape(B, 256, -1)
        x = self.pos_drop_img(x + self.pos_embed_img)
        x = self.encoder_img(x)
        x = self.norm_img(x)
        return x

    def forward_features_smiles(self, x):
        x = self.pos_drop_smiles(x + self.pos_embed_smiles)
        x = self.encoder_smiles(x)
        x = self.norm_smiles(x)
        return x


    def forward(self, inputs):
        image, smile= inputs[0], inputs[1]
        image = image.to(self.device)
        smile = smile.to(self.device)

        B, _, _, _ = image.shape
        image_feature = self.forward_img_features(image) # (B, 256, 256)
        image_feature = self.conv1d_img(image_feature)
        image_feature = image_feature.reshape(B, -1)  # 128*922

        smile_feature = self.smiles_embeddings(smile)  # (B, 256, 256)
        smile_feature = self.forward_features_smiles(smile_feature)
        smile_feature = self.conv1d_smile(smile_feature)
        smile_feature = smile_feature.reshape(B, -1)  # 128*922

        image_feature = image_feature / image_feature.norm(dim=1, keepdim=True)
        smile_feature = smile_feature / smile_feature.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()

        logits_per_image = logit_scale * image_feature @ smile_feature.t()
        logits_per_text = logits_per_image.t()

        return B, logits_per_image, logits_per_text

    def __call__(self, data, train=True):
        inputs = data
        B, logits_per_image, logits_per_text = self.forward(inputs)
        labels = torch.tensor(np.arange(B)).to(self.device)
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels.to(self.device))
        loss = (loss_i + loss_t) / 2
        return loss
