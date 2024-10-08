"""
@author:chenyankai, queyue
@file:model.py
@time:2024/6/28
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import src.powerboard as board
import src.data_loader as data_loader
import numpy as np
import math


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def get_scores(self, user_index):
        raise NotImplementedError


class LightGCN(BasicModel):
    def __init__(self, dataset):
        super(LightGCN, self).__init__()
        self.dataset: data_loader.LoadData = dataset

        self.__init_model()

    def __init_model(self):
        self.num_users = self.dataset.get_num_users()
        self.num_items = self.dataset.get_num_items()
        self.dim = board.args.dim
        self.num_layers = board.args.layers
        self.user_embed = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.dim)
        self.item_embed = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.dim)
        
        nn.init.normal_(self.user_embed.weight, std=0.1)
        nn.init.normal_(self.item_embed.weight, std=0.1)
        board.cprint('initializing with NORMAL distribution.')

        self.f = nn.Sigmoid()
        self.Graph = self.dataset.load_sparse_graph()


    # Embedding aggregation
    def aggregate_embed(self): 
        user_embed = self.user_embed.weight # [#users, dim]
        item_embed = self.item_embed.weight # [#items, dim]

        con_original_embed = torch.cat([user_embed, item_embed], dim=0) # [#users + #items, dim]
        con_embed_list = [con_original_embed] # [1, #users + #items, dim]

        for layer in range(self.num_layers):
            con_original_embed = torch.sparse.mm(self.Graph, con_original_embed) # [#users + #items, dim]
            con_embed_list.append(con_original_embed) # [1 + layer, #users + #items, dim]
        
        con_embed_list = torch.stack(con_embed_list, dim=1)
        return con_embed_list[:self.num_users, :], con_embed_list[self.num_users:, :] # [n_entity, num_layer + 1, dim]
    
    
    def pooling(self, embed: torch.Tensor):
        # return embed.view(embed.shape[:-2] + (-1,))  # concate pooling
        return torch.mean(embed, dim=1)  # avg pooling
    
    
    def _BPR_loss(self, user_embed, pos_embed, neg_embed):
        pos_scores = torch.mul(user_embed, pos_embed)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(user_embed, neg_embed)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        return loss
    
    
    def loss(self, user_index, pos_index, neg_index):
        user_emb, item_emb = self.aggregate_embed()
        
        nocon_u_e = user_emb[user_index]
        nocon_pos_e = item_emb[pos_index]
        nocon_neg_e = item_emb[neg_index]
        
        user_emb = self.pooling(user_emb)
        item_emb = self.pooling(item_emb)
        
        u_e = user_emb[user_index]
        pos_e = item_emb[pos_index]
        neg_e = item_emb[neg_index]
        
        reg_loss = 0.5 * (torch.norm(nocon_u_e[:, 0, :]) ** 2
                       + torch.norm(nocon_pos_e[:, 0, :]) ** 2
                       + torch.norm(nocon_neg_e[:, 0, :]) ** 2) / float(len(user_index))
        
        loss1 = self._BPR_loss(u_e, pos_e, neg_e)
        return loss1, reg_loss
    

    def get_scores(self, user_index):
        all_user_embed, all_item_embed = self.aggregate_embed()
        
        all_user_embed = self.pooling(all_user_embed)
        all_item_embed = self.pooling(all_item_embed)
        
        user_embed = all_user_embed[user_index.long()]
        scores = self.f(torch.matmul(user_embed, all_item_embed.t()))
        return scores


class ConditionalVAE(nn.Module):
    def __init__(self):
        """
        CVAE的构造函数，定义网络结构
        :param input_dim: 输入x的维度
        :param condition_dim: 条件u的维度
        :param hidden_dim: 编码器/解码器隐藏层的维度
        :param latent_dim: 潜在变量z的维度
        """
        super(ConditionalVAE, self).__init__()
        
        self.input_dim = board.args.dim
        self.condition_dim = board.args.dim
        self.hidden_dim = board.args.hidden_dim
        self.latent_dim = board.args.latent_dim
        
        # 编码器部分：将x和u拼接在一起作为输入
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim + self.condition_dim, self.hidden_dim),
            nn.ReLU()
        )
        
        # 均值和对数方差
        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.latent_dim)

        # 解码器部分：将z和u拼接在一起作为输入
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim + self.condition_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
            nn.Sigmoid()  # 输出范围[0, 1]
        )

    def encode(self, x, u):
        """
        编码器：将x和u一起编码到潜在空间
        :param x: 输入数据
        :param u: 条件数据
        :return: 潜在空间的均值mu和对数方差logvar
        """
        x_u = torch.cat([x, u], dim=1)  # 将x和u拼接
        h = self.encoder(x_u)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        重参数化采样
        :param mu: 均值
        :param logvar: 对数方差
        :return: 潜在变量z
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, u):
        """
        解码器：将z和u一起解码为重构的x
        :param z: 潜在变量
        :param u: 条件数据
        :return: 重构的数据
        """
        z_u = torch.cat([z, u], dim=1)  # 将z和u拼接
        return self.decoder(z_u)

    def forward(self, x, u):
        """
        前向传播：编码 -> 重参数化 -> 解码
        :param x: 输入数据
        :param u: 条件数据
        :return: 重构的x，均值mu和对数方差logvar
        """
        mu, logvar = self.encode(x, u)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, u)
        return recon_x, mu, logvar
    
    def loss(self, recon_x, x, mu, logvar):
        """
        计算CVAE的损失函数：重构损失和KL散度
        :param recon_x: 重构的数据
        :param x: 原始输入数据
        :param mu: 潜在空间的均值
        :param logvar: 潜在空间的对数方差
        :return: 总损失（ELBO）
        """
        # 重构损失
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        # KL散度
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + kld_loss