"""
@author:chenyankai, queyue
@file:model.py
@time:2024/6/28
"""
import torch
import torch.nn as nn
import src.powerboard as board
import src.data_loader as data_loader
import numpy as np
import math


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def get_scores(self, user_index):
        raise NotImplementedError


class BiGeaR_tch(BasicModel):
    def __init__(self, dataset):
        super(BiGeaR_tch, self).__init__()
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

