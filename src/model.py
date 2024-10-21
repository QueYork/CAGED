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


class ConditionalVAE(BasicModel):
    def __init__(self):
        super(ConditionalVAE, self).__init__()
        self.__init_model()

    def __init_model(self):
        self.input_dim = board.args.dim
        self.condition_dim = board.args.dim
        self.hidden_dim = board.args.hidden_dim
        self.latent_dim = board.args.latent_dim
        
        self.sc_var = board.args.sc_var
        
        # Encoder
        self.encoder = nn.Sequential()
        for i, dim in enumerate(self.hidden_dim):
            if i == 0:
                self.encoder.add_module(name="L{:d}".format(i), module=nn.Linear(self.input_dim + self.condition_dim, dim))
            else:
                self.encoder.add_module(name="L{:d}".format(i), module=nn.Linear(self.hidden_dim[i-1], dim))
            if i + 1 < len(self.hidden_dim):
                self.encoder.add_module(name="A{:d}".format(i), module=nn.ReLU())    
        print("CVAE Encoder Structure: ", self.encoder)
        
        # Encoder output: z's mean and log var
        self.fc_mu = nn.Linear(self.hidden_dim[-1], self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim[-1], self.latent_dim)

        # Decoder
        self.decoder = nn.Sequential()
        for i, dim in enumerate(self.hidden_dim):
            if i == 0:
                self.decoder.add_module(name="L{:d}".format(i), module=nn.Linear(self.latent_dim + self.condition_dim, dim))
                self.decoder.add_module(name="A{:d}".format(i), module=nn.ReLU()) 
            else: 
                self.decoder.add_module(name="L{:d}".format(i), module=nn.Linear(self.hidden_dim[i-1], dim))
                self.decoder.add_module(name="A{:d}".format(i), module=nn.ReLU()) 
        self.decoder.add_module(name="L{:d}".format(len(self.hidden_dim)), module=nn.Linear(self.hidden_dim[-1], self.input_dim))
        self.decoder.add_module(name="A{:d}".format(len(self.hidden_dim)), module=nn.Sigmoid())
        print("CVAE Decoder Structure", self.decoder)
        
    def encode(self, x, u):
        x_u = torch.cat([x, u], dim=1)  
        h = self.encoder(x_u)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, u):
        z_u = torch.cat([z, u], dim=1)  
        return self.decoder(z_u)
    
    def forward(self, x, u):        # (x, u) pairs
        x_sig = torch.sigmoid(x)
        u_sig = torch.sigmoid(u)
        
        mu, logvar = self.encode(x_sig, u_sig)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, u_sig)
        return recon_x, x_sig, mu, logvar
    
    def loss(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss, kld_loss
    
    def get_scores(self, x, u):  
        recon_x, x_sig, mu, logvar = self.forward(x, u)
        scores = F.mse_loss(recon_x, x_sig, reduction='none') - 0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        scores = torch.sum(scores, dim=1)
        return self.sc_var*(-scores).exp()
    