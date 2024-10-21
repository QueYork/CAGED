import src.powerboard as board
import numpy as np
import torch
import src.utils as utils
from torch import optim
import scipy.sparse as sp
import os

class GCNLoss:
    def __init__(self, model):
        self.model = model
        self.weight_decay = board.args.weight
        self.lr = board.args.lr
        self.opt = optim.Adam(model.parameters(), lr=self.lr)

    def stage(self, user_index, pos_index, neg_index):
        loss, reg_loss = self.model.loss(user_index, pos_index, neg_index)

        reg_loss *= self.weight_decay
        loss += reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()
    
class VAELoss:
    def __init__(self, model):
        self.model = model
        self.lr = board.args.lr2
        self.opt = optim.Adam(model.parameters(), lr=self.lr)

    def stage(self, x, u):
        recon_x, x_sig, mu, logvar = self.model(x, u)
        recon, kld = self.model.loss(recon_x, x_sig, mu, logvar)
        loss = recon + kld
        self.opt.zero_grad()
        loss.backward(retain_graph=True)
        self.opt.step()

        return loss.cpu().item(), recon.cpu().item(), kld.cpu().item()

def Train_vae(data_loader, vae, user_emb, item_emb, loss_f, batch_size, epoch):          
    vae.train()

    loss = []
    recon = []
    kld = []
    for batch in data_loader:
        user, item = batch
        user = user.to(device=board.DEVICE)
        item = item.to(device=board.DEVICE)
        
        loss_user_side, recon_user_side, kld_user_side = loss_f.stage(x=item_emb[item], u=user_emb[user])
        loss.append(loss_user_side)
        recon.append(recon_user_side)
        kld.append(kld_user_side)
        
        loss_item_side, recon_item_side, kld_item_side = loss_f.stage(x=user_emb[user], u=item_emb[item])
        loss.append(loss_item_side)
        recon.append(recon_item_side)
        kld.append(kld_item_side)
    
    avg_loss = np.mean(loss) / batch_size
    avg_recon = np.mean(recon) / batch_size
    avg_kld  = np.mean(kld) / batch_size
    
    info = f'Avg loss: {avg_loss: .6f} | Avg recon loss: {avg_recon: .6f} | Avg kld loss: {avg_kld: .6f}'
    
    # vae.eval()
    # with torch.no_grad():
    #     print(f'CVAE EPOCH[{epoch + 1}/{num_epoch}]: Avg loss: {avg_loss: .3f} | Avg recon loss: {avg_recon: .3f} | Avg kld loss: {avg_kld: .3f}')
    #     Is = [3264, 3361, 1251, 933, 998, 3784, 3657, 1192, 1640, 812, 110, 592, 1232, 2803, 918, 2615, 3896, 857]
    #     u = user_emb[11].repeat((len(Is), 1))
    #     i = item_emb[torch.LongTensor(Is).to(board.DEVICE)]
    #     print(vae.get_scores(i, u))
    # vae.train()
    
    return info
    
def Weight_Inference(vae, user_emb, item_emb, data_loader):
    vae.eval()
    
    # Initial empty adj matrix
    GraphSize = torch.Size([user_emb.shape[0] + item_emb.shape[0], user_emb.shape[0] + item_emb.shape[0]])  
    Graph = torch.sparse.FloatTensor(torch.LongTensor([[], []]), torch.FloatTensor([]), GraphSize).to(device=board.DEVICE)
    
    for batch in data_loader:
        user, item = batch
        user = user.to(device=board.DEVICE)
        item = item.to(device=board.DEVICE)
        
        # Scores condition on user/item
        with torch.no_grad():
            scores_con_user = vae.get_scores(x=item_emb[item], u=user_emb[user])
            scores_con_item = vae.get_scores(x=user_emb[user], u=item_emb[item])
        
        # Concate 2 types of scores
        item_coors = item + user_emb.shape[0]
        row_coordinate = torch.concat((user, item_coors))
        col_coordinate = torch.concat((item_coors, user))
        scores = torch.concat((scores_con_user, scores_con_item))
        
        SubGraph = torch.sparse.FloatTensor(torch.stack([row_coordinate, col_coordinate]), scores, GraphSize)
        Graph += SubGraph
    return Graph
    
def Train_full(dataset, model, epoch, loss_f, neg_ratio=1, summarizer=None):
    model.train()

    with utils.timer(name='Sampling'):
        samples = utils.uniform_sampler(dataset=dataset, neg_ratio=neg_ratio)
    user_index = torch.Tensor(samples[:, 0]).long()
    pos_item_index = torch.Tensor(samples[:, 1]).long()
    neg_item_index = torch.Tensor(samples[:, 2]).long()

    user_index, pos_item_index, neg_item_index = utils.shuffle(user_index, pos_item_index, neg_item_index)

    user_index = user_index.to(device=board.DEVICE)
    pos_item_index = pos_item_index.to(device=board.DEVICE)
    neg_item_index = neg_item_index.to(device=board.DEVICE)
 
    num_batch = len(user_index) // board.args.train_batch + 1
    avg_loss = 0.

    if board.args.model in ['gcn']:
        batch = utils.minibatch(user_index, pos_item_index, neg_item_index, batch_size=board.args.train_batch)
        for batch_i, (b_user_idx, b_pos_item_idx, b_neg_item_idx) in enumerate(batch):
            loss_all_i = loss_f.stage(b_user_idx, b_pos_item_idx, b_neg_item_idx)
            avg_loss += loss_all_i

        avg_loss /= num_batch
        time_info = utils.timer.dict()
        utils.timer.zero()
        
        info = f'all_loss:{avg_loss: .3f} | time cost-{time_info}|'
        if board.args.tensorboard:
            summarizer.add_scalar(f'GCNLoss/Overall_loss', avg_loss, epoch)
            
    else:
        raise NotImplementedError('Wrong model type selection for training!')

    return info


def batch_infer(tensors):
    true_items = tensors[0]
    pred_items = tensors[1].numpy()
    hit_data = utils.get_hit_data(true_items, pred_items)
    pre, recall, ndcg = [], [], []

    for k in board.args.topks:
        recall_k, pre_k = utils.Recall_Precision_K(true_items, hit_data, k)
        pre.append(pre_k)
        recall.append(recall_k)
        ndcg.append(utils.NDCG_K(true_items, hit_data, k))

    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg)}


def Inference(dataset, model, epoch, summarizer=None):
    test_batch_size = board.args.test_batch
    model.eval()
    max_k = max(board.args.topks)
    test_dict = dataset.get_test_dict()
    results = {'precision': np.zeros(len(board.args.topks)),
               'recall': np.zeros(len(board.args.topks)),
               'ndcg': np.zeros(len(board.args.topks)),
               'auc': np.zeros(1)}

    with torch.no_grad():
        all_users = list(test_dict.keys())
        try:
            assert test_batch_size < len(all_users) // 10
        except:
            print('test_batch_size is too large')
        users_list, score_list, true_items_list, pred_item_list = [], [], [], []
        num_batch = len(all_users) // test_batch_size + 1
        for batch_users in utils.minibatch(all_users, batch_size=test_batch_size):
            pos_item_trans, _ = dataset._get_user_posItems(batch_users)
            ground_true = [test_dict[u] for u in batch_users]
            batch_users = torch.Tensor(batch_users).long()
            batch_users = batch_users.to(board.DEVICE)
            scores = model.get_scores(batch_users)

            exclude_index, exclude_item = [], []
            for i, items in enumerate(pos_item_trans):
                exclude_index.extend([i] * len(items))
                exclude_item.extend(items)

            scores[exclude_index, exclude_item] = -(1 << 10)
            _, socres_k_index = torch.topk(scores, largest=True, k=max_k)

            users_list.append(batch_users)
            true_items_list.append(ground_true)
            pred_item_list.append(socres_k_index.cpu())


        assert num_batch == len(users_list)
        tensors = zip(true_items_list, pred_item_list)

        pre_results = []
        for tensor in tensors:
            pre_results.append(batch_infer(tensor))

        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(all_users))
        results['precision'] /= float(len(all_users))
        results['ndcg'] /= float(len(all_users))
        if board.args.tensorboard:
            marker = '__'
            for x in board.args.topks:
                marker += str(x) + '_'
            summarizer.add_scalars(f'Test/Recall{marker}',
                                   {str(board.args.topks[i]): results['recall'][i] for i in
                                    range(len(board.args.topks))},
                                   epoch)
            summarizer.add_scalars(f'Test/Precision{marker}',
                                   {str(board.args.topks[i]): results['precision'][i] for i in
                                    range(len(board.args.topks))},
                                   epoch)
            summarizer.add_scalars(f'Test/NDCG{marker}',
                                   {str(board.args.topks[i]): results['ndcg'][i] for i in range(len(board.args.topks))},
                                   epoch)

        return results
