import os
import sys
from os.path import join
import warnings
warnings.filterwarnings('ignore')

PATH = os.path.dirname(os.path.abspath(__file__))
ROOT = join(PATH, '../')
sys.path.append(ROOT)

from torch.utils.tensorboard import SummaryWriter
import src.data_loader as Data_Loader
import datetime
import pytz
import torch
import numpy as np
import logging
import src.powerboard as board
import src.utils as utils
import src.evals as evals
import src.model as model

MODEL = {
    'gcn': model.LightGCN,
    'vae': model.ConditionalVAE
}
LOSS_F = {
    'gcn': evals.GCNLoss,
    'vae': evals.VAELoss
}

def vanilla_gcn_train():
    utils.set_seed(board.SEED)
    print('--SEED--:', board.SEED)

    dataset = Data_Loader.LoadData(data_name=board.args.dataset)
    model = MODEL[board.args.model](dataset=dataset)
    model = model.to(board.DEVICE)
    loss_f = LOSS_F[board.args.model](model)

    # log file path
    path = join(board.BOARD_PATH, board.args.dataset)
    timezone = pytz.timezone('Asia/Shanghai')
    nowtime = datetime.datetime.now(tz=timezone)
    log_path = join(path, nowtime.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + board.args.model)

    # init tensorboard
    if board.args.tensorboard:
        summarizer: SummaryWriter = SummaryWriter(log_path)
    else:
        summarizer = None
        board.cprint('tensorboard disabled.')

    try:
        max_recall20 = 0

        # logger initializer
        log_name = utils.create_log_name(log_path)
        utils.log_config(path=log_path, name=log_name, level=logging.DEBUG, console_level=logging.DEBUG, console=True)
        logging.info(board.args)

        for epoch in range(board.args.epoch):
            info = evals.Train_full(dataset=dataset, model=model, epoch=epoch, loss_f=loss_f,
                                    neg_ratio=board.args.neg_ratio, summarizer=summarizer)

            board.cprint(f'[testing at epoch-{epoch}]')
            results = evals.Inference(dataset=dataset, model=model, epoch=epoch, summarizer=summarizer)

            logging.info(f'[testing at epoch-{epoch}]')
            logging.info(results)

            logging.info(f'EPOCH[{epoch + 1}/{board.args.epoch}] {info} ')
            
            if max_recall20 < results['recall'][0]:
                max_recall20 = results['recall'][0]
                logging.info(f'Summary at recall = {max_recall20}')
               
                file = f"{board.args.model}-{board.args.dataset}-{board.args.dim}.pth.tar"
                weight_file = os.path.join(board.FILE_PATH, file)

                torch.save({'user_embed': model.user_embed, 'item_embed': model.item_embed}, weight_file)

    except:
        raise NotImplementedError('Error in running main file')

    finally:
        if board.args.tensorboard:
            summarizer.close()

def pretrain_vae_main():
    utils.set_seed(board.SEED)
    print('--SEED--:', board.SEED)

    # Load dataset
    dataset = Data_Loader.LoadData(data_name=board.args.dataset)
    u_neighbors = dataset.get_neighbors()
    dataset_vae = torch.utils.data.TensorDataset(u_neighbors[:, 0], u_neighbors[:, 1])
    data_loader = torch.utils.data.DataLoader(dataset_vae, batch_size=board.args.train_batch, 
                                              shuffle=True, generator=torch.Generator().manual_seed(board.SEED))

    # log file path
    path = join(board.BOARD_PATH, board.args.dataset)
    timezone = pytz.timezone('Asia/Shanghai')
    nowtime = datetime.datetime.now(tz=timezone)
    log_path = join(path, nowtime.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + board.args.model)

    # logger initializer
    log_name = utils.create_log_name(log_path)
    utils.log_config(path=log_path, name=log_name, level=logging.DEBUG, console_level=logging.DEBUG, console=True)
    logging.info(board.args)
    
    # init tensorboard
    if board.args.tensorboard:
        summarizer: SummaryWriter = SummaryWriter(log_path)
    else:
        summarizer = None
        board.cprint('tensorboard disabled.')
        
    """
    ********************* CVAE Pretrain ************************************
    """
    # Initialize 2 models
    gcn = MODEL[board.args.model](dataset=dataset)
    gcn = gcn.to(board.DEVICE)
    gcn.eval()
    
    vae = MODEL['vae']()
    vae = vae.to(board.DEVICE)
    loss_f = LOSS_F['vae'](vae)
    
    # Load best GCN embeddings
    file = f"{board.args.model}-{board.args.dataset}-{board.args.dim}.pth.tar"
    weight_file = os.path.join(board.FILE_PATH, file)
    ckpt = torch.load(weight_file, map_location=board.DEVICE)
    gcn.user_embed = ckpt['user_embed']
    gcn.item_embed = ckpt['item_embed']
    
    with torch.no_grad():
        user_emb, item_emb = gcn.aggregate_embed()
        user_emb = gcn.pooling(user_emb)
        item_emb = gcn.pooling(item_emb)
    
    # Train
    for epoch in range(board.args.epoch2):
        info = evals.Train_vae(data_loader=data_loader, vae=vae, user_emb=user_emb, item_emb=item_emb, 
                                loss_f=loss_f, batch_size=board.args.train_batch, epoch=epoch)
        logging.info(f'CVAE EPOCH[{epoch + 1}/{board.args.epoch2}] {info} ')
    
    # Generate adj matrix
    dataset.Graph = evals.Weight_Inference(vae, user_emb, item_emb, data_loader)
    
    """
    ********************* GCN Train ************************************
    """
    model = MODEL[board.args.model](dataset=dataset)
    model = model.to(board.DEVICE)
    loss_f = LOSS_F[board.args.model](model)
    
    try:
        max_recall20 = 0
        
        for epoch in range(board.args.epoch):
            info = evals.Train_full(dataset=dataset, model=model, epoch=epoch, loss_f=loss_f,
                                    neg_ratio=board.args.neg_ratio, summarizer=summarizer)

            board.cprint(f'[testing at epoch-{epoch}]')
            results = evals.Inference(dataset=dataset, model=model, epoch=epoch, summarizer=summarizer)

            logging.info(f'[testing at epoch-{epoch}]')
            logging.info(results)

            logging.info(f'EPOCH[{epoch + 1}/{board.args.epoch}] {info} ')
            
            if max_recall20 < results['recall'][0]:
                max_recall20 = results['recall'][0]
                logging.info(f'Summary at recall = {max_recall20}')

    except:
        raise NotImplementedError('Error in running main file')

    finally:
        if board.args.tensorboard:
            summarizer.close()

def vae_gcn_main():
    utils.set_seed(board.SEED)
    print('--SEED--:', board.SEED)

    # Load dataset
    dataset = Data_Loader.LoadData(data_name=board.args.dataset)
    u_neighbors = dataset.get_neighbors()
    dataset_vae = torch.utils.data.TensorDataset(u_neighbors[:, 0], u_neighbors[:, 1])
    data_loader = torch.utils.data.DataLoader(dataset_vae, batch_size=board.args.train_batch, 
                                              shuffle=True, generator=torch.Generator().manual_seed(board.SEED))

    # log file path
    path = join(board.BOARD_PATH, board.args.dataset)
    timezone = pytz.timezone('Asia/Shanghai')
    nowtime = datetime.datetime.now(tz=timezone)
    log_path = join(path, nowtime.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + board.args.model)

    # logger initializer
    log_name = utils.create_log_name(log_path)
    utils.log_config(path=log_path, name=log_name, level=logging.DEBUG, console_level=logging.DEBUG, console=True)
    logging.info(board.args)
    
    # init tensorboard
    if board.args.tensorboard:
        summarizer: SummaryWriter = SummaryWriter(log_path)
    else:
        summarizer = None
        board.cprint('tensorboard disabled.')
        
    # Initialize 2 models
    gcn = MODEL[board.args.model](dataset=dataset)
    gcn = gcn.to(board.DEVICE)
    loss_gcn = LOSS_F[board.args.model](gcn)
    
    vae = MODEL['vae']()
    vae = vae.to(board.DEVICE)
    loss_vae = LOSS_F['vae'](vae)
    
    max_recall20 = 0
    epsilon = 1.0
    eps_decay = 1.0 / (board.args.epoch * board.args.eps_decay)
    for epoch in range(board.args.epoch):
        """
        ********************* GCN Train **************************
        """
        info = evals.Train_full(dataset=dataset, model=gcn, epoch=epoch, loss_f=loss_gcn, 
                                neg_ratio=board.args.neg_ratio, summarizer=summarizer)

        board.cprint(f'[testing at epoch-{epoch}]')
        results = evals.Inference(dataset=dataset, model=gcn, epoch=epoch, summarizer=summarizer)

        logging.info(f'[testing at epoch-{epoch}]')
        logging.info(results)

        logging.info(f'EPOCH[{epoch + 1}/{board.args.epoch}] {info} ')
        
        if max_recall20 < results['recall'][0]:
            max_recall20 = results['recall'][0]
            logging.info(f'Summary at recall = {max_recall20}')
        
        """
        ********************* CVAE Train **************************
        """
        gcn.eval()
        
        with torch.no_grad():
            user_emb, item_emb = gcn.aggregate_embed()
            user_emb = gcn.pooling(user_emb)
            item_emb = gcn.pooling(item_emb)
        
        for epoch2 in range(board.args.epoch2):
            info = evals.Train_vae(data_loader=data_loader, vae=vae, user_emb=user_emb, item_emb=item_emb, 
                                    loss_f=loss_vae, batch_size=board.args.train_batch, epoch=epoch2)
            logging.info(f'CVAE EPOCH[{epoch2 + 1}/{board.args.epoch2}] {info} ')
        
        # Update adj matrix for GCN aggregation
        if epsilon > 0:
            epsilon -= (epsilon * eps_decay)
        elif epsilon < 0:
            epsilon = 0
        with torch.no_grad():
            new_graph = epsilon * gcn.Graph + (1 - epsilon) * evals.Weight_Inference(vae, user_emb, item_emb, data_loader)
            del gcn.Graph 
            torch.cuda.empty_cache()  
            gcn.Graph = new_graph.coalesce()
      
if __name__ == '__main__':
    # vanilla_gcn_train()
    # pretrain_vae_main()
    vae_gcn_main()