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

def main():
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
        
    # Initialize lightgcn
    gcn = MODEL[board.args.model](dataset=dataset).to(board.DEVICE)
    loss_gcn = LOSS_F[board.args.model](gcn)
    
    # Initialize vae 
    vae = MODEL['vae']()
    vae = vae.to(board.DEVICE)
    loss_vae = LOSS_F['vae'](vae)
    
    max_recall20 = 0
    max_info = None
    
    eps_decay = board.args.eps_decay
    weight_graph = dataset.get_root_degree() * dataset.Graph
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
            max_info = results
            logging.info(f'Summary at recall = {max_recall20}')
            
            """
            ********************* CVAE Train (only enable when making progress) **************************
            """
            if board.args.enable_vae:
                gcn.eval()
                
                with torch.no_grad():
                    user_emb, item_emb = gcn.aggregate_embed()
                    user_emb = gcn.pooling(user_emb)
                    item_emb = gcn.pooling(item_emb)
                
                for epoch2 in range(board.args.epoch2):
                    info = evals.Train_vae(data_loader=data_loader, vae=vae, user_emb=user_emb, item_emb=item_emb, 
                                            loss_f=loss_vae, batch_size=board.args.train_batch)
                    logging.info(f'CVAE EPOCH[{epoch2 + 1}/{board.args.epoch2}] {info} ')
                
                with torch.no_grad():
                    new_graph = (1 - eps_decay) * gcn.Graph + eps_decay * weight_graph * evals.Weight_Inference(vae, user_emb, item_emb, data_loader)
                    del gcn.Graph 
                    torch.cuda.empty_cache()  
                    gcn.Graph = new_graph.coalesce()
    
    logging.info('Final result (highest recall@20): ')   
    logging.info(max_info)
      
      
if __name__ == '__main__':
    main()
    