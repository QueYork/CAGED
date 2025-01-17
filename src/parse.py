import argparse

def parse_args():
    parse = argparse.ArgumentParser(description='Lossless')
    # fixed
    parse.add_argument('--dataset', type=str, default='movie', help='accessible datasets from [movie, pinterest, epinions]')
    parse.add_argument('--tensorboard', type=bool, default=True, help='enable tensorboard')
    parse.add_argument('--model', type=str, default='gcn', help='models to be trained from [lossless]')  
    parse.add_argument('--train_file', type=str, default='train.txt')
    parse.add_argument('--test_file', type=str, default='test.txt')
    parse.add_argument('--train_batch', type=int, default=2048, help='batch size in training')
    parse.add_argument('--test_batch', type=int, default=100, help='batch size in testing')
    parse.add_argument('--layers', type=int, default=2, help='the layer number')
    parse.add_argument('--dim', type=int, default=256, help='embedding dimension')
    parse.add_argument('--latent_dim', type=int, default=256, help='dimension of latent variable z')
    parse.add_argument('--norm_a', type=float, default=1., help='normal distribution')                           
    parse.add_argument('--neg_ratio', type=int, default=1, help='the number of negative sampling')
    parse.add_argument('--seed', type=int, default=2021, help='random seed')
    parse.add_argument('--topks', nargs='+', type=int, default=[20, 100], help='top@k test list') 
    
    # adjustable
    parse.add_argument('--enable_vae', type=int, default=1, help='enable vae, set to 0 leads to vanilla lightgcn training')
    parse.add_argument('--epoch', type=int, default=1000, help='epoch of lightgcn training')
    parse.add_argument('--epoch2', type=int, default=1, help='epoch of vae training')
    parse.add_argument('--lr', type=float, default=5e-4, help='learning rate of lightgcn')
    parse.add_argument('--lr2', type=float, default=5e-4, help='learning rate of vae')
    parse.add_argument('--weight', type=float, default=1e-4, help='the weight of l2 norm')     
    parse.add_argument('--hidden_dim', nargs='+', type=int, default=[512, 1024, 512], help='vae neuron number')
    parse.add_argument('--sc_var', type=float, default=1, help='vae score variance')
    parse.add_argument('--eps_decay', type=float, default=5e-4, help='epsilon decay ratio')
    parse.add_argument('--beta', type=float, default=1, help='kl score amplifier')
    
    return parse.parse_args() 