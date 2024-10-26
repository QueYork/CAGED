ps -ef | grep tensorboard | awk '{print $2}' | xargs kill -9 
tensorboard --port 6007 --logdir /root/fyp/src/results/movie

nohup python -u train.py --dataset movie --enable_vae 0 --lr 5e-4 --weight 1e-3 --epoch 800 >/dev/null 2>&1 &
nohup python -u train.py --dataset pinterest --enable_vae 0 --lr 5e-4 --weight 1e-4 --epoch 300 >/dev/null 2>&1 &




nohup python -u train.py --dataset movie --lr 5e-4 --weight 1e-3 --hidden_dim 512 1024 512 --sc_var 0.04 --thold 0 --epoch 1000 --lambda_ 3.5 > /dev/null 2>&1 &
nohup python -u train.py --dataset pinterest --lr 5e-4 --weight 1e-3 --eps_decay 0.1875 --sc_var 0.04 --epoch 400 >/dev/null 2>&1 &