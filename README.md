ps -ef | grep tensorboard | awk '{print $2}' | xargs kill -9 
tensorboard --port 6007 --logdir /root/fyp/src/results/

nohup python -u train.py --dataset movie --lr 5e-4 --weight 1e-4 --epoch 150 >/dev/null 2>&1 &