# Train
```
# Train transformer
# T:negtive loss weight
CUDA_VISIBLE_DEVICES=0 python train.py --model transformer --epoch 9 --drop 8 --T 1.1 --thw 1.5 --tbName ${any tensorboard name}

# Train RNN 
CUDA_VISIBLE_DEVICES=1 python train.py --model rnn --T 2  --thw 1 --tbName ${any tensorboard name}
```
