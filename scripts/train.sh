#! /usr/bin/env bash



python train.py  --dataset VOC \
                  --batch_size 32 \
                    --end_epoch 50 \
                    --lr 5e-3 \
                    --size 224 \
                    --max_grad_norm 20.0 \
                    --wandb_name damamnet