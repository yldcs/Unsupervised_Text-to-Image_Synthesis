gpu_ids='7'

option_name='GSC'
model_name='GSC_train'
dataset_name='GSC'
NET_E='./data/pretrainDAMSM/net_text_encoder_epoch_100.pth'
NET_G='None'
interval_epoch=10
interval_display=300
ckpt_name='GSC'
batch_size=35
num_workers_train=1

python train.py --option_name ${option_name} \
  --model_name ${model_name} \
  --dataset_name ${dataset_name} \
  --ckpt_name ${ckpt_name} \
  --NET_E ${NET_E} \
  --NET_G ${NET_G} \
  --interval_epoch ${interval_epoch} \
  --interval_display ${interval_display} \
  --batch_size ${batch_size} \
  --gpu_ids ${gpu_ids} \
  --num_workers_train ${num_workers_train}
