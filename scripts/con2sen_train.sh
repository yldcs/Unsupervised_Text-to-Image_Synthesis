# devices
gpu_ids="7"

# factory config
option_name='con2sen_train'
dataset_name='con2sen_train'
model_name='con2sen_train'
ckpt_name='con2sen_new'
load_path='./data/outputs/checkpoints/con2sen_new/net_Dec_40000.pth'

python train.py --option_name ${option_name} \
  --dataset_name ${dataset_name} \
  --model_name ${model_name} \
  --ckpt_name ${ckpt_name} \
  --gpu_ids ${gpu_ids}
