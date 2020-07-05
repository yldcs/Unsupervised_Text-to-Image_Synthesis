# devices
gpu_ids="0,1"

# factory config
option_name='base'
dataset_name='base'
model_name='base'
network_name='base'

python train.py --option_name ${option_name} \
  --dataset_name ${dataset_name} \
  --model_name ${model_name} \
  --network_name ${network_name} \
  --gpu_ids ${gpu_ids}
