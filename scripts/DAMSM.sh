gpu_ids='7'

option_name='DAMSM'
model_name='DAMSM_train'
dataset_name='DAMSM'
batch_size=140

python train.py --option_name ${option_name} \
  --model_name ${model_name} \
  --dataset_name ${dataset_name} \
  --batch_size ${batch_size} \
  --gpu_ids ${gpu_ids}
      


