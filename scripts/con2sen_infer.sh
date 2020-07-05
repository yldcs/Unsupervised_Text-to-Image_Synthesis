gpu_ids='6'

option_name='con2sen_infer'
dataset_name='con2sen_infer'
model_name='con2sen_infer'

batch_size=800
load_path='./outputs/checkpoints/con2sen_new/net_Enc_it_60000.pth'

python pseudo_infer.py --option_name ${option_name} \
  --model_name ${model_name} \
  --dataset_name ${dataset_name} \
  --load_path ${load_path} \
  --batch_size ${batch_size} \
  --gpu_ids ${gpu_ids}
