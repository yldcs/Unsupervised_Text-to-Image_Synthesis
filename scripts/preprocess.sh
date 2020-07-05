# devices
gpu_ids="0,1"

# factory config
option_name='base'
data_dir='./data'
outputs_dir='./data/coco'

python preprocessing/process_data.py --option_name ${option_name} \
  --data_dir ${data_dir} \
  --outputs_dir ${outputs_dir}
