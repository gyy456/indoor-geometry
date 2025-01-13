## prepare normal
python normal.py -s <dataset_path>  -m <output_path> 

## prepare depth
python depth_infer.py -s <dataset_path> -m <output_path>

## scale the depth
python make_depth_scale.py --base_dir <dataset_path> --depth_dir <depth_dir>

## start train
python partition_train.py -s <dataset_path>  -m <output_path>    --depths ture  
