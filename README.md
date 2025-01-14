## prepare normal
python normal.py -s <dataset_path>  -m <output_path> 

## prepare depth
python depth_infer.py -s <dataset_path> -m <output_path>

## scale the depth
python utils/make_depth_scale.py --base_dir <dataset_path> --depth_dir <depth_dir>

## start train  
## if gpu can't load all pictures
python partition_train.py -s <dataset_path>  -m <output_path>  --depths ture  --iterations 50000

## if gpu can load all pictures
python train_all_load.py -s <dataset_path>  -m <output_path>   --depths ture  --iterations 50000
## dataset structure
dataset_path
├── images
├── renders_depth
├── renders_normal
└── sparse   
    └── 0
        ├── cameras.bin
        ├── depth_params.json
        ├── images.bin
        ├── points3D.bin
        └── points3D.ply    