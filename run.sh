# CUDA_LAUNCH_BLOCKING=1 python Main.py -config 'configs/p3.yaml' -dataset 'data/syn_large_v/'
# CUDA_LAUNCH_BLOCKING=1 python Main.py -config 'configs/p2.yaml' -dataset 'data/syn_large_v/'
# CUDA_LAUNCH_BLOCKING=1 python Main.py -config 'configs/base.yaml' -dataset 'data/syn_large_v/'

CUDA_LAUNCH_BLOCKING=1 python Main.py -config 'configs/p2.yaml' -dataset 'data/syn_periodic/'
# CUDA_LAUNCH_BLOCKING=1 python Main.py -config 'configs/base.yaml' -dataset 'data/syn_periodic/'

# CUDA_LAUNCH_BLOCKING=1 python Main.py -config 'configs/p3.yaml' -dataset 'data/cc100/'
# CUDA_LAUNCH_BLOCKING=1 python Main.py -config 'configs/base.yaml' -dataset 'data/cc100/'