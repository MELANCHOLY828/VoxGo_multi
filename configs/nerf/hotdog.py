_base_ = '../default.py'

expname = 'hotdog'
basedir = '/data/liufengyi/Results/VoxGo_rewrite'

data = dict(
    datadir='/data/liufengyi/Datasets/nerf_synthetic/nerf_synthetic/hotdog/',
    dataset_type='blender',
    white_bkgd=True,
    half_res=True, 
)

