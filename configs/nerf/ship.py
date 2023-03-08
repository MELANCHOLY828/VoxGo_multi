_base_ = '../default.py'

expname = 'ship_out_4'
basedir = '/data/liufengyi/Results/VoxGo_rewrite'

data = dict(
    datadir='/data/liufengyi/Datasets/nerf_synthetic/ship',
    dataset_type='blender',
    white_bkgd=True,
    half_res=True, 
)

