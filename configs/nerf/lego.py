_base_ = '../default.py'

# expname = 'lego_1*128_160_fast'
expname = 'lego'
basedir = '/data/liufengyi/Results/VoxGo_rewrite'

data = dict(
    datadir='/data/liufengyi/Datasets/nerf_synthetic/nerf_synthetic/lego/',
    dataset_type='blender',
    white_bkgd=True,
    half_res=True, 
)

