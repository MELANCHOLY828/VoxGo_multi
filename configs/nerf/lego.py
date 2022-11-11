_base_ = '../default.py'

expname = 'dvgo_lego'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='/data1/liufengyi/all_datasets/mvsnerf/nerf_synthetic/nerf_synthetic/lego/',
    dataset_type='blender',
    white_bkgd=True,
)

