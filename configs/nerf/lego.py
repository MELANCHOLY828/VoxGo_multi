_base_ = '../default.py'

expname = 'dvgo_lego'
basedir = '/data1/liufengyi/get_results/VoxGo_rewrite/logs'

data = dict(
    datadir='/data1/liufengyi/all_datasets/mvsnerf/nerf_synthetic/nerf_synthetic/lego/',
    dataset_type='blender',
    white_bkgd=True,
)

