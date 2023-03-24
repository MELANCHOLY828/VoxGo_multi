_base_ = '../default.py'

basedir = '/data/liufengyi/Results/VoxGo_multi/'

data = dict(
    dataset_type='facevideo',
    ndc=True,
    # width=640,
    # height=480,
    multi_scale = [],
    factor = [4.225],
    flag_multi = False,
    # multi_scale = [2000, 8000, 12000],
    # factor = [16.9, 8.45, 4, 2],
    
)

coarse_train = dict(
    N_iters=0,
)

fine_train = dict(
    N_iters=30000,
    N_rand=4096,
    weight_distortion=0.01,
    # pg_scale=[2000, 8000, 12000, 20000],
    pg_scale=[],
    ray_sampler='random',
    tv_before=1e9,
    tv_dense_before=10000,
    weight_tv_density=1e-5,
    weight_tv_k0=1e-6,
    weight_tv_hash=1e-5,
    plane_tv_before=0,
    plane_tv_after=0,
    l1_time_planes = 0.0001,   #权重
    time_smoothness_weight = 0.001,   #权重
)

fine_model_and_render = dict(
    num_voxels=256**3,
    mpi_depth=128,
    rgbnet_dim=9,
    rgbnet_width=64,
    world_bound_scale=1,
    fast_color_thres=1e-3,
    
    hash_channel = 28,
    hidden_features = 128,
    hidden_layers = 1,
    out_features = 28,
    use_fine = True, 
    use_sh = True, 
    hash_type='DenseGrid',        #Add
    
    
    flag_video = True,
    hash_t = False,     #对时间t使用mlp编码还是hash编码
    mlp_t = True,
    plane_t = False,
    flag_xyzt = False,
    add_mlp = True,
    
    t_config = {
        'flag_video': True,
        'hash_t': False,     #对时间t使用mlp编码还是hash编码
        'mlp_t': True,
        'plane_t': False,
        'flag_xyzt': False,
        'add_mlp': True,
        'mlp_pe': True,
    },
    grid_config = {
    'grid_dimensions': 2,
    'input_coordinate_dim': 4,
    'output_coordinate_dim': 16,
    'resolution': [64, 64, 64, 150],
    # 'multiscale_res': [1, 2, 4, 8],
    'multiscale_res': [8],
    },
    
)

