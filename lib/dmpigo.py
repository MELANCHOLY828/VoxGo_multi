import os
import time
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
from torch_scatter import scatter_add, segment_coo
from . import sh

from . import grid
from .dvgo import Raw2Alpha, Alphas2Weights, render_utils_cuda

class LatentCode(nn.Module):
    def __init__(self,
                D = 2, W = 128, in_channels_t = 1, out_channels_t = 64):
        super(LatentCode, self).__init__()
        self.D = D
        self.W = W
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_t, W) 
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"t_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # output layers
        self.LatentCode = nn.Linear(W, out_channels_t) 

    def forward(self, t):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        input_t = t
        for i in range(self.D):
            input_t = getattr(self, f"t_encoding_{i+1}")(input_t.float())

        LatentCode = self.LatentCode(input_t)  #得到的体素

        return LatentCode

'''Model'''
#
class DirectMPIGO(torch.nn.Module):    
    def __init__(self, xyz_min, xyz_max,
                 num_voxels=0, mpi_depth=0,
                 mask_cache_path=None, mask_cache_thres=1e-3, mask_cache_world_size=None,
                 fast_color_thres=0,
                 density_type='DenseGrid', k0_type='DenseGrid',
                 hash_type='DenseGrid',   #自己加的 hashtable
                 hash_channel = 28,
                 hash_config={},
                 density_config={}, k0_config={},
                 rgbnet_dim=0,
                 rgbnet_depth=3, rgbnet_width=128,
                 hidden_features=128, hidden_layers = 2, out_features=28,
                 use_fine = True,
                 use_sh = True,
                 flag_video = True,
                 viewbase_pe=4,
                 savepath = '/data/liufengyi/Results/VoxGo_rewrite',
                 **kwargs):
        super(DirectMPIGO, self).__init__()
        #该组参数在模型训练时不会更新（即调用optimizer.step()后该组参数不会变化，只可人为地改变它们的值），但是该组参数又作为模型参数不可或缺的一部分
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))    
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.fast_color_thres = fast_color_thres   #0.001
        self.savepath = savepath
        self.use_fine = use_fine
        self.use_sh = use_sh
        self.flag_video = flag_video
        self.ratio1 = []
        self.ratio2 = []
        self.ratio3 = []
        # determine init grid resolution    网格体素数量 空间大小  得到网格分辨率
        self._set_grid_resolution(num_voxels, mpi_depth) 
        if self.flag_video:   #视频合成，有时间t  先对时间t进行latent code
            out_channels_t = 64
            self.LatentCode = LatentCode(out_channels_t = out_channels_t)
            hash_channel_ = hash_channel + out_channels_t
            print("t-encoding structure: ", self.LatentCode)
        else:
            hash_channel_ = hash_channel
        if self.use_fine:
            # init hash voxel grid
            self.hash_channel = hash_channel
            self.hash_type = hash_type
            self.hash_config = hash_config
            self.hidden_features = hidden_features
            self.hidden_layers = hidden_layers
            self.out_features = out_features
            
            # init density bias so that the initial contribution (the alpha values)
            # of each query points on a ray is equal   
            # 初始密度偏差，以便射线上每个查询点的初始贡献（alpha 值）相等
            self.act_shift = grid.DenseGrid(
                    channels=1, world_size=[1,1,mpi_depth],
                    xyz_min=xyz_min, xyz_max=xyz_max)
            self.act_shift.grid.requires_grad = False
            with torch.no_grad():
                g = np.full([mpi_depth], 1./mpi_depth - 1e-6)
                p = [1-g[0]]
                for i in range(1, len(g)):
                    p.append((1-g[:i+1].sum())/(1-g[:i].sum()))
                for i in range(len(p)):
                    self.act_shift.grid[..., i].fill_(np.log(p[i] ** (-1/self.voxel_size_ratio) - 1))
            
            
            self.hash = grid.create_grid(
                    hash_type, channels=self.hash_channel, world_size=self.world_size,
                    xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                    use_fine = self.use_fine,
                    hidden_features = self.hidden_features, 
                    hidden_layers = self.hidden_layers,
                    out_features = self.out_features,
                    config=self.hash_config)
            
            self.mlpnet = nn.Sequential(
                    nn.Linear(hash_channel_, hidden_features), nn.ReLU(inplace=True),
                    *[
                        nn.Sequential(nn.Linear(hidden_features, hidden_features), nn.ReLU(inplace=True))
                        for _ in range(hidden_layers)
                    ],
                    nn.Linear(hidden_features, out_features),
                )
            self.rgbnet_kwargs = {
                'rgbnet_dim': 0,   #9
                'rgbnet_depth': 0, 'rgbnet_width': 0,   #3 ， 64
                'viewbase_pe': 0,   #0
                }
            if not self.use_sh:
                rgbnet_dim = out_features - 1
                dim0 = (3+3*viewbase_pe*2) + rgbnet_dim   #3+9
                self.rgbnet_kwargs = {
                'rgbnet_dim': rgbnet_dim,   #9
                'rgbnet_depth': rgbnet_depth, 'rgbnet_width': rgbnet_width,   #3 ， 64
                'viewbase_pe': viewbase_pe,   #0
                }
                self.register_buffer('viewfreq', torch.FloatTensor([(2**i) for i in range(viewbase_pe)]))
                self.rgbnet = nn.Sequential(
                    nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True),
                    *[
                        nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                        for _ in range(rgbnet_depth-2)
                    ],
                    nn.Linear(rgbnet_width, 3),
                )  
                print('rgbnet structure', self.rgbnet)                
            load_mlp = False
            if load_mlp:
                print("loading mlp weights......")
                path = "/data/liufengyi/Results/mlp.tar"
                ckpt = torch.load(path)
                self.mlpnet.load_state_dict(ckpt['model_state_dict'])
            # nn.init.constant_(self.mlpnet[-1].bias, 0)
            print('dvgo: hash voxel grid', self.hash)
            print('dvgo: MLPNet structure', self.mlpnet)
            
        else:    
            # init density voxel grid
            self.density_type = density_type
            self.density_config = density_config
            self.density = grid.create_grid(
                    density_type, channels=1, world_size=self.world_size,   #[ 93,  88, 128]
                    xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                    config=self.density_config)     #DenseGrid(channels=1, world_size=[372, 352, 128])

            # init density bias so that the initial contribution (the alpha values)
            # of each query points on a ray is equal   
            # 初始密度偏差，以便射线上每个查询点的初始贡献（alpha 值）相等
            self.act_shift = grid.DenseGrid(
                    channels=1, world_size=[1,1,mpi_depth],
                    xyz_min=xyz_min, xyz_max=xyz_max)
            self.act_shift.grid.requires_grad = False
            with torch.no_grad():
                g = np.full([mpi_depth], 1./mpi_depth - 1e-6)
                p = [1-g[0]]
                for i in range(1, len(g)):
                    p.append((1-g[:i+1].sum())/(1-g[:i].sum()))
                for i in range(len(p)):
                    self.act_shift.grid[..., i].fill_(np.log(p[i] ** (-1/self.voxel_size_ratio) - 1))

            # init color representation
            # feature voxel grid + shallow MLP  (fine stage)
            self.rgbnet_kwargs = {
                'rgbnet_dim': rgbnet_dim,   #9
                'rgbnet_depth': rgbnet_depth, 'rgbnet_width': rgbnet_width,   #3 ， 64
                'viewbase_pe': viewbase_pe,   #0
            }
            self.k0_type = k0_type   #DenseGrid
            self.k0_config = k0_config
            if rgbnet_dim <= 0:   #no
                # color voxel grid  (coarse stage)
                self.k0_dim = 3
                self.k0 = grid.create_grid(
                    k0_type, channels=self.k0_dim, world_size=self.world_size,
                    xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                    config=self.k0_config)
                self.rgbnet = None
            else:   #yes
                self.k0_dim = rgbnet_dim
                self.k0 = grid.create_grid(
                        k0_type, channels=self.k0_dim, world_size=self.world_size,
                        xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                        config=self.k0_config)    #DenseGrid(channels=9, world_size=[93, 88, 128])
                self.register_buffer('viewfreq', torch.FloatTensor([(2**i) for i in range(viewbase_pe)]))
                dim0 = (3+3*viewbase_pe*2) + self.k0_dim   #3+9
                self.rgbnet = nn.Sequential(
                    nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True),
                    *[
                        nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                        for _ in range(rgbnet_depth-2)
                    ],
                    nn.Linear(rgbnet_width, 3),
                )       #3层MLP网络 12-64  64-64   64-3
                nn.init.constant_(self.rgbnet[-1].bias, 0)   #bias初始化为0

            print('dmpigo: densitye grid', self.density)
            print('dmpigo: feature grid', self.k0)
            print('dmpigo: mlp', self.rgbnet)

        # Using the coarse geometry if provided (used to determine known free space and unknown space)
        # Re-implement as occupancy grid (2021/1/31)
        self.mask_cache_path = mask_cache_path
        self.mask_cache_thres = mask_cache_thres
        if mask_cache_world_size is None:   #yes
            mask_cache_world_size = self.world_size   #[93, 88, 128]
        if mask_cache_path is not None and mask_cache_path:   #no
            mask_cache = grid.MaskGrid(
                    path=mask_cache_path,
                    mask_cache_thres=mask_cache_thres).to(self.xyz_min.device)
            self_grid_xyz = torch.stack(torch.meshgrid(
                torch.linspace(self.xyz_min[0], self.xyz_max[0], mask_cache_world_size[0]),
                torch.linspace(self.xyz_min[1], self.xyz_max[1], mask_cache_world_size[1]),
                torch.linspace(self.xyz_min[2], self.xyz_max[2], mask_cache_world_size[2]),
            ), -1)
            mask = mask_cache(self_grid_xyz)
        else:   #yes
            mask = torch.ones(list(mask_cache_world_size), dtype=torch.bool)   #[93, 88, 128]
        self.mask_cache = grid.MaskGrid(
                path=None, mask=mask,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max)

    def _set_grid_resolution(self, num_voxels, mpi_depth):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.mpi_depth = mpi_depth
        r = (num_voxels / self.mpi_depth / (self.xyz_max - self.xyz_min)[:2].prod()).sqrt()
        self.world_size = torch.zeros(3, dtype=torch.long)
        self.world_size[:2] = (self.xyz_max - self.xyz_min)[:2] * r
        self.world_size[2] = self.mpi_depth
        self.voxel_size_ratio = 256. / mpi_depth
        print('dmpigo: world_size      ', self.world_size)
        print('dmpigo: voxel_size_ratio', self.voxel_size_ratio)

    def get_kwargs(self):
        if self.use_fine:
            return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'num_voxels': self.num_voxels,
            'mpi_depth': self.mpi_depth,
            'voxel_size_ratio': self.voxel_size_ratio,
            'mask_cache_path': self.mask_cache_path,
            'mask_cache_thres': self.mask_cache_thres,
            'mask_cache_world_size': list(self.mask_cache.mask.shape),
            'fast_color_thres': self.fast_color_thres,
            # 'density_type': self.density_type,
            # 'k0_type': self.k0_type,
            'hash_type': self.hash_type,
            # 'density_config': self.density_config,
            # 'k0_config': self.k0_config,
            'hash_config' : self.hash_config,
            'use_fine' : self.use_fine,
            'hash_channel' : self.hash_channel,
            'hidden_features' : self.hidden_features,
            'hidden_layers' : self.hidden_layers,
            'out_features' : self.out_features,
            'use_sh' : self.use_sh,
            'flag_video': self.flag_video,
            **self.rgbnet_kwargs,
        }
        else:
            return {
                'xyz_min': self.xyz_min.cpu().numpy(),
                'xyz_max': self.xyz_max.cpu().numpy(),
                'num_voxels': self.num_voxels,
                'mpi_depth': self.mpi_depth,
                'voxel_size_ratio': self.voxel_size_ratio,
                'mask_cache_path': self.mask_cache_path,
                'mask_cache_thres': self.mask_cache_thres,
                'mask_cache_world_size': list(self.mask_cache.mask.shape),
                'fast_color_thres': self.fast_color_thres,
                'density_type': self.density_type,
                'k0_type': self.k0_type,
                'density_config': self.density_config,
                'k0_config': self.k0_config,
                'use_fine' : self.use_fine,
                **self.rgbnet_kwargs,
            }

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels, mpi_depth):
        print('dmpigo: scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels, mpi_depth)
        print('dmpigo: scale_volume_grid scale world_size from', ori_world_size.tolist(), 'to', self.world_size.tolist())
        if self.use_fine:
            self.hash.scale_volume_grid(self.world_size)
        else:
            self.density.scale_volume_grid(self.world_size)
            self.k0.scale_volume_grid(self.world_size)

        if np.prod(self.world_size.tolist()) <= 256**3:
            self_grid_xyz = torch.stack(torch.meshgrid(
                torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]),
                torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]),
                torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2]),
            ), -1)
            if self.use_fine:
                dens = self.mlpnet(self.hash.get_dense_grid().squeeze().permute(1,2,3,0).reshape(-1,self.hash_channel))[:,0].reshape(*self.world_size,-1).permute(3,0,1,2).unsqueeze(0) + self.act_shift.grid
                self_alpha = F.max_pool3d(self.activate_density(dens), kernel_size=3, padding=1, stride=1)[0,0]
                self.mask_cache = grid.MaskGrid(
                        path=None, mask=self.mask_cache(self_grid_xyz) & (self_alpha>self.fast_color_thres),
                        xyz_min=self.xyz_min, xyz_max=self.xyz_max)
            else:
                dens = self.density.get_dense_grid() + self.act_shift.grid
                self_alpha = F.max_pool3d(self.activate_density(dens), kernel_size=3, padding=1, stride=1)[0,0]
                self.mask_cache = grid.MaskGrid(
                        path=None, mask=self.mask_cache(self_grid_xyz) & (self_alpha>self.fast_color_thres),
                        xyz_min=self.xyz_min, xyz_max=self.xyz_max)

        print('dmpigo: scale_volume_grid finish')

    @torch.no_grad()
    def update_occupancy_cache(self):
        ori_p = self.mask_cache.mask.float().mean().item()
        cache_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.mask_cache.mask.shape[0]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.mask_cache.mask.shape[1]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.mask_cache.mask.shape[2]),
        ), -1)
        cache_grid_density = self.density(cache_grid_xyz)[None,None]
        cache_grid_alpha = self.activate_density(cache_grid_density)
        cache_grid_alpha = F.max_pool3d(cache_grid_alpha, kernel_size=3, padding=1, stride=1)[0,0]
        self.mask_cache.mask &= (cache_grid_alpha > self.fast_color_thres)
        new_p = self.mask_cache.mask.float().mean().item()
        print(f'dmpigo: update mask_cache {ori_p:.4f} => {new_p:.4f}')

    def update_occupancy_cache_lt_nviews(self, rays_o_tr, rays_d_tr, imsz, render_kwargs, maskout_lt_nviews):
        print('dmpigo: update mask_cache lt_nviews start')
        eps_time = time.time()
        count = torch.zeros_like(self.density.get_dense_grid()).long()
        device = count.device
        for rays_o_, rays_d_ in zip(rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            ones = grid.DenseGrid(1, self.world_size, self.xyz_min, self.xyz_max)
            for rays_o, rays_d in zip(rays_o_.split(8192), rays_d_.split(8192)):
                ray_pts, ray_id, step_id, N_samples = self.sample_ray(
                        rays_o=rays_o.to(device), rays_d=rays_d.to(device), **render_kwargs)
                ones(ray_pts).sum().backward()
            count.data += (ones.grid.grad > 1)
        ori_p = self.mask_cache.mask.float().mean().item()
        self.mask_cache.mask &= (count >= maskout_lt_nviews)[0,0]
        new_p = self.mask_cache.mask.float().mean().item()
        print(f'dmpigo: update mask_cache {ori_p:.4f} => {new_p:.4f}')
        torch.cuda.empty_cache()
        eps_time = time.time() - eps_time
        print(f'dmpigo: update mask_cache lt_nviews finish (eps time:', eps_time, 'sec)')

    def density_total_variation_add_grad(self, weight, dense_mode):
        wxy = weight * self.world_size[:2].max() / 128
        wz = weight * self.mpi_depth / 128
        self.density.total_variation_add_grad(wxy, wxy, wz, dense_mode)

    def k0_total_variation_add_grad(self, weight, dense_mode):
        wxy = weight * self.world_size[:2].max() / 128
        wz = weight * self.mpi_depth / 128
        self.k0.total_variation_add_grad(wxy, wxy, wz, dense_mode)
        
    def hash_total_variation_add_grad(self, weight, dense_mode):
        wxy = weight * self.world_size[:2].max() / 128
        wz = weight * self.mpi_depth / 128
        self.hash.total_variation_add_grad(wxy, wxy, wz, dense_mode)        
        
    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        shape = density.shape
        return Raw2Alpha.apply(density.flatten(), 0, interval).reshape(shape)

    def sample_ray(self, rays_o, rays_d, near, far, stepsize, latent_t=None, **render_kwargs):
        '''Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            near, far:        the near and far distance of the rays.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        '''
        assert near==0 and far==1
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        N_samples = int((self.mpi_depth-1)/stepsize) + 1
        ray_pts, mask_outbbox = render_utils_cuda.sample_ndc_pts_on_rays(
            rays_o, rays_d, self.xyz_min, self.xyz_max, N_samples)
        mask_inbbox = ~mask_outbbox
        if self.flag_video:
            latent_t = latent_t.unsqueeze(1).expand(latent_t.shape[0],ray_pts.shape[1],latent_t.shape[1])
            latent_t = latent_t[mask_inbbox]
        ray_pts = ray_pts[mask_inbbox]
        if mask_inbbox.all():
            ray_id, step_id = create_full_step_id(mask_inbbox.shape)
        else:
            ray_id = torch.arange(mask_inbbox.shape[0]).view(-1,1).expand_as(mask_inbbox)[mask_inbbox]
            step_id = torch.arange(mask_inbbox.shape[1]).view(1,-1).expand_as(mask_inbbox)[mask_inbbox]
        if self.flag_video:    
            return ray_pts, ray_id, step_id, N_samples, latent_t
        else:
            return ray_pts, ray_id, step_id, N_samples

    def forward(self, rays_o, rays_d, viewdirs, global_step=None, image_t=None, **render_kwargs):
        '''Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        '''
        assert len(rays_o.shape)==2 and rays_o.shape[-1]==3, 'Only suuport point queries in [N, 3] format'
        ret_dict = {}
        N = len(rays_o)
        if self.flag_video:
            latent_t = self.LatentCode(image_t.unsqueeze(-1))
            ray_pts, ray_id, step_id, N_samples, latent_t = self.sample_ray(
                rays_o=rays_o, rays_d=rays_d, latent_t=latent_t, **render_kwargs)
        else:    
            # sample points on rays
            ray_pts, ray_id, step_id, N_samples = self.sample_ray(
                    rays_o=rays_o, rays_d=rays_d, **render_kwargs)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio

        # skip known free space
        if self.mask_cache is not None:   #yes
            mask = self.mask_cache(ray_pts)    #这个函数里面也有cuda算子
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            if self.flag_video:
                latent_t = latent_t[mask]     
                    
        if self.use_fine:
            #hashmlp
            mlp_grid = self.hash(ray_pts)
            if self.flag_video:
                mlp_grid = torch.cat([mlp_grid, latent_t], dim=-1)
            mlp_features = self.mlpnet(mlp_grid)
            density = mlp_features[:,0] + self.act_shift(ray_pts)
            sh_val = mlp_features[:,1:]
            alpha = self.activate_density(density, interval)
            
            if self.fast_color_thres > 0:  #yes
                mask = (alpha > self.fast_color_thres)
                ray_pts = ray_pts[mask]
                ray_id = ray_id[mask]   #rays的标号
                step_id = step_id[mask] #一条光线上深度的标号
                density = density[mask]
                alpha = alpha[mask]
                sh_val = sh_val[mask]
                self.ratio2 += [mask.sum()/mask.shape[0]]
                
            weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
            
            if self.fast_color_thres > 0:
                mask = (weights > self.fast_color_thres)
                weights = weights[mask]
                alpha = alpha[mask]
                ray_pts = ray_pts[mask]
                ray_id = ray_id[mask]
                step_id = step_id[mask]
                sh_val = sh_val[mask]
                self.ratio3 += [mask.sum()/mask.shape[0]]

                                 
            if self.use_sh:
                sh_deg = 2
                viewdirs = viewdirs[ray_id]
                if ray_pts.shape[0] == 0:
                    rgb = torch.zeros_like(ray_pts)
                else:
                    # #-----------将sh系数存入mat文件-------------#
                    # import scipy.io as sio
                    # import numpy as np      
                    # sio.savemat('/data/liufengyi/Results/others/mat/Materials.mat', {'ray_pts': ray_pts, 'sh_val': sh_val})
                    # #-----------end---------#
                    rgb = sh.eval_sh(sh_deg, sh_val.reshape(
                        *sh_val.shape[:-1],-1, (sh_deg + 1) ** 2), viewdirs)
                    m = nn.Sigmoid()
                    rgb = m(rgb)
            else:
                # view-dependent color emission
                viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
                viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
                viewdirs_emb = viewdirs_emb[ray_id]
                rgb_feat = torch.cat([sh_val, viewdirs_emb], -1)
                rgb_logit = self.rgbnet(rgb_feat)
                rgb = torch.sigmoid(rgb_logit)
        else:
            # query for alpha w/ post-activation
            density = self.density(ray_pts) + self.act_shift(ray_pts)
            alpha = self.activate_density(density, interval)   #用到了cuda算子
            if self.fast_color_thres > 0:   #yes
                mask = (alpha > self.fast_color_thres)
                ray_pts = ray_pts[mask]
                ray_id = ray_id[mask]
                step_id = step_id[mask]
                alpha = alpha[mask]

            # compute accumulated transmittance   weights就是T*alpha
            weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
            if self.fast_color_thres > 0:
                mask = (weights > self.fast_color_thres)
                ray_pts = ray_pts[mask]
                ray_id = ray_id[mask]
                step_id = step_id[mask]
                alpha = alpha[mask]
                weights = weights[mask]

            # query for color
            vox_emb = self.k0(ray_pts)

            if self.rgbnet is None:
                # no view-depend effect
                rgb = torch.sigmoid(vox_emb)
            else:
                # view-dependent color emission
                viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
                viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
                viewdirs_emb = viewdirs_emb[ray_id]
                rgb_feat = torch.cat([vox_emb, viewdirs_emb], -1)
                rgb_logit = self.rgbnet(rgb_feat)
                rgb = torch.sigmoid(rgb_logit)

        # Ray marching
        rgb_marched = segment_coo(
                src=(weights.unsqueeze(-1) * rgb),
                index=ray_id,
                out=torch.zeros([N, 3]),
                reduce='sum')
        if render_kwargs.get('rand_bkgd', False) and global_step is not None:
            rgb_marched += (alphainv_last.unsqueeze(-1) * torch.rand_like(rgb_marched))
        else:
            rgb_marched += (alphainv_last.unsqueeze(-1) * render_kwargs['bg'])
        s = (step_id+0.5) / N_samples
        ret_dict.update({
            'alphainv_last': alphainv_last,
            'weights': weights,
            'rgb_marched': rgb_marched,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            'ray_id': ray_id,
            'n_max': N_samples,
            's': s,
        })

        if render_kwargs.get('render_depth', False):
            with torch.no_grad():
                depth = segment_coo(
                        src=(weights * s),
                        index=ray_id,
                        out=torch.zeros([N]),
                        reduce='sum')
            ret_dict.update({'depth': depth})

        return ret_dict


@functools.lru_cache(maxsize=128)
def create_full_step_id(shape):
    ray_id = torch.arange(shape[0]).view(-1,1).expand(shape).flatten()
    step_id = torch.arange(shape[1]).view(1,-1).expand(shape).flatten()
    return ray_id, step_id

