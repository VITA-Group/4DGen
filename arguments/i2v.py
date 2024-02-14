OptimizationParams = dict(
    static_iterations = 1099,
    coarse_iterations = 1000,
    iterations = 3000, # don't set it to 0 !!!
    position_lr_max_steps = 3000,
    position_lr_delay_mult = 1,  #1,
    pruning_interval = 100,
    pruning_interval_fine = 100000,
    percent_dense = 0.01,
    densify_grad_threshold_fine_init = 0.5,
    densify_grad_threshold_coarse = 0.01,
    densify_grad_threshold_after = 0.1,
    densification_interval = 100,
    opacity_reset_interval = 100, # not used
    lambda_lpips = 2,
    lambda_dssim = 2,
    lambda_pts = 0,
    lambda_zero123 = 0.5, # default 0.5
    fine_rand_rate = 0.8
)

ModelParams = dict(
    frame_num = 14,
    name="toy0",
    rife=False,
)

ModelHiddenParams = dict(
    grid_merge = 'cat',
    # grid_merge = 'mul',
    multires = [1, 2, 4, 8 ],
    defor_depth = 2,
    net_width = 256,
    plane_tv_weight = 0,
    time_smoothness_weight = 0,
    l1_time_planes =  0,
    weight_decay_iteration=0,
    bounds=2,
    no_ds=True,
    # no_dr=True,
    no_do=True,
    no_dc=True,
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 32,
     'resolution': [64, 64, 64, 7]  #8 is frame numbers/2
    }
)
