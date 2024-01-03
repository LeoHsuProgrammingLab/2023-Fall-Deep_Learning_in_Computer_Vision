model_name = [
    "model_A",
    "model_B",
    "based_cnn",  
    "pre_resnext101_64x4d_v1", 
    "pre_effnet_v2_l_v1", 
    "pre_effnet_b7",
    "pre_regnet_y_128gf_e2e_v1",
    "pre_regnet_y_16gf_e2e_v1"
]

config_B = {
    'num_epochs': 100,
    'patience': 25,
    'model_name': model_name[1],
    'batch_size': 64,
    'lr': 0.000075, 
    'weight_decay': 0.01,
    'log': "stepLR gamma: 0.1, step: 10"
}

config_A = {
    'num_epochs': 100,
    'patience': 25,
    'model_name': model_name[0],
    'batch_size': 64,
    'lr': 0.0005, 
    'weight_decay': 0.01,
    'log': "consineAnnealingLR T_max: 10, eta_min: lr/20"
}