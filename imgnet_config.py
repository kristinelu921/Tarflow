class Config:
    dataset = "imagenet"
    img_size = 224
    patch_size = 16
    n_channels = 3
    dataset_size = 1281167


    #patch embed
    num_patches = (img_size//patch_size)**2
    d_patch = n_channels * patch_size * patch_size
    
    #PE
    max_seq_length = num_patches + 1

    #ViT
    d_model: int = 32
    debug: bool = True
    layer_norm_eps: float = 1e-5
    init_range: float = 0.02
    n_layers = 2 #number of transformer layers
    dropout = 0.1
    r_mlp = 2 #scales size of intermed. layer

    #AttentionHead
    n_heads = 2
    d_head = d_model//n_heads

    #Training
    epochs = 3
    mask = True
    has_scheduler = True
    batch_size = 1000
    eta_min_scale = 0.0001

    #learning rate scheduler
    initial_lr = 1
    weight_decay = 1e-4
    num_warmup_steps = dataset_size//(batch_size*5) #1 epoch
    total_training_steps = epochs*(dataset_size//batch_size)
    lr_min = 1e-4
    lr_max = 5e-4


    #tarflow
    n_flow_steps = 4
    permutation = True
    

    #noising
    noise_std = 0.05

    #evaluation
    evaluate = False
    n_classes = 10

    #guidance
    guidance_on = True
