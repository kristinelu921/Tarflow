class Config:
    dataset = "mnist"
    img_size = 28
    patch_size = 14
    n_channels = 1

    #PE
    max_seq_length = (img_size//patch_size)**2 + 1

    #ViT
    d_model: int = 96
    debug: bool = True
    layer_norm_eps: float = 1e-5
    init_range: float = 0.02
    n_layers = 4 #number of transformer layers
    dropout = 0.1
    r_mlp = 2 #scales size of intermed. layer

    #AttentionHead
    n_heads = 2
    d_head = d_model//n_heads

    #classifier
    n_classes = 10 #number of classes to classify into

    #Training
    epochs = 1000

    mask = False
    has_scheduler = False
    batch_size = 2000
    alpha = 5e-4
    eta_min_scale = 0.0001

    #visualization
