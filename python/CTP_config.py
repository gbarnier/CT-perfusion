# Configuration/parameters for training
class config():

    # Flag to display information
    info = True
    debug = False
    rec_freq = 1
    seed = None

    ############################# Data / Labels ################################
    include_dev = True
    train_file_list = []
    dev_file_list = []
    test_file_list = []

    ############################# Training parameters ##########################
    # Training parameters
    num_epochs = 10
    batch_size = 512 # Set to None for batch gradient descent

    # Optimization
    optim_method = 'adam' # Optimization method
    learning_rate = 0.001 # Initial learning rate
    # 4 types of rate decay schedules: 'decay', 'step', 'exp', 'sqrt'
    decay_rate = 0.001 # When using 'decay': lr0/(1+decay_rate*epoch)
    decay_step_size = 2 # When using 'step': Number of epochs between 2 consecutive decays
    decay_gamma = 0.995 # When using 'exp': Multiplicative factor lr_new = lr_old * decay_gamma
    l2_reg_lambda = 0 # Weight decay: L2 regularization

    # Initialization
    init_method = 'xavier'

    # Loss function
    loss_function = 'mse'

    # GPU/CPU
    device = 'cpu'

    # Random
    seed_id = 100
    avg_size = 2

    # Mask
    tmax_sup = 1500 # Upper bound for TMax
    tmax_inf = 0 # Lower bound for TMax
    ct_sup = 120 # Hounsfield unit
    ct_inf = -100 # Hounsefield unit

    # Save paths
    save_model = True

    # Accuracy
    ac_threshold = 0.40 # Accuracy threshold for TMax
    ac_threshold_zone1 = 0.20 # Accuracy threshold for TMax
    ac_threshold_zone2 = 0.40 # Accuracy threshold for TMax
    ac_threshold_zone3 = 0.60 # Accuracy threshold for TMax
    eps_stability = 1.0e-4 # Accuracy threshold for TMax [0.1s]

    ############################# Architectures ################################
    # Basline: 1 hidden layer
    baseline_n_hidden = 40

    # fc6: 5 hidden layers
    fc6_n_hidden = 500,500,500,300,300

    # Last activation function (Lealy ReLU)
    leaky_slope = 0.0

    ############################### Plotting ###################################
    linewidth = 2.0
