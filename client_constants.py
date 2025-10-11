"""
Constants and configuration keys for the Flower client.

This module contains all the constants, default values, and configuration keys
used throughout the Flower client implementation.

Author: Team1-FL-RHLA
Version: 1.0.0
"""

# =============================================================================
# BASIC NUMERIC CONSTANTS
# =============================================================================

DEFAULT_ZERO_VALUE = 0
DEFAULT_ONE_VALUE = 1
DEFAULT_TWO_VALUE = 2
DEFAULT_THREE_VALUE = 3
DEFAULT_TWELVE_VALUE = 12
DEFAULT_SIXTEEN_VALUE = 16
DEFAULT_THIRTY_TWO_VALUE = 32
DEFAULT_ONE_HUNDRED_VALUE = 100
DEFAULT_FIVE_HUNDRED_VALUE = 500
DEFAULT_ONE_THOUSAND_TWENTY_FOUR_VALUE = 1024
DEFAULT_SEVEN_HUNDRED_SIXTY_EIGHT_VALUE = 768
DEFAULT_ONE_FIFTY_FIVE_TWENTY_EIGHT_VALUE = 150528

# =============================================================================
# FLOATING POINT CONSTANTS
# =============================================================================

DEFAULT_ZERO_POINT_ZERO = 0.0
DEFAULT_ZERO_POINT_FIVE = 0.5
DEFAULT_ONE_POINT_ZERO = 1.0
DEFAULT_ZERO_POINT_ZERO_ONE = 0.01
DEFAULT_ZERO_POINT_ZERO_ZERO_ONE = 0.001
DEFAULT_ZERO_POINT_ONE = 0.1

# =============================================================================
# STRING CONSTANTS
# =============================================================================

DEFAULT_DATASET = 'cifar100'
DEFAULT_MODEL = 'google/vit-base-patch16-224-in21k'
DEFAULT_DATA_TYPE = 'image'
DEFAULT_PEFT = 'lora'
DEFAULT_LORA_LAYER = DEFAULT_TWELVE_VALUE
DEFAULT_TAU = DEFAULT_THREE_VALUE
DEFAULT_ROUND = DEFAULT_FIVE_HUNDRED_VALUE
DEFAULT_OPTIMIZER = 'adamw'
DEFAULT_NUM_USERS = DEFAULT_ONE_HUNDRED_VALUE
DEFAULT_NUM_SELECTED_USERS = DEFAULT_ONE_VALUE
DEFAULT_NUM_CLASSES = DEFAULT_ONE_HUNDRED_VALUE
DEFAULT_MODEL_HETEROGENEITY = 'depthffm_fim'
DEFAULT_GROUP_ID = DEFAULT_ZERO_VALUE
DEFAULT_MEMORY_BATCH_SIZE = DEFAULT_SIXTEEN_VALUE
DEFAULT_MEMORY_THRESHOLD = DEFAULT_THIRTY_TWO_VALUE
DEFAULT_PYTORCH_MPS_RATIO = '0.0'
DEFAULT_LOG_PATH_PREFIX = './logs/client_'
DEFAULT_CLASS_PREFIX = 'class_'
DEFAULT_UNKNOWN_VALUE = 'unknown'
DEFAULT_NONE_VALUE = 'none'
DEFAULT_DEFAULT_VALUE = 'default'
DEFAULT_DIRICHLET_TYPE = 'dirichlet'
DEFAULT_CPU_DEVICE = 'cpu'
DEFAULT_CUDA_DEVICE = 'cuda'
DEFAULT_MPS_DEVICE = 'mps'
DEFAULT_IMAGE_DATA_TYPE = 'image'
DEFAULT_TEXT_DATA_TYPE = 'text'
DEFAULT_SENTIMENT_DATA_TYPE = 'sentiment'
DEFAULT_QUERY_MODULE = 'query'
DEFAULT_VALUE_MODULE = 'value'
DEFAULT_DIR_PARTITION_MODE = 'dir'
DEFAULT_ADAMW_OPTIMIZER = 'adamw'
DEFAULT_LORA_PEFT = 'lora'
DEFAULT_CIFAR100_DATASET = 'cifar100'
DEFAULT_LEDGAR_DATASET = 'ledgar'
DEFAULT_BATCH_FORMAT_3_ELEMENTS = DEFAULT_THREE_VALUE
DEFAULT_BATCH_FORMAT_2_ELEMENTS = DEFAULT_TWO_VALUE
DEFAULT_DIMENSION_1 = DEFAULT_ONE_VALUE

# =============================================================================
# CONFIGURATION KEY CONSTANTS
# =============================================================================

CONFIG_KEY_DATASET = 'dataset'
CONFIG_KEY_MODEL = 'model'
CONFIG_KEY_DATA_TYPE = 'data_type'
CONFIG_KEY_PEFT = 'peft'
CONFIG_KEY_LORA_LAYER = 'lora_layer'
CONFIG_KEY_LORA_RANK = 'lora_rank'
CONFIG_KEY_LORA_ALPHA = 'lora_alpha'
CONFIG_KEY_LORA_DROPOUT = 'lora_dropout'
CONFIG_KEY_LORA_TARGET_MODULES = 'lora_target_modules'
CONFIG_KEY_LORA_BIAS = 'lora_bias'
CONFIG_KEY_BATCH_SIZE = 'batch_size'
CONFIG_KEY_LOCAL_LR = 'local_lr'
CONFIG_KEY_TAU = 'tau'
CONFIG_KEY_ROUND = 'round'
CONFIG_KEY_OPTIMIZER = 'optimizer'
CONFIG_KEY_NUM_WORKERS = 'num_workers'
CONFIG_KEY_SHUFFLE_TRAINING = 'shuffle_training'
CONFIG_KEY_DROP_LAST = 'drop_last'
CONFIG_KEY_SHUFFLE_EVAL = 'shuffle_eval'
CONFIG_KEY_DROP_LAST_EVAL = 'drop_last_eval'
CONFIG_KEY_LOGGING_BATCHES = 'logging_batches'
CONFIG_KEY_EVAL_BATCHES = 'eval_batches'
CONFIG_KEY_NUM_USERS = 'num_users'
CONFIG_KEY_NUM_SELECTED_USERS = 'num_selected_users'
CONFIG_KEY_IID = 'iid'
CONFIG_KEY_NONIID_TYPE = 'noniid_type'
CONFIG_KEY_PAT_NUM_CLS = 'pat_num_cls'
CONFIG_KEY_PARTITION_MODE = 'partition_mode'
CONFIG_KEY_DIR_CLS_ALPHA = 'dir_cls_alpha'
CONFIG_KEY_DIR_PAR_BETA = 'dir_par_beta'
CONFIG_KEY_MODEL_HETEROGENEITY = 'model_heterogeneity'
CONFIG_KEY_FREEZE_DATASPLIT = 'freeze_datasplit'
CONFIG_KEY_NUM_CLASSES = 'num_classes'
CONFIG_KEY_SEED = 'seed'
CONFIG_KEY_GPU_ID = 'gpu_id'
CONFIG_KEY_FORCE_CPU = 'force_cpu'
CONFIG_KEY_HETEROGENEOUS_GROUP = 'heterogeneous_group'
CONFIG_KEY_USER_GROUPID_LIST = 'user_groupid_list'
CONFIG_KEY_BLOCK_IDS_LIST = 'block_ids_list'
CONFIG_KEY_LABEL2ID = 'label2id'
CONFIG_KEY_ID2LABEL = 'id2label'
CONFIG_KEY_LOGGER = 'logger'
CONFIG_KEY_ACCELERATOR = 'accelerator'
CONFIG_KEY_LOG_PATH = 'log_path'
CONFIG_KEY_DEVICE = 'device'
CONFIG_KEY_DATASET_INFO = 'dataset_info'
CONFIG_KEY_CLIENT_DATA_INDICES = 'client_data_indices'
CONFIG_KEY_DATA_COLLATOR = 'data_collator'
CONFIG_KEY_TRAIN_SAMPLES = 'train_samples'
CONFIG_KEY_TEST_SAMPLES = 'test_samples'
CONFIG_KEY_DATA_LOADED = 'data_loaded'
CONFIG_KEY_DATASET_NAME = 'dataset_name'
CONFIG_KEY_MODEL_NAME = 'model_name'
CONFIG_KEY_LABELS = 'labels'
CONFIG_KEY_LOSSES = 'losses'
CONFIG_KEY_ACCURACIES = 'accuracies'
CONFIG_KEY_ROUNDS = 'rounds'
CONFIG_KEY_PIXEL_VALUES = 'pixel_values'
CONFIG_KEY_TOTAL_LOSS = 'total_loss'
CONFIG_KEY_TOTAL_CORRECT = 'total_correct'
CONFIG_KEY_TOTAL_SAMPLES = 'total_samples'
CONFIG_KEY_NUM_BATCHES = 'num_batches'
CONFIG_KEY_TR_LABELS = '_tr_labels'

# =============================================================================
# DEFAULT CONFIGURATION PATHS
# =============================================================================

DEFAULT_CONFIG_PATH = "experiments/flower/cifar100_vit_lora/fim/image_cifar100_vit_fedavg_fim-6_9_12-noniid-pat_10_dir-noprior-s50-e50.yaml"

# =============================================================================
# NETWORK CONFIGURATION
# =============================================================================

DEFAULT_SERVER_ADDRESS = "localhost"
DEFAULT_SERVER_PORT = 8080
DEFAULT_CLIENT_ID = DEFAULT_ZERO_VALUE

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

DEFAULT_SEED = DEFAULT_ONE_VALUE
DEFAULT_GPU_ID = -1
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_TRAINING_STEPS_PER_EPOCH = 10
DEFAULT_MIN_LEARNING_RATE = DEFAULT_ZERO_POINT_ZERO_ZERO_ONE
DEFAULT_LEARNING_RATE = DEFAULT_ZERO_POINT_ZERO_ONE
DEFAULT_LOCAL_EPOCHS = DEFAULT_ONE_VALUE

# =============================================================================
# MODEL ARCHITECTURE CONSTANTS
# =============================================================================

VIT_BASE_HIDDEN_SIZE = DEFAULT_SEVEN_HUNDRED_SIXTY_EIGHT_VALUE
VIT_LARGE_HIDDEN_SIZE = DEFAULT_ONE_THOUSAND_TWENTY_FOUR_VALUE
LORA_RANK = 64
LORA_ALPHA = DEFAULT_SIXTEEN_VALUE
LORA_DROPOUT = DEFAULT_ZERO_POINT_ONE
LORA_TARGET_MODULES = [DEFAULT_QUERY_MODULE, DEFAULT_VALUE_MODULE]
LORA_BIAS = DEFAULT_NONE_VALUE

# =============================================================================
# TRAINING CONSTANTS
# =============================================================================

DEFAULT_LOGGING_BATCHES = DEFAULT_THREE_VALUE  # Number of batches to log during training
DEFAULT_EVAL_BATCHES = DEFAULT_THREE_VALUE     # Number of batches to log during evaluation
DEFAULT_FLATTENED_SIZE_CIFAR = DEFAULT_ONE_FIFTY_FIVE_TWENTY_EIGHT_VALUE  # 3*224*224 for CIFAR-100 with ViT

# =============================================================================
# ERROR MESSAGES
# =============================================================================

ERROR_NO_DATA_INDICES = "Client {client_id} has no data indices"
ERROR_INVALID_BATCH_FORMAT = "Invalid batch format: {batch_type}"
ERROR_NO_TEST_DATASET = "No test dataset available for evaluation"
ERROR_NO_TRAINING_DATASET = "No actual dataset found for training"
ERROR_INVALID_CONFIG = "Configuration dictionary cannot be None or empty"
ERROR_INVALID_DATASET = "Unknown dataset: {dataset_name}"

# =============================================================================
# LOGGING MESSAGES
# =============================================================================

LOG_CLIENT_INITIALIZED = "Client {client_id} initialized with {num_cores} CPU cores"
LOG_DATASET_LOADED = "Successfully loaded dataset: {dataset_name}"
LOG_TRAINING_COMPLETED = "Client {client_id} completed training: Loss={loss:.4f}"
LOG_EVALUATION_COMPLETED = "Client {client_id} evaluation: Loss={loss:.4f}, Accuracy={accuracy:.4f}"

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

DATASET_CONFIGS = {
    DEFAULT_CIFAR100_DATASET: {CONFIG_KEY_NUM_CLASSES: DEFAULT_ONE_HUNDRED_VALUE, CONFIG_KEY_DATA_TYPE: DEFAULT_IMAGE_DATA_TYPE},
    DEFAULT_LEDGAR_DATASET: {CONFIG_KEY_NUM_CLASSES: DEFAULT_TWO_VALUE, CONFIG_KEY_DATA_TYPE: DEFAULT_TEXT_DATA_TYPE},
}

# =============================================================================
# NON-IID CONFIGURATION DEFAULTS
# =============================================================================

DEFAULT_NONIID_TYPE = DEFAULT_DIRICHLET_TYPE
DEFAULT_PAT_NUM_CLS = 10
DEFAULT_PARTITION_MODE = DEFAULT_DIR_PARTITION_MODE
DEFAULT_DIR_ALPHA = DEFAULT_ZERO_POINT_FIVE
DEFAULT_DIR_BETA = DEFAULT_ONE_POINT_ZERO

# =============================================================================
# DATA PROCESSING CONSTANTS
# =============================================================================

DEFAULT_BATCH_SIZE = 128
DEFAULT_NUM_WORKERS = DEFAULT_ZERO_VALUE  # Avoid multiprocessing issues in federated setting
DEFAULT_SHUFFLE_TRAINING = True
DEFAULT_DROP_LAST = True
DEFAULT_SHUFFLE_EVAL = False
DEFAULT_DROP_LAST_EVAL = False
