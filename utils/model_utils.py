from peft import LoraConfig, get_peft_model, LoKrConfig

import torch
import torch.nn as nn
import copy

from transformers import AutoModelForImageClassification, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model

################################### model setup ########################################
def model_setup(args):
    """
    Set up and configure models for federated learning with LoRA (Low-Rank Adaptation).
    
    This function initializes different types of pre-trained models and applies LoRA configuration
    for efficient fine-tuning in federated learning scenarios. It supports both text classification
    (BERT) and image classification (Vision Transformer) models.
    
    Args:
        args: Configuration object containing model parameters. Expected attributes:
            - model (str): Model identifier ('bert-base-uncased' or 'facebook/deit-small-patch16-224')
            - num_classes (int): Number of output classes for classification
            - device (torch.device): Device to move the model to (CPU/GPU)
            - lora_max_rank (int): LoRA rank (r) and alpha for PEFT config
            - LOKR (bool): If True, use LoKrConfig instead of LoraConfig
            - FlexLoRA (bool): If True (ViT only), apply Kaiming init to lora_B
            - label2id (dict): Mapping from labels to IDs (for ViT models)
            - id2label (dict): Mapping from IDs to labels (for ViT models)
    
    Returns:
        tuple: A tuple containing:
            - args: The input arguments (unchanged)
            - net_glob: The configured model with LoRA adapters applied
            - global_model: Deep copy of the model's state dictionary for federated learning
            - model_dim: Total number of parameters in the model
    
    Supported Models:
        1. BERT (args.model == 'bert-base-uncased'):
           - Loads 'google/bert_uncased_L-12_H-128_A-2' via AutoModelForSequenceClassification
           - LoRA/LoKR config: r=args.lora_max_rank, alpha=args.lora_max_rank, targets query/value
           - Dropout: 0.1, no bias; LoKR used if args.LOKR is True
        
        2. Vision Transformer (args.model == 'facebook/deit-small-patch16-224'):
           - Loads facebook/deit-small-patch16-224 with label2id/id2label from args
           - LoRA/LoKR config: r=args.lora_max_rank, alpha=args.lora_max_rank, targets query/value
           - modules_to_save=["classifier"]; FlexLoRA: Kaiming init for lora_B if args.FlexLoRA
    
    Raises:
        SystemExit: If an unrecognized model is specified
    
    Example:
        >>> args = argparse.Namespace()
        >>> args.model = 'bert-base-uncased'
        >>> args.num_classes = 10
        >>> args.device = torch.device('cuda')
        >>> args, model, global_state, dim = model_setup(args)
    """
    if args.model == 'bert-base-uncased':
        model = AutoModelForSequenceClassification.from_pretrained(
            'google/bert_uncased_L-12_H-128_A-2', # https://huggingface.co/google/bert_uncased_L-4_H-256_A-4
            num_labels=args.num_classes
        )
        config = LoraConfig(
            r=args.lora_max_rank,
            lora_alpha=args.lora_max_rank,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none"
        )

        if args.LOKR:
            config = LoKrConfig(
                r=args.lora_max_rank,
                alpha=args.lora_max_rank,
                target_modules=["query", "value"],
            )

        net_glob = get_peft_model(model, config)
        net_glob.to(args.device)
    elif args.model == 'facebook/deit-small-patch16-224':
        model = AutoModelForImageClassification.from_pretrained(
            'facebook/deit-small-patch16-224',
            label2id=args.label2id,
            id2label=args.id2label,
            ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        )

        config = LoraConfig(
            r=args.lora_max_rank,
            lora_alpha=args.lora_max_rank,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["classifier"],
        )

        if args.LOKR:
            config = LoKrConfig(
                r=args.lora_max_rank,
                alpha=args.lora_max_rank,
                target_modules=["query", "value"],
                modules_to_save=["classifier"],
            )

        net_glob = get_peft_model(model, config)

        if args.FlexLoRA:
            with torch.no_grad():
                for name, param in net_glob.named_parameters():
                    if "lora_B" in name:
                        # recommended: small std so you don't blow up training
                        nn.init.kaiming_uniform_(param, a=0, mode="fan_in", nonlinearity="linear")

        net_glob.to(args.device)
    else:
        exit('Error: unrecognized model')

    global_model = copy.deepcopy(net_glob.state_dict())

    return args, net_glob, global_model, model_dim(global_model)

def model_dim(model):
    """
    Compute total number of parameters (dimension) in a state dict.

    Args:
        model: State dict (e.g. from model.state_dict()) or dict of tensors.

    Returns:
        int: Sum of numel() over all tensors in model.
    """
    flat = [torch.flatten(model[k]) for k in model.keys()]
    s = 0
    for p in flat: 
        s += p.shape[0]
    return s


def model_clip(model, clip):
    """
    Clip the model update (in place) by global norm; skip batch-norm stats.

    Computes L2 norm over all tensors except those with 'num_batches_tracked',
    'running_mean', or 'running_var' in the key. If total_norm > clip, scales
    all (non-skipped) tensors by clip / total_norm.

    Args:
        model: State dict of updates (modified in place).
        clip (float): Maximum allowed global norm.

    Returns:
        tuple: (model, total_norm) — the same dict and the norm before clipping.
    """
    model_norm=[]
    for k in model.keys():
        if 'num_batches_tracked' in k or 'running_mean' in k or 'running_var' in k:
            continue
        model_norm.append(torch.norm(model[k]))
        
    total_norm = torch.norm(torch.stack(model_norm))
    clip_coef = clip / (total_norm + 1e-8)
    if clip_coef < 1:
        for k in model.keys():
            if 'num_batches_tracked' in k or 'running_mean' in k or 'running_var' in k:
                continue
            model[k] = model[k] * clip_coef
    return model, total_norm

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save training state to a file.

    Args:
        state: Object to save (e.g. dict with model state_dict, epoch, optimizer).
        is_best: Unused; reserved for saving a separate best checkpoint.
        filename: Path for the checkpoint file (default 'checkpoint.pth.tar').
    """
    torch.save(state, filename)

def get_trainable_values(net, mydevice=None):
    """
    Flatten all trainable parameter values into a single 1D tensor.

    Parameters are traversed in the same order as net.parameters(); only
    those with requires_grad=True are included.

    Args:
        net: PyTorch module.
        mydevice (optional): Device to place the output tensor on; default is CPU.

    Returns:
        torch.Tensor: 1D float tensor of length equal to total trainable numel().
    """
    trainable=filter(lambda p: p.requires_grad, net.parameters())
    paramlist=list(trainable) 
    N=0
    for params in paramlist:
        N+=params.numel()
    if mydevice:
        X=torch.empty(N,dtype=torch.float).to(mydevice)
    else:
        X=torch.empty(N,dtype=torch.float)
    X.fill_(0.0)
    offset=0
    for params in paramlist:
        numel=params.numel()
        with torch.no_grad():
            X[offset:offset+numel].copy_(params.data.view_as(X[offset:offset+numel].data))
        offset+=numel

    return X

def put_trainable_values(net, X):
    """
    Copy values from a 1D tensor back into the net's trainable parameters (in place).

    Order of parameters must match the order used by get_trainable_values(net).
    Only parameters with requires_grad=True are updated.

    Args:
        net: PyTorch module to update.
        X: 1D tensor of length equal to total trainable numel().
    """
    trainable=filter(lambda p: p.requires_grad, net.parameters())
    paramlist=list(trainable)
    offset=0
    for params in paramlist:
        numel=params.numel()
        with torch.no_grad():
            params.data.copy_(X[offset:offset+numel].data.view_as(params.data))
        offset+=numel
