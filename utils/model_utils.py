from peft import LoraConfig, get_peft_model
import torch
import copy
from transformers import AutoModelForImageClassification, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model

################################### model setup ########################################
def model_setup(args):
    if args.model == 'bert-base-uncased':
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.num_classes)
        config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none"
        )
        net_glob = get_peft_model(model, config)
        net_glob.to(args.device)
    elif args.model == 'google/vit-base-patch16-224-in21k':
        model = AutoModelForImageClassification.from_pretrained(
            args.model,
            label2id=args.label2id,
            id2label=args.id2label,
            ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        )

        config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["classifier"],
        )
        net_glob = get_peft_model(model, config)
        net_glob.to(args.device)
    else:
        exit('Error: unrecognized model')

    global_model = copy.deepcopy(net_glob.state_dict())
    return args, net_glob, global_model, model_dim(global_model)

def model_dim(model):
    '''
    compute model dimension
    '''
    flat = [torch.flatten(model[k]) for k in model.keys()]
    s = 0
    for p in flat: 
        s += p.shape[0]
    return s