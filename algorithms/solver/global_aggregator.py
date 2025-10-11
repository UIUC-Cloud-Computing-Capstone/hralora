def average_lora_depthfl(args, global_model, loc_updates):
    '''
    hetero average
    '''
    model_update_avg_dict = {}

    for k in global_model.keys():
        if 'lora' in k or 'classifier' in k:
            for loc_update in loc_updates:
                if k in loc_update:
                    if k in model_update_avg_dict:
                        model_update_avg_dict[k].append(loc_update[k])
                    else:
                        model_update_avg_dict[k] = []
                        model_update_avg_dict[k].append(loc_update[k])

    for k in global_model.keys():
        if k in model_update_avg_dict:
            global_model[k] = global_model[k].detach().cpu() +  sum(model_update_avg_dict[k]) / len(model_update_avg_dict[k])

    return global_model
    '''
    weighted hetero average
    '''
    model_update_avg_dict = {}
    model_weights_cnt = {}
    model_weights_list = {}

    for k in global_model.keys():
        if 'lora' in k or 'classifier' in k: # classifier is not included
            for client_i, loc_update in enumerate(loc_updates):
                if k in loc_update:
                    if k in model_update_avg_dict:
                        model_update_avg_dict[k].append(loc_update[k])
                        model_weights_cnt[k] += num_samples[client_i]
                        model_weights_list[k].append(num_samples[client_i])
                    else:
                        model_update_avg_dict[k] = []
                        model_update_avg_dict[k].append(loc_update[k])
                        model_weights_cnt[k] = 0.0
                        model_weights_cnt[k] += num_samples[client_i]
                        model_weights_list[k] = []
                        model_weights_list[k].append(num_samples[client_i])

    for k in global_model.keys():
        if k in model_update_avg_dict:
            weight_based = model_weights_cnt[k]
            model_weights_list[k] = [item/weight_based for item in model_weights_list[k]]
            global_model[k] = global_model[k].detach().cpu() + sum([model*weight for model, weight in zip(model_update_avg_dict[k], model_weights_list[k])])

    return global_model