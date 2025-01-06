import torch

def update_params(filepath):
    model = torch.load(filepath, map_location=torch.device('cpu'))
    new_params = {}
    for key in model["model"].keys():
        if key.split('.')[1] == 'text_decoder':
            parts = key.split('.')
            new_key = f'sem_seg_head.predictor.' + '.'.join(parts[1:])
            new_params[new_key] = model["model"][key]
            print(new_key)
    return new_params
            
new_params = update_params('grit_b_densecap.pth')
xdecoder_params = torch.load('xdecoder_focall_last.pt', map_location=torch.device('cpu'))
xdecoder_params.update(new_params)
torch.save(xdecoder_params, 'xdecoder_focall_last_updated.pt')