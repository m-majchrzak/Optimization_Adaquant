import torch
import torch.nn.functional as F
from tqdm import tqdm


def optimize_layer_adaquant(layer, in_out, test_batch_size=100, train_batch_size=50, iters=100, progress=True):
    """ 
    layer - layer from the model
    in_out - list of [in, out] pairs
    test_batch_size
    train_batch_size
    iters - number of iternations
    progress - flag whether to show progress bar using tgdm
    """
    ### CREATE CACHED INPUTS AND OUTPUTS
    cached_inps = torch.cat([x[0] for x in in_out]).to(layer.weight.device)
    cached_outs = torch.cat([x[1] for x in in_out]).to(layer.weight.device)

    test_idx = torch.randperm(cached_inps.size(0))[:test_batch_size]
    test_inp = cached_inps[test_idx]
    test_out = cached_outs[test_idx]

    ### METRIC
    mse_before = F.mse_loss(layer(test_inp), test_out) 

    ### LEARNING RATE
    # lr_factor = 1e-2
    # Those hyperparameters tuned for 8 bit and checked on mobilenet_v2 and resnet50
    # Have to verify on other bit-width and other models
    lr_qpin = 1e-1#lr_factor * (test_inp.max() - test_inp.min()).item()  # 1e-1
    lr_qpw = 1e-3#lr_factor * (layer.weight.max() - layer.weight.min()).item()  # 1e-3
    lr_w = 1e-5#lr_factor * layer.weight.std().item()  # 1e-5
    lr_b = 1e-3#lr_factor * layer.bias.std().item()  # 1e-3

    ### OPTIMIZER
    opt_w = torch.optim.Adam([layer.weight], lr=lr_w) 
    if hasattr(layer, 'bias') and layer.bias is not None: opt_bias = torch.optim.Adam([layer.bias], lr=lr_b)
    opt_qparams_in = torch.optim.Adam([layer.quantize_input.running_range,
                                       layer.quantize_input.running_zero_point], lr=lr_qpin)
    opt_qparams_w = torch.optim.Adam([layer.quantize_weight.running_range,
                                      layer.quantize_weight.running_zero_point], lr=lr_qpw)

    ### LOOP THROUGH ITERATIONS
    losses = [] # optional
    for j in (tqdm(range(iters)) if progress else range(iters)):

        ### CHOOSE INPUTS and OUTPUTS
        train_idx = torch.randperm(cached_inps.size(0))[:train_batch_size] # nie ma loop przez batche - zawsze robimy tylko losową część inputów???
        train_inp = cached_inps[train_idx]#.cuda()
        train_out = cached_outs[train_idx]#.cuda()

        ### PUT INPUT THROUGH LAYER AND SAVE LOSS
        qout = layer(train_inp)
        loss = F.mse_loss(qout, train_out)
        losses.append(loss.item()) # optonal

        ### ZERO OPTIMIZER GRADIENTS
        opt_w.zero_grad()
        if hasattr(layer, 'bias') and layer.bias is not None: opt_bias.zero_grad()
        opt_qparams_in.zero_grad()
        opt_qparams_w.zero_grad()

        ### BACKPROPAGATION
        loss.backward()

        ### OPTIMIZERS STEP
        opt_w.step()
        if hasattr(layer, 'bias') and layer.bias is not None: opt_bias.step()
        opt_qparams_in.step()
        opt_qparams_w.step()

            ### OPTIONAL - save losses
            # if len(losses) < 10:
            #     total_loss = loss.item()
            # else:
            #     total_loss = np.mean(losses[-10:])
            # print("mse out: {}, pc mean loss: {}, total: {}".format(mse_out.item(), mean_loss.item(), total_loss))

    ### METRIC
    mse_after = F.mse_loss(layer(test_inp), test_out)

    return mse_before.item(), mse_after.item()

