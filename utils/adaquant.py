import torch
import torch.nn.functional as F
from tqdm import tqdm


def optimize_layer_adaquant(layer, in_out, test_batch_size=100, train_batch_size=50, iters=100, progress=True):
    """ 
    Function that optimizes the given layer with AdaQuant algorithm.
    Params:
        layer - layer from the model
        in_out - list of [in, out] pairs
        test_batch_size
        train_batch_size
        iters - number of iterations
        progress - flag whether to show progress bar using tgdm
    """
    #create cached inputs and outputs
    cached_inps = torch.cat([x[0] for x in in_out]).to(layer.weight.device)
    cached_outs = torch.cat([x[1] for x in in_out]).to(layer.weight.device)

    test_idx = torch.randperm(cached_inps.size(0))[:test_batch_size]
    test_inp = cached_inps[test_idx]
    test_out = cached_outs[test_idx]

    #calculate mse before
    mse_before = F.mse_loss(layer(test_inp), test_out) 

    #learning rate
    # Original hyperparameters from the repo tuned for 8 bit.
    lr_qpin = 1e-1
    lr_qpw = 1e-3
    lr_w = 1e-5
    lr_b = 1e-3

    #optimizer
    opt_w = torch.optim.Adam([layer.weight], lr=lr_w) 
    if hasattr(layer, 'bias') and layer.bias is not None: opt_bias = torch.optim.Adam([layer.bias], lr=lr_b)
    opt_qparams_in = torch.optim.Adam([layer.quantize_input.running_range,
                                       layer.quantize_input.running_zero_point], lr=lr_qpin)
    opt_qparams_w = torch.optim.Adam([layer.quantize_weight.running_range,
                                      layer.quantize_weight.running_zero_point], lr=lr_qpw)

    # loop through iterations
    for j in (tqdm(range(iters)) if progress else range(iters)):

        #choose inputs and outputs
        train_idx = torch.randperm(cached_inps.size(0))[:train_batch_size]
        train_inp = cached_inps[train_idx]
        train_out = cached_outs[train_idx]

        #put input through layer and save loss
        qout = layer(train_inp)
        loss = F.mse_loss(qout, train_out)

        #zero optimizer gradients
        opt_w.zero_grad()
        if hasattr(layer, 'bias') and layer.bias is not None: opt_bias.zero_grad()
        opt_qparams_in.zero_grad()
        opt_qparams_w.zero_grad()

        #backpropagation
        loss.backward()

        #optimizers step
        opt_w.step()
        if hasattr(layer, 'bias') and layer.bias is not None: opt_bias.step()
        opt_qparams_in.step()
        opt_qparams_w.step()

    #calculate mse after
    mse_after = F.mse_loss(layer(test_inp), test_out)

    return mse_before.item(), mse_after.item()

