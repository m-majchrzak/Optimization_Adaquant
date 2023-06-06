import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import scipy.optimize as opt
import math

methods = ['Nelder-Mead','Powell','COBYLA']

def adaquant(layer, cached_inps, cached_outs, test_inp, test_out, lr1=1e-4, lr2=1e-2, iters=100, progress=True, batch_size=50):
    print("\nRun adaquant")
    mse_before = F.mse_loss(layer(test_inp), test_out)

    # lr_factor = 1e-2
    # Those hyperparameters tuned for 8 bit and checked on mobilenet_v2 and resnet50
    # Have to verify on other bit-width and other models
    lr_qpin = 1e-1#lr_factor * (test_inp.max() - test_inp.min()).item()  # 1e-1
    lr_qpw = 1e-3#lr_factor * (layer.weight.max() - layer.weight.min()).item()  # 1e-3
    lr_w = 1e-5#lr_factor * layer.weight.std().item()  # 1e-5
    lr_b = 1e-3#lr_factor * layer.bias.std().item()  # 1e-3

    opt_w = torch.optim.Adam([layer.weight], lr=lr_w)
    if hasattr(layer, 'bias') and layer.bias is not None: opt_bias = torch.optim.Adam([layer.bias], lr=lr_b)
    opt_qparams_in = torch.optim.Adam([layer.quantize_input.running_range,
                                       layer.quantize_input.running_zero_point], lr=lr_qpin)
    opt_qparams_w = torch.optim.Adam([layer.quantize_weight.running_range,
                                      layer.quantize_weight.running_zero_point], lr=lr_qpw)

    losses = []
    for j in (tqdm(range(iters)) if progress else range(iters)):
        idx = torch.randperm(cached_inps.size(0))[:batch_size]

        train_inp = cached_inps[idx]#.cuda()
        train_out = cached_outs[idx]#.cuda()

        qout = layer(train_inp)
        loss = F.mse_loss(qout, train_out)

        losses.append(loss.item())
        opt_w.zero_grad()
        if hasattr(layer, 'bias') and layer.bias is not None: opt_bias.zero_grad()
        opt_qparams_in.zero_grad()
        opt_qparams_w.zero_grad()
        loss.backward()
        opt_w.step()
        if hasattr(layer, 'bias') and layer.bias is not None: opt_bias.step()
        opt_qparams_in.step()
        opt_qparams_w.step()

            # if len(losses) < 10:
            #     total_loss = loss.item()
            # else:
            #     total_loss = np.mean(losses[-10:])
            # print("mse out: {}, pc mean loss: {}, total: {}".format(mse_out.item(), mean_loss.item(), total_loss))

    mse_after = F.mse_loss(layer(test_inp), test_out)
    return mse_before.item(), mse_after.item()


def optimize_layer(layer, in_out, optimize_weights=False):
    batch_size = 100

    # if layer.name == 'features.17.conv.0.0' or layer.name == 'features.17.conv.1.0':
    #     dump("mobilenet_v2", layer, in_out)
    # return 0, 0, 0, 0, 0, 0

    cached_inps = torch.cat([x[0] for x in in_out]).to(layer.weight.device)
    cached_outs = torch.cat([x[1] for x in in_out]).to(layer.weight.device)

    idx = torch.randperm(cached_inps.size(0))[:batch_size]

    test_inp = cached_inps[idx]
    test_out = cached_outs[idx]

    # mse_before, mse_after = optimize_qparams(layer, cached_inps, cached_outs, test_inp, test_out)
    # mse_before_opt = mse_before
    # print("MSE before qparams: {}".format(mse_before))
    # print("MSE after qparams: {}".format(mse_after))

    if optimize_weights:
        mse_before, mse_after = adaquant(layer, cached_inps, cached_outs, test_inp, test_out, iters=100, lr1=1e-5, lr2=1e-4)
        mse_before_opt = mse_before
        print("MSE before adaquant: {}".format(mse_before))
        print("MSE after adaquant: {}".format(mse_after))
        torch.cuda.empty_cache()
    else:
        mse_before, mse_after = optimize_qparams(layer, cached_inps, cached_outs, test_inp, test_out)
        mse_before_opt = mse_before
        print("MSE before qparams: {}".format(mse_before))
        print("MSE after qparams: {}".format(mse_after))

    mse_after_opt = mse_after

    with torch.no_grad():
        N = test_out.numel()
        snr_before = (1/math.sqrt(N)) * math.sqrt(N * mse_before_opt) / torch.norm(test_out).item()
        snr_after = (1/math.sqrt(N)) * math.sqrt(N * mse_after_opt) / torch.norm(test_out).item()

    # optimize_rounding(layer, cached_inps, cached_outs, test_inp, test_out, iters=7000)
    # optimize_qparams(layer, cached_inps, cached_outs, test_inp, test_out)
    # optimize_rounding(layer, cached_inps, cached_outs, test_inp, test_out, iters=2000)
    # optimize_qparams(layer, cached_inps, cached_outs, test_inp, test_out)
    # optimize_rounding(layer, cached_inps, cached_outs, test_inp, test_out, iters=2000)
    # optimize_qparams(layer, test_inp, test_out)

    kurt_in = kurtosis(test_inp).item()
    kurt_w = kurtosis(layer.weight).item()

    del cached_inps
    del cached_outs
    torch.cuda.empty_cache()

    return mse_before_opt, mse_after_opt, snr_before, snr_after, kurt_in, kurt_w


def kurtosis(x):
    var = torch.mean((x - x.mean())**2)
    return torch.mean((x - x.mean())**4 / var**2)


def dump(model_name, layer, in_out):
    path = os.path.join("dump", model_name, layer.name)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

    if hasattr(layer, 'groups'):
        f = open(os.path.join(path, "groups_{}".format(layer.groups)), 'x')
        f.close()

    cached_inps = torch.cat([x[0] for x in in_out])
    cached_outs = torch.cat([x[1] for x in in_out])
    torch.save(cached_inps, os.path.join(path, "input.pt"))
    torch.save(cached_outs, os.path.join(path, "output.pt"))
    torch.save(layer.weight, os.path.join(path, 'weight.pt'))
    if layer.bias is not None:
        torch.save(layer.bias, os.path.join(path, 'bias.pt'))
