from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction
import scipy.optimize as opt
import numpy as np
import os

QParams = namedtuple('QParams', ['range', 'zero_point', 'num_bits'])

_DEFAULT_FLATTEN = (1, -1)
_DEFAULT_FLATTEN_GRAD = (0, -1)


def _deflatten_as(x, x_full):
    shape = list(x.shape) + [1] * (x_full.dim() - x.dim())
    return x.view(*shape)


methods = ['Nelder-Mead','Powell','COBYLA']


def lp_norm(x, xq, p):
    err = torch.mean(torch.abs(xq - x) ** p)
    return err


def mse(x, xq):
    err = torch.mean((xq - x) ** 2)
    return err


def tensor_range(x, pcq=False):
    if pcq:
        return x.view(x.shape[0], -1).max(dim=-1)[0] - x.view(x.shape[0], -1).min(dim=-1)[0]
    else:
        return x.max() - x.min()


def zero_point(x, pcq=False):
    if pcq:
        return x.view(x.shape[0], -1).min(dim=-1)[0]
    else:
        return x.min()


def quant_err(p, t, num_bits=4, metric='mse'):
    qp = QParams(range=t.new_tensor(p[0]), zero_point=t.new_tensor(p[1]), num_bits=num_bits)
    tq = quantize_with_grad(t, num_bits=qp.num_bits, qparams=qp)
    # TODO: Add other metrics
    return mse(t, tq).item()

def quant_round_constrain(t1, t2, trange, tzp):
    qp = QParams(range=t1.new_tensor(trange), zero_point=t1.new_tensor(tzp), num_bits=4)
    t1q = quantize_with_grad(t1, num_bits=qp.num_bits, qparams=qp, dequantize=False)
    t2q = quantize_with_grad(t2, num_bits=qp.num_bits, qparams=qp, dequantize=False)
    out=torch.max(torch.min(t2q,t1q+1),t1q-1)
    # TODO: Add other metrics
    return dequantize(out,num_bits=qp.num_bits, qparams=qp)

def calculate_qparams(x, num_bits, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0,  reduce_type='mean', keepdim=False, true_zero=False,per_ch_input=False,quant_mode = 'maxmin'):
    alpha_gaus = {1:1.24,2:1.71,3:2.215,4:2.55,5:2.93,6:3.28,7:3.61,8:3.92}
    alpha_gaus_positive = {1:1.71,2:2.215,3:2.55,4:2.93,5:3.28,6:3.61,7:3.92,8:4.2}

    alpha_laplas = {1:1.05,2:1.86,3:2.83,4:5.03,5:6.2,6:7.41,7:8.64,8:9.89}
    alpha_laplas_positive = {1:1.86,2:2.83,3:5.03,4:6.2,5:7.41,6:8.64,7:9.89,8:11.16}
    if per_ch_input:
        x = x.transpose(0,1)
    with torch.no_grad():
        x_flat = x.flatten(*flatten_dims)
        if quant_mode =='mean_std' and num_bits<8:
            mu   = x_flat.mean() if x_flat.dim() == 1 else x_flat.mean(-1)
            std  = x_flat.std() if x_flat.dim() == 1 else x_flat.std(-1)
            b = torch.abs(x_flat-mu).mean() if x_flat.dim() == 1 else torch.mean(torch.abs(x_flat-mu.unsqueeze(1)),-1)
            minv = x_flat.min() if x_flat.dim() == 1 else x_flat.min(-1)[0]
            maxv = x_flat.max() if x_flat.dim() == 1 else x_flat.max(-1)[0]
            min_values = _deflatten_as(torch.max(mu - 6*std,minv), x)  
            max_values = _deflatten_as(torch.min(mu + 6*std,maxv), x)
        else:
            if x_flat.dim() == 1:
                min_values = _deflatten_as(x_flat.min(), x)
                max_values = _deflatten_as(x_flat.max(), x)
            else:
                min_values = _deflatten_as(x_flat.min(-1)[0], x)
                max_values = _deflatten_as(x_flat.max(-1)[0], x)
        if reduce_dim is not None:
            if reduce_type == 'mean':
                min_values = min_values.mean(reduce_dim, keepdim=keepdim)
                max_values = max_values.mean(reduce_dim, keepdim=keepdim)
            else:
                min_values = min_values.min(reduce_dim, keepdim=keepdim)[0]
                max_values = max_values.max(reduce_dim, keepdim=keepdim)[0]

        min_values[min_values > 0] = 0
        max_values[max_values < 0] = 0
        range_values = max_values - min_values
        range_values[range_values==0] = 1
        return QParams(range=range_values, zero_point=min_values,
                       num_bits=num_bits)


class UniformQuantize(InplaceFunction):

    @staticmethod
    def forward(ctx, input, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN,
                reduce_dim=0, dequantize=True, signed=True, stochastic=False, inplace=False):

        ctx.inplace = inplace
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if qparams is None:
            assert num_bits is not None, "either provide qparams of num_bits to quantize"
            qparams = calculate_qparams(
                input, num_bits=num_bits, flatten_dims=flatten_dims, reduce_dim=reduce_dim)

        zero_point = qparams.zero_point
        num_bits = qparams.num_bits
        qmin = -(2.**(num_bits - 1)) if signed else 0.
        qmax = qmin + 2.**num_bits - 1.
        running_range=qparams.range.clamp(min=1e-6,max=1e5)
        scale = running_range / (qmax - qmin)
        running_zero_point_round = Round().apply(qmin-zero_point/scale,False)
        zero_point = (qmin-running_zero_point_round.clamp(qmin,qmax))*scale    
        output.add_(qmin * scale - zero_point).div_(scale)
        if stochastic:
            noise = output.new(output.shape).uniform_(-0.5, 0.5)
            output.add_(noise)
        # quantize
        output.clamp_(qmin, qmax).round_()
        if dequantize:
            output.mul_(scale).add_(
                zero_point - qmin * scale)  # dequantize
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None, None, None, None, None, None

class Round(InplaceFunction):

    @staticmethod
    def forward(ctx, input,inplace):

        ctx.inplace = inplace                                                                          
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()
        output.round_()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input,None



class UniformQuantizeGrad(InplaceFunction):

    @staticmethod
    def forward(ctx, input, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN_GRAD,
                reduce_dim=0, dequantize=True, signed=False, stochastic=True):
        ctx.num_bits = num_bits
        ctx.qparams = qparams
        ctx.flatten_dims = flatten_dims
        ctx.stochastic = stochastic
        ctx.signed = signed
        ctx.dequantize = dequantize
        ctx.reduce_dim = reduce_dim
        ctx.inplace = False
        return input

    @staticmethod
    def backward(ctx, grad_output):
        qparams = ctx.qparams
        with torch.no_grad():
            if qparams is None:
                assert ctx.num_bits is not None, "either provide qparams of num_bits to quantize"
                qparams = calculate_qparams(
                    grad_output, num_bits=ctx.num_bits, flatten_dims=ctx.flatten_dims, reduce_dim=ctx.reduce_dim, reduce_type='extreme')

            grad_input = quantize(grad_output, num_bits=None,
                                  qparams=qparams, flatten_dims=ctx.flatten_dims, reduce_dim=ctx.reduce_dim,
                                  dequantize=True, signed=ctx.signed, stochastic=ctx.stochastic, inplace=False)
        return grad_input, None, None, None, None, None, None, None


def conv2d_biprec(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, num_bits_grad=None):
    out1 = F.conv2d(input.detach(), weight, bias,
                    stride, padding, dilation, groups)
    out2 = F.conv2d(input, weight.detach(), bias.detach() if bias is not None else None,
                    stride, padding, dilation, groups)
    out2 = quantize_grad(out2, num_bits=num_bits_grad, flatten_dims=(1, -1))
    return out1 + out2 - out1.detach()


def linear_biprec(input, weight, bias=None, num_bits_grad=None):
    out1 = F.linear(input.detach(), weight, bias)
    out2 = F.linear(input, weight.detach(), bias.detach()
                    if bias is not None else None)
    out2 = quantize_grad(out2, num_bits=num_bits_grad)
    return out1 + out2 - out1.detach()


def quantize_with_grad(input, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0, dequantize=True, signed=True, stochastic=False, inplace=False):
                                                                        
    if inplace:
        output = input
    else:
        output = input.clone()
    if qparams is None:
        assert num_bits is not None, "either provide qparams of num_bits to quantize"
        qparams = calculate_qparams(
            input, num_bits=num_bits, flatten_dims=flatten_dims, reduce_dim=reduce_dim)
    zero_point = qparams.zero_point
    num_bits = qparams.num_bits
    qmin = -(2.**(num_bits - 1)) if signed else 0.
    qmax = qmin + 2.**num_bits - 1.
    # ZP quantization for HW compliance
    running_range=qparams.range.clamp(min=1e-6,max=1e5)
    scale = running_range / (qmax - qmin)
    running_zero_point_round = Round().apply(qmin-zero_point/scale,False)
    zero_point = (qmin-running_zero_point_round.clamp(qmin,qmax))*scale
    output.add_(qmin * scale - zero_point).div_(scale)
    if stochastic:
        noise = output.new(output.shape).uniform_(-0.5, 0.5)
        output.add_(noise)
    # quantize
    output = Round().apply(output.clamp_(qmin, qmax),inplace)
    if dequantize:
        output.mul_(scale).add_(
            zero_point - qmin * scale)  # dequantize
    return output

def dequantize(input, num_bits=None, qparams=None,signed=False, inplace=False):
                                                                        
    if inplace:
        output = input
    else:
        output = input.clone()
    zero_point = qparams.zero_point
    num_bits = qparams.num_bits
    qmin = -(2.**(num_bits - 1)) if signed else 0.
    qmax = qmin + 2.**num_bits - 1.
    scale = qparams.range / (qmax - qmin)        
    output.mul_(scale).add_(
        zero_point - qmin * scale)  # dequantize
    return output

def quantize(x, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0, dequantize=True, signed=False, stochastic=False, inplace=False):
    return UniformQuantize().apply(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed, stochastic, inplace)


def quantize_grad(x, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN_GRAD, reduce_dim=0, dequantize=True, signed=False, stochastic=True):
    return UniformQuantizeGrad().apply(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed, stochastic)


class QuantMeasure(nn.Module):
    def __init__(self, num_bits=8, shape_measure=(1,), flatten_dims=_DEFAULT_FLATTEN,
                 inplace=False, dequantize=True, stochastic=False, momentum=0.1, measure=False,per_ch_input=False,reduce_dim=0, cal_qparams=False):
        super(QuantMeasure, self).__init__()
        self.register_buffer('running_zero_point', torch.zeros(*shape_measure))
        self.register_buffer('running_range', torch.zeros(*shape_measure))
        self.measure = measure
        if self.measure:
            self.register_buffer('num_measured', torch.zeros(1))
        self.flatten_dims = flatten_dims
        self.momentum = momentum
        self.dequantize = dequantize
        self.stochastic = stochastic
        self.inplace = inplace
        self.num_bits = num_bits
        self.per_ch_input = per_ch_input
        self.reduce_dim = reduce_dim
        self.cal_qparams = cal_qparams

    def forward(self, input, qparams=None):

        if self.training or self.measure:
            if qparams is None:
                if self.cal_qparams:
                    init = np.array([tensor_range(input, pcq=False).item(), zero_point(input, pcq=False).item()])
                    res = opt.minimize(lambda p: quant_err(p, input, num_bits=self.num_bits, metric='mse'), init, method=methods[0])
                    qparams = QParams(range=input.new_tensor(res.x[0]), zero_point=input.new_tensor(res.x[1]), num_bits=self.num_bits)
                    print("Measure and optimize: bits - {}, error before - {:.6f}, error after {:.6f}".format(self.num_bits, quant_err(init, input), res.fun))
                else:
                    reduce_dim = None if self.per_ch_input else self.reduce_dim
                    qparams = calculate_qparams(input, num_bits=self.num_bits, flatten_dims=self.flatten_dims, reduce_dim=reduce_dim,per_ch_input=self.per_ch_input)

            with torch.no_grad():
                if self.measure:
                    momentum = self.num_measured / (self.num_measured + 1)
                    self.num_measured += 1
                else:
                    momentum = self.momentum
                self.running_zero_point.mul_(momentum).add_(
                    qparams.zero_point * (1 - momentum))
                self.running_range.mul_(momentum).add_(
                    qparams.range * (1 - momentum))
        else:
            qparams = QParams(range=self.running_range,
                              zero_point=self.running_zero_point, num_bits=self.num_bits)
        if self.measure:
            return input
        else:
            if self.per_ch_input: input=input.transpose(0,1)
            q_input = quantize(input, qparams=qparams, dequantize=self.dequantize,
                               stochastic=self.stochastic, inplace=self.inplace)
            if self.per_ch_input: q_input=q_input.transpose(0,1)
            return q_input


class QuantThUpdate(nn.Module):
    def __init__(self, num_bits=8, shape_measure=(1,), flatten_dims=_DEFAULT_FLATTEN,
                 inplace=False, dequantize=True, stochastic=False, momentum=0.1, measure=False,per_ch_input=False,reduce_dim=0):
        super(QuantThUpdate, self).__init__()
        self.running_zero_point = nn.Parameter(torch.ones(*shape_measure))
        self.running_range = nn.Parameter(torch.ones(*shape_measure))
        self.measure = measure
        self.flatten_dims = flatten_dims
        self.dequantize = dequantize
        self.stochastic = stochastic
        self.inplace = inplace
        self.num_bits = num_bits
        self.per_ch_input = per_ch_input
        self.reduce_dim = reduce_dim

    def forward(self, input, qparams=None):
        qparams = QParams(range=self.running_range,
                          zero_point=self.running_zero_point, num_bits=self.num_bits)
        
        if self.per_ch_input: input=input.transpose(0,1)
        q_input = quantize_with_grad(input, qparams=qparams, dequantize=self.dequantize,
                           stochastic=self.stochastic, inplace=self.inplace)
        if self.per_ch_input: q_input=q_input.transpose(0,1)
        return q_input



class QConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, num_bits=8, num_bits_weight=8, num_bits_grad=None, perC=True, biprecision=False, measure=False, cal_qparams=False):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits
        self.num_bits_grad = num_bits_grad
        self.measure = measure
        self.equ_scale = nn.Parameter(torch.ones(out_channels, 1, 1, 1))
        if measure:
            self.quantize_input = QuantMeasure(
                self.num_bits, shape_measure=(1, 1, 1, 1), flatten_dims=(1, -1), measure=measure, cal_qparams=cal_qparams)
            self.quantize_weight = QuantMeasure(
                self.num_bits, shape_measure=(out_channels if perC else 1, 1, 1, 1), flatten_dims=(1,-1) if perC else (0,-1), measure=measure, reduce_dim=None if perC else 0)
        else:
            self.quantize_input = QuantThUpdate(
                self.num_bits, shape_measure=(1, 1, 1, 1), flatten_dims=(1, -1), measure=measure)
            self.quantize_weight = QuantThUpdate(
                self.num_bits, shape_measure=(out_channels if perC else 1, 1, 1, 1), flatten_dims=(1,-1) if perC else (0,-1), measure=measure, reduce_dim=None if perC else 0)
        self.biprecision = biprecision
        self.cal_params = cal_qparams
        self.quantize = True

    def forward(self, input):
        qinput = self.quantize_input(input) if self.quantize else input
        qweight = self.quantize_weight(self.weight * self.equ_scale) if self.quantize and not self.cal_params else self.weight  
        if not self.measure and os.environ.get('DEBUG')=='True':
            assert  qinput.unique().numel()<=2**self.num_bits
            assert  qweight[0].unique().numel()<=2**self.num_bits_weight
        if self.bias is not None:
            qbias = self.bias if (self.measure or not self.quantize) else quantize(self.bias, num_bits=self.num_bits_weight + self.num_bits,flatten_dims=(0, -1))
        else:
            qbias = None
        if not self.biprecision or self.num_bits_grad is None:
            output = F.conv2d(qinput, qweight, qbias, self.stride,
                              self.padding, self.dilation, self.groups)
            if self.num_bits_grad is not None:
                output = quantize_grad(
                    output, num_bits=self.num_bits_grad, flatten_dims=(1, -1))
        else:
            output = conv2d_biprec(qinput, qweight, qbias, self.stride,
                                   self.padding, self.dilation, self.groups, num_bits_grad=self.num_bits_grad)
        return output



class QLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, num_bits=8, num_bits_weight=8, num_bits_grad=None, perC=True, biprecision=False,measure=False, cal_qparams=False):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits
        self.num_bits_grad = num_bits_grad
        self.biprecision = biprecision
        self.equ_scale = nn.Parameter(torch.ones(out_features, 1))
        if measure:
            self.quantize_input = QuantMeasure(self.num_bits,measure=measure, cal_qparams=cal_qparams)
            self.quantize_weight = QuantMeasure(self.num_bits,shape_measure=(out_features if perC else 1, 1), flatten_dims=(1,-1) if perC else (0,-1), measure=measure,reduce_dim=None if perC else 0)
        else:
            self.quantize_input = QuantThUpdate(self.num_bits,measure=measure)
            self.quantize_weight = QuantThUpdate(self.num_bits,shape_measure=(out_features if perC else 1, 1), flatten_dims=(1,-1) if perC else (0,-1), measure=measure,reduce_dim=None if perC else 0)
        self.measure = measure
        self.cal_params = cal_qparams
        self.quantize = True

    def forward(self, input):
        qinput = self.quantize_input(input) if self.quantize else input
        qweight = self.quantize_weight(self.weight * self.equ_scale) if self.quantize and not self.cal_params else self.weight
        if not self.measure and os.environ.get('DEBUG')=='True':
            assert  qinput.unique().numel()<=2**self.num_bits
            assert  qweight[0].unique().numel()<=2**self.num_bits_weight
        if self.bias is not None:
            qbias = self.bias if (self.measure or not self.quantize) else quantize(
                self.bias, num_bits=self.num_bits_weight + self.num_bits,
                flatten_dims=(0, -1))
        else:
            qbias = None

        if not self.biprecision or self.num_bits_grad is None:
            output = F.linear(qinput, qweight, qbias)
            if self.num_bits_grad is not None:
                output = quantize_grad(
                    output, num_bits=self.num_bits_grad)
        else:
            output = linear_biprec(qinput, qweight, qbias, self.num_bits_grad)
        return output
