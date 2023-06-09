import time
#import logging
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.nn.utils import clip_grad_norm_
from utils.meters import AverageMeter, accuracy
from functools import partial
from utils.quantize import QConv2d, QLinear, quant_round_constrain
import numpy as np 
from torch.nn import functional as F
import copy 

def _flatten_duplicates(inputs, target, batch_first=True, expand_target=True):
    duplicates = inputs.size(1)
    if not batch_first:
        inputs = inputs.transpose(0, 1)
    inputs = inputs.flatten(0, 1)

    if expand_target:
        if batch_first:
            target = target.view(-1, 1).expand(-1, duplicates)
        else:
            target = target.view(1, -1).expand(duplicates, -1)
        target = target.flatten(0, 1)
    return inputs, target


def _average_duplicates(outputs, target, batch_first=True):
    """assumes target is not expanded (target.size(0) == batch_size) """
    batch_size = target.size(0)
    reduce_dim = 1 if batch_first else 0
    if batch_first:
        outputs = outputs.view(batch_size, -1, *outputs.shape[1:])
    else:
        outputs = outputs.view(-1, batch_size, *outputs.shape[1:])
    outputs = outputs.mean(dim=reduce_dim)
    return outputs



class Trainer(object):

    def __init__(self, model, criterion, optimizer, pruner=None,
                 device_ids=[0], device=torch.cuda, dtype=torch.float,
                 distributed=False, local_rank=-1, adapt_grad_norm=None,
                 loss_scale=1., grad_clip=-1, print_freq=100, epoch=0, update_only_th=False, optimize_rounding=False):
        self.model = model
        self.fp_state_dict=copy.deepcopy(model.state_dict())
        self.criterion = criterion
        self.epoch = epoch
        self.training_steps = 0
        self.optimizer = optimizer
        self.pruner = pruner
        self.device = device
        self.dtype = dtype
        self.local_rank = local_rank
        self.print_freq = print_freq
        self.grad_clip = grad_clip
        self.grad_scale = None
        self.loss_scale = loss_scale
        self.adapt_grad_norm = adapt_grad_norm
        self.iter = 0
        self.update_only_th = update_only_th
        self.optimize_rounding = optimize_rounding
        if isinstance(self.criterion,nn.KLDivLoss):
            self.output_embed_fp32=torch.load('output_embed_calib', device)
        
        if update_only_th and not optimize_rounding:
            for name,p in model.named_parameters():
                if 'fc' not in name and 'bias' not in name:
                    p.requires_grad=False

        if distributed:
            self.model = nn.parallel.DistributedDataParallel(model,
                                                             device_ids=device_ids,
                                                             output_device=device_ids[0])
        elif device_ids and len(device_ids) > 1:
            self.model = nn.DataParallel(model, device_ids)
        else:
            self.model = model
        
        self.bias_mean = {}
        for name, module in self.model.named_modules():
            module.register_forward_hook(partial(self.save_activation,name))

    def save_activation(self,name, mod, inp, out):
        
        if isinstance(mod, QConv2d) or isinstance(mod, QLinear) or isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
            
            reduce_dims= (0,2,3) if isinstance(mod, QConv2d) else (0)
            if name in self.bias_mean:
                
                self.bias_mean[name].add_(torch.mean(out.detach(),reduce_dims).cpu())
                self.bias_mean[name+'.count'] += 1
            else:
                self.bias_mean[name] = torch.mean(out.detach(),reduce_dims).cpu()
                self.bias_mean[name+'.count'] = 1

    def _grad_norm(self, inputs_batch, target_batch, chunk_batch=1):
        self.model.zero_grad()
        for inputs, target in zip(inputs_batch.chunk(chunk_batch, dim=0),
                                  target_batch.chunk(chunk_batch, dim=0)):
            target = target.to(self.device)
            inputs = inputs.to(self.device, dtype=self.dtype)

            # compute output
            output = self.model(inputs)
            loss = self.criterion(output, target)

            if chunk_batch > 1:
                loss = loss / chunk_batch

            loss.backward()   # accumulate gradient
        grad = clip_grad_norm_(self.model.parameters(), float('inf'))
        return grad

    def _step(self, inputs_batch, target_batch, training=False, average_output=False, chunk_batch=1):
        outputs = []
        total_loss = 0

        if training:
            self.optimizer.zero_grad()
            self.optimizer.update(self.epoch, self.training_steps)

        for i, (inputs, target) in enumerate(zip(inputs_batch.chunk(chunk_batch, dim=0),
                                                 target_batch.chunk(chunk_batch, dim=0))):
            target = target.to(self.device)
            inputs = inputs.to(self.device, dtype=self.dtype)
            mixup = None
            if training:
                self.optimizer.pre_forward()

            # compute output
            output = self.model(inputs)

            if mixup is not None:
                target = mixup.mix_target(target, output.size(-1))

            if average_output:
                if isinstance(output, list) or isinstance(output, tuple):
                    output = [_average_duplicates(out, target) if out is not None else None
                              for out in output]
                else:
                    output = _average_duplicates(output, target)
            if isinstance(self.criterion,nn.KLDivLoss):
                emb=torch.zeros(output.shape)
                for t in range(target.shape[0]):
                    
                    emb[t]=self.output_embed_fp32[target[t].tolist()]
                    
                loss = self.criterion(F.log_softmax(output), F.softmax(emb.to(output)))
            else:
                loss = self.criterion(output, target)
            grad = None

            if chunk_batch > 1:
                loss = loss / chunk_batch

            if isinstance(output, list) or isinstance(output, tuple):
                output = output[0]

            outputs.append(output.detach())
            total_loss += float(loss)

            if training:
                if i == 0:
                    self.optimizer.pre_backward()
                if self.grad_scale is not None:
                    loss = loss * self.grad_scale
                if self.loss_scale is not None:
                    loss = loss * self.loss_scale
                loss.backward()   # accumulate gradient
                if self.update_only_th and not self.optimize_rounding:
                    for p in self.model.parameters():
                        if p.shape[0]==1000 or p.dim()==2:
                            p.grad=None 
        if training:  # post gradient accumulation
            if self.loss_scale is not None:
                for p in self.model.parameters():
                    if p.grad is None:
                        continue
                    p.grad.data.div_(self.loss_scale)

            if self.grad_clip > 0:
                grad = clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()  # SGD step           
            self.training_steps += 1
            if self.optimize_rounding:
                sd=self.model.state_dict()
                for key in sd:
                    if  'quantize_weight' in key and 'range' in key:
                        trange = sd[key]
                        tzp = sd[key.replace('range','zero_point')]
                        weights_name = key.replace('quantize_weight.running_range','weight')
                        t1 = self.fp_state_dict[weights_name.replace('module.','')]
                        t2 = sd[weights_name]
                        new_weight = quant_round_constrain(t1, t2, trange, tzp)
                        sd[weights_name] = new_weight
        outputs = torch.cat(outputs, dim=0)
        return outputs, total_loss, grad

    def forward(self, data_loader, num_steps=None, training=False, duplicates=1,
                average_output=False, chunk_batch=1,rec=False):
        if rec: output_embed={}
        meters = {name: AverageMeter()
                  for name in ['step', 'data', 'loss', 'prec1', 'prec5']}
        if training and self.grad_clip > 0:
            meters['grad'] = AverageMeter()

        batch_first = True
        if training and isinstance(self.model, nn.DataParallel) or chunk_batch > 1:
            batch_first = False
        if average_output:
            assert duplicates > 1 and batch_first, "duplicates must be > 1 for output averaging"

        def meter_results(meters):
            results = {name: meter.avg for name, meter in meters.items()}
            results['error1'] = 100. - results['prec1']
            results['error5'] = 100. - results['prec5']
            return results

        end = time.time()
        for i, data in enumerate(data_loader, 0):
            inputs, target = data
            inputs = inputs.to(self.device)
            target = target.to(self.device)
        
            if training and duplicates > 1 and self.adapt_grad_norm is not None \
                    and i % self.adapt_grad_norm == 0:
                grad_mean = 0
                num = inputs.size(1)
                for j in range(num):
                    grad_mean += float(self._grad_norm(inputs.select(1, j), target))
                grad_mean /= num
                grad_all = float(self._grad_norm(
                    *_flatten_duplicates(inputs, target, batch_first)))
                self.grad_scale = grad_mean / grad_all

            # measure data loading time
            meters['data'].update(time.time() - end)
            if duplicates > 1:  
                inputs, target = _flatten_duplicates(inputs, target, batch_first,
                                                     expand_target=not average_output)

            output, loss, grad = self._step(inputs, target,
                                            training=training,
                                            average_output=average_output,
                                            chunk_batch=chunk_batch)
            if rec:
                with torch.no_grad():
                    for i in range(target.shape[0]):
                        tt=target[i]
                        emb=output[i]
                        output_embed[tt.tolist()]=emb
            if self.pruner is not None:
                with torch.no_grad():
                    if training:
                        compression_rate = self.pruner.calc_param_masks(self.model,i%self.print_freq==0,i+self.epoch*len(data_loader))
                    self.model=self.pruner.prune_layers(self.model)

            # measure accuracy and record loss
            #prec1, prec5 = accuracy(output, target, topk=(1, 5))
            prec1, prec5 = accuracy(output, target, topk=(1, 1))
            meters['loss'].update(float(loss), inputs.size(0))
            meters['prec1'].update(float(prec1), inputs.size(0))
            meters['prec5'].update(float(prec5), inputs.size(0))
            if grad is not None:
                meters['grad'].update(float(grad), inputs.size(0))

            # measure elapsed time
            meters['step'].update(time.time() - end)
            end = time.time()

            if num_steps is not None and i >= num_steps or (self.update_only_th and training and i>2):
                break
        if self.pruner is not None:
                self.pruner.save_eps(epoch=self.epoch+1)
                self.pruner.save_masks(epoch=self.epoch+1)
        
        if rec: torch.save(output_embed,'output_embed_calib')
        return meter_results(meters)

    def train(self, data_loader, duplicates=1, average_output=False, chunk_batch=1):
        # switch to train mode
        self.model.train()
        return self.forward(data_loader, duplicates=duplicates, training=True, average_output=average_output, chunk_batch=chunk_batch)

    def validate(self, data_loader, average_output=False, duplicates=1,num_steps=None,rec=False):
        # switch to evaluate mode
        self.model.eval()
        with torch.no_grad():
            return self.forward(data_loader, num_steps=num_steps, duplicates=duplicates, average_output=average_output, training=False,rec=rec)

    def cal_bn_stats(self, data_loader, average_output=False, duplicates=1,num_steps=None,rec=False):
        # switch to evaluate mode
        self.model.train()
        with torch.no_grad():
            return self.forward(data_loader, num_steps=num_steps, duplicates=duplicates, average_output=average_output,
                                training=False)

    def get_quantization_params(self,activation_only=True):
        sd=self.model.state_dict()
        Qparams=[]
        for key in sd.keys():
            if 'zero_point' in key or 'range' in key:
                if activation_only and 'weight' in key:
                    continue
                Qvalues = sd[key].view(-1).tolist()
                for qv in Qvalues:
                    Qparams.append(qv)
        return np.array(Qparams)

    def set_quantization_params(self,Qparams,activation_only=True):
        sd=self.model.state_dict()
        count = 0
        for key in sd.keys():
            if 'zero_point' in key or 'range' in key:
                if activation_only and 'weight' in key:
                    continue
                key_shape = sd[key].shape
                num_vq=key_shape[0]
                sd[key].copy_(torch.tensor(Qparams[count:count+num_vq]).to(sd[key]).reshape(key_shape))
                count+=num_vq
    
    def evaluate_calibration_clipped(self,Qparams, data_loader, average_output=False, duplicates=1,num_steps=None):
        print('Powell iteration #',self.iter)
        self.iter+=1
        with torch.no_grad():
            self.set_quantization_params(Qparams)
            res = self.validate(data_loader, average_output=average_output, duplicates=duplicates,num_steps=num_steps)
            return res['error1']
     
