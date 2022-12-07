# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch

from .base import BaseObserver
from .utils import lp_loss
from torch.nn import functional as F


class MinmaxObserver(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(MinmaxObserver, self).__init__(module_type, bit_type,
                                             calibration_mode)
        self.symmetric = self.bit_type.signed

    def update(self, v):
        self.v = v
        v = self.reshape_tensor(v)
        # self.v = v
        cur_max = v.max(axis=1).values
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)

        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()


    def get_quantization_params(self, x, others=None, asymmetric=False, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val
        self.input = x
        self.others = others

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        scale = torch.ones_like(max_val, dtype=torch.float32)
        zero_point = torch.zeros_like(max_val, dtype=torch.int64)

        if asymmetric:
            self.symmetric = False

        def round_ln(x, type=None):
            if type == 'ceil':
                return torch.ceil(torch.div(torch.log(x),torch.log(torch.Tensor([2]).cuda())))
            elif type == 'floor':
                return torch.floor(torch.div(torch.log(x),torch.log(torch.Tensor([2]).cuda())))
            else:
                y = torch.floor(torch.div(torch.log(x),torch.log(torch.Tensor([2]).cuda())))
                out = torch.gt((x-2**y),(2**(y+1)-x))
                return out+y
            # floor = torch.floor(torch.div(torch.log(x),torch.log(torch.Tensor([2]).cuda())))
            # for j in range(self.v.shape[0]):
        def get_out(x, j, quant=False):
            
            if self.calibration_mode == 'channel_wise':
                weight = self.v[j,...].unsqueeze(0)
                if self.others:
                    bias = self.others[0][j].unsqueeze(0)
            else:
                weight = self.v
                if self.others:
                    bias = self.others[0]
            # FIXME:
            # input_gt = self.input[0]
            # input_q = self.input[0]
            input_gt = self.input
            input_q = self.input
            if self.module_type == 'activation':
                if quant == True:
                    return x
                else:
                    # return self.input[0]
                    return self.input
            elif self.module_type == 'conv_weight':
                if quant == True:
                    return F.conv2d(input_q, x, bias, self.others[1], self.others[2], self.others[3], self.others[4])
                    # return x
                else:
                    return F.conv2d(input_gt, weight, bias, self.others[1], self.others[2], self.others[3], self.others[4])
                    # return weight
            elif self.module_type == 'linear_weight': 
                if quant == True:
                    return F.linear(input_q, x, bias)  
                    # return x
                else:
                    return F.linear(input_gt, weight, bias)
                    # return weight 

        def round_x(scale, x, zero_point=0):
            alpha_round = round_ln(scale, 'round').cuda()
            alpha_floor = round_ln(scale, 'floor').cuda()
            alpha = alpha_round
            zero_point = torch.Tensor([zero_point]).cuda()
            # print(scale.shape)
            if self.calibration_mode == 'layer_wise':
                dim = 1
            else:
                dim = scale.shape[0]
            for j in range(dim):
                if dim == 1:
                    weight = x.cuda()
                    if self.module_type == 'activation':
                        # FIXME:
                        # weight = self.input[0]
                        weight = self.input
                else:
                    weight = x[j,...].unsqueeze(0).cuda()
                weight_1 = ((weight / 2**alpha_floor[j] + zero_point).round().clamp(qmin, qmax) -
                zero_point) * 2**alpha_floor[j]
                out_1 = get_out(weight_1, j, quant=True)
                # out_1 = weight_1
                weight_2 = ((weight / 2**(alpha_floor[j]+1) + zero_point).round().clamp(qmin, qmax) -
                zero_point) * 2**(alpha_floor[j]+1)
                out_2 = get_out(weight_2, j, quant=True)
                # out_2 = weight_2
                out = get_out(weight, j, quant=False)
                # out = weight
                score1 = lp_loss(out, out_1, p=2.0, reduction='all')
                score2 = lp_loss(out, out_2, p=2.0, reduction='all')
                score = [score1, score2]
                if score.index(min(score)) == 0:
                    alpha[j] = alpha_floor[j]
                else:
                    alpha[j] = alpha_floor[j]+1
            return alpha

        if self.symmetric:
            zero_point = torch.zeros_like(max_val, dtype=torch.int64)
            max_val = torch.max(-min_val, max_val)
            scale = max_val / (float(qmax - qmin) / 2)
            # TODO: ########### 2^n ############
            alpha_x = round_x(scale, self.v)
            scale = 2**alpha_x
            # if self.module_type in ['conv_weight', 'linear_weight']:
            #     # alpha_x = round_x(scale, self.v)
            #     # scale = 2**alpha_x
            #     pass
            # elif self.module_type == 'activation':
            #     alpha_x = round_x(scale, self.v)
            #     scale = 2**alpha_x
            #     # pass
            # ####################################
            scale.clamp_(self.eps)
        else:
            # zero_point = torch.zeros_like(max_val, dtype=torch.int64)
            scale = (max_val - min_val) / float(qmax - qmin)
            # TODO: ########### 2^n ############
            alpha_x = round_x(scale, self.v)
            scale = 2**alpha_x
            ####################################
            scale.clamp_(self.eps)
            zero_point = qmin - torch.round(min_val / scale)
            zero_point.clamp_(qmin, qmax)
        return scale, zero_point
