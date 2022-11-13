# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch

from .base import BaseObserver
from .utils import lp_loss


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


    def get_quantization_params(self, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        scale = torch.ones_like(max_val, dtype=torch.float32)
        zero_point = torch.zeros_like(max_val, dtype=torch.int64)

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
            
        def round_x(scale, x, zero_point=0):
            alpha_round = round_ln(scale, 'round')
            alpha_floor = round_ln(scale, 'floor')
            alpha = alpha_round
            zero_point = torch.Tensor([zero_point]).cuda()
            # print(scale.shape)
            if self.calibration_mode == 'layer_wise':
                dim = 1
            else:
                dim = scale.shape[0]
            for j in range(dim):
                if dim == 1:
                    weight = x
                else:
                    weight = x[j,...].unsqueeze(0)
                weight_1 = ((weight / 2**alpha_floor[j] + zero_point).round().clamp(qmin, qmax) -
                zero_point) * 2**alpha_floor[j]
                weight_2 = ((weight / 2**(alpha_floor[j]+1) + zero_point).round().clamp(qmin, qmax) -
                zero_point) * 2**(alpha_floor[j]+1)
                score1 = lp_loss(weight, weight_1, p=2.0, reduction='all')
                score2 = lp_loss(weight, weight_2, p=2.0, reduction='all')
                score = [score1, score2]
                if score.index(min(score)) == 0:
                    alpha[j] = alpha_floor[j]
                else:
                    alpha[j] = alpha_floor[j]+1
            return alpha

        if self.symmetric:
            # TODO: add the hardware friendly channel-wise quantization scheme
            # FIXME:
            if self.calibration_mode == 'hw_channel_wise':
                scale_global = torch.ones_like(max_val.max(), dtype=torch.float32)
                # zero_point = torch.zeros_like(max_val.max(), dtype=torch.int64)
                max_val_global = torch.max(-(min_val.min()), max_val.max())
                # FIXME:
                K = 3
                scale_global = max_val_global / (float(qmax - qmin) / 2) / 2**K
                scale_global.clamp_(self.eps)
                scale_original = torch.ones_like(max_val, dtype=torch.float32)
                max_val = torch.max(-min_val, max_val)
                scale_original = max_val / (float(qmax - qmin) / 2)
                scale_original.clamp_(self.eps)
                ratio = torch.ones_like(max_val, dtype=torch.float32)
                ratio = (scale_original / scale_global)

                alpha = round_ln(ratio, K)
                # print(alpha)
                scale = scale_global*2**alpha
                scale.clamp_(self.eps)
                zero_point = torch.zeros_like(max_val, dtype=torch.int64)
                # print(torch.max(-((scale-scale_original)/scale_original).min()*100, ((scale-scale_original)/scale_original).max())*100)
                
            else:
                zero_point = torch.zeros_like(max_val, dtype=torch.int64)
                max_val = torch.max(-min_val, max_val)
                scale = max_val / (float(qmax - qmin) / 2)
                # TODO: ########### 2^n ############
                # if self.module_type in ['conv_weight', 'linear_weight']:
                #     # alpha_round = round_ln(scale, 'round')
                #     alpha_x = round_x(scale, self.v)
                #     # print(alpha_round==alpha_x)
                #     scale = 2**alpha_x
                # elif self.module_type == 'activation':
                #     # alpha_round = round_ln(scale, 'round')
                #     alpha_x = round_x(scale, self.v)
                #     # print(alpha_round==alpha_x)
                #     scale = 2**alpha_x
                # alpha_round = round_ln(scale, 'round')
                alpha_x = round_x(scale, self.v)
                # print(alpha_round==alpha_x)
                scale = 2**alpha_x
            # ####################################
                scale.clamp_(self.eps)
        else:
            # zero_point = torch.zeros_like(max_val, dtype=torch.int64)
            scale = (max_val - min_val) / float(qmax - qmin)
            # ########### 2^n ############
            # alpha = round_ln(scale, 'round')
            # scale = 2**alpha
            ####################################
            scale.clamp_(self.eps)
            zero_point = qmin - torch.round(min_val / scale)
            zero_point.clamp_(qmin, qmax)
        return scale, zero_point
