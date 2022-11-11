# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch

from .base import BaseObserver


class MinmaxObserver(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(MinmaxObserver, self).__init__(module_type, bit_type,
                                             calibration_mode)
        self.symmetric = self.bit_type.signed

    def update(self, v):
        v = self.reshape_tensor(v)
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
            # y = torch.floor(torch.div(torch.log(x),torch.log(torch.Tensor([2]).cuda())))
            # out = torch.gt((x-2**y),(2**(y+1)-x))
            # # print((out+y).min())
            # if k:
            #     # # TODO:
            #     # return torch.clamp((out+y), min=-1, max=k-1)
            #     return torch.clamp((out+y), min=0, max=k)
            # else:
            #     return out+y
                # return y
            ############### round() ##################
            # y = (torch.div(torch.log(x),torch.log(torch.Tensor([2]).cuda()))).round()
            # # print(y)
            # # TODO:
            # return torch.clamp(y, min=-1, max=k-1)
            # return torch.clamp(y, min=0, max=k)

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
                #     alpha = round_ln(scale, 'round')
                # # elif self.module_type == 'activation':
                # #     # FIXME:
                # #     alpha = round_ln(scale, 'floor')
                #     scale = 2**alpha
                # ####################################
                scale.clamp_(self.eps)
        else:
            # zero_point = torch.zeros_like(max_val, dtype=torch.int64)
            scale = (max_val - min_val) / float(qmax - qmin)
            # TODO: ########### 2^n ############
            # alpha = round_ln(scale)
            # scale = 2**alpha
            ####################################
            scale.clamp_(self.eps)
            zero_point = qmin - torch.round(min_val / scale)
            zero_point.clamp_(qmin, qmax)
        return scale, zero_point
