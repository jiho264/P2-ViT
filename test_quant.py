import argparse
import math
import os
import time
import random

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

from config import Config
from models import *
from generate_data import generate_data
import numpy as np

parser = argparse.ArgumentParser(description='FQ-ViT')

parser.add_argument('model',
                    choices=[
                        'deit_tiny', 'deit_small', 'deit_base', 'vit_base',
                        'vit_large', 'swin_tiny', 'swin_small', 'swin_base'
                    ],
                    help='model')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--quant', default=False, action='store_true')
parser.add_argument('--ptf', default=False, action='store_true')
parser.add_argument('--lis', default=False, action='store_true')
parser.add_argument('--quant-method',
                    default='minmax',
                    choices=['minmax', 'ema', 'omse', 'percentile'])
# TODO: 100 --> 32
parser.add_argument('--calib-batchsize',
                    default=100,
                    type=int,
                    help='batchsize of calibration set')
parser.add_argument("--mode", default=0,
                        type=int, 
                        help="mode of calibration data, 0: PSAQ-ViT, 1: Gaussian noise, 2: Real data")
# TODO: 10 --> 1
parser.add_argument('--calib-iter', default=1, type=int)
# TODO: 100 --> 200
parser.add_argument('--val-batchsize',
                    default=200,
                    type=int,
                    help='batchsize of validation set')
parser.add_argument('--num-workers',
                    default=16,
                    type=int,
                    help='number of data loading workers (default: 16)')
parser.add_argument('--device', default='cuda', type=str, help='device')
parser.add_argument('--print-freq',
                    default=100,
                    type=int,
                    help='print frequency')
parser.add_argument('--seed', default=0, type=int, help='seed')


def str2model(name):
    d = {
        'deit_tiny': deit_tiny_patch16_224,
        'deit_small': deit_small_patch16_224,
        'deit_base': deit_base_patch16_224,
        'vit_base': vit_base_patch16_224,
        'vit_large': vit_large_patch16_224,
        'swin_tiny': swin_tiny_patch4_window7_224,
        'swin_small': swin_small_patch4_window7_224,
        'swin_base': swin_base_patch4_window7_224,
    }
    print('Model: %s' % d[name].__name__)
    return d[name]


def seed(seed=0):
    import os
    import random
    import sys

    import numpy as np
    import torch
    sys.setrecursionlimit(100000)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def main():
    args = parser.parse_args()
    seed(args.seed)

    device = torch.device(args.device)
    cfg = Config(args.ptf, args.lis, args.quant_method)
    model = str2model(args.model)(pretrained=True, cfg=cfg)
    model = model.to(device)

    # Note: Different models have different strategies of data preprocessing.
    model_type = args.model.split('_')[0]
    if model_type == 'deit':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        crop_pct = 0.875
    elif model_type == 'vit':
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        crop_pct = 0.9
    elif model_type == 'swin':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        crop_pct = 0.9
    else:
        raise NotImplementedError

    train_transform = build_transform(mean=mean, std=std, crop_pct=crop_pct)
    val_transform = build_transform(mean=mean, std=std, crop_pct=crop_pct)

    # Data
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    val_dataset = datasets.ImageFolder(valdir, val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    # switch to evaluate mode
    model.eval()

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().to(device)

    train_dataset = datasets.ImageFolder(traindir, train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.calib_batchsize,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # # TODO: Compute the hessian metrics
    # from pyhessian import hessian
    
    # # TODO:
    # #####################################################
    # print("Calculating the sensitiveties via the averaged Hessian trace.......")
    # batch_num = 10
    # trace_list = []
    # for i, (inputs, labels) in enumerate(train_loader):
    #     hessian_comp = hessian(model,
    #                     criterion,
    #                     data=(inputs, labels),
    #                     cuda=args.device)
    #     name, trace = hessian_comp.trace()
    #     trace_list.append(trace)
    #     if i == batch_num - 1:
    #         break
   
    # # top_eigenvalues, _ = hessian_comp.eigenvalues()
    # # trace = hessian_comp.trace()
    # # density_eigen, density_weight = hessian_comp.density()
    # # print('\n***Top Eigenvalues: ', top_eigenvalues)

    # new_global_hessian_track = []
    # for i in range(int(len(trace_list))):
    #     hessian_track = trace_list[i]
    #     hessian_track = [abs(x) for x in hessian_track]
    #     min_h = min(hessian_track)
    #     max_h = max(hessian_track)
    #     averaged_hessian_track = [(elem-min_h)/(max_h-min_h) for elem in hessian_track]
    #     new_global_hessian_track.append(averaged_hessian_track)

    # mean_hessian = []
    # # min_hessian = []
    # # max_hessian = []
    # layer_num = len(trace_list[0])
    # for i in range(layer_num):
    #     new_hessian = [sample[i] for sample in new_global_hessian_track]
    #     mean_hessian.append(sum(new_hessian)/len(new_hessian))
    #     # min_hessian.append(min(new_hessian))
    #     # max_hessian.append(max(new_hessian))

    # print(name)
    # print('\n***Trace: ', mean_hessian)
    # # exit()
    mean_hessian = [0.21988081262180068, 0.48965385469843303, 0.5190564507106766, 0.7903327571286561, 0.10463252547458368, 0.1974192417775237, 0.37749819890654507, 0.22746509088392797, 0.14045321394501942, 0.12046402409483636, 0.4566360596171609, 0.2082136539084841, 0.16708860889720462, 0.1612862728954133, 0.45526429921049394, 0.25559916217836, 0.17358442670588808, 0.10532683129218154, 0.23995013507557209, 0.1450345256250551, 0.13427725741648433, 0.0870769326883701, 0.26997701990544803, 0.12656191796912142, 0.10580154408793159, 0.08128132656112048, 0.2945377070797008, 0.1003372146935972, 0.12117457131886937, 0.05840371962761006, 0.2747470243680096, 0.061144536954956905, 0.11709974765208778, 0.08695703422752873, 0.1799415141938365, 0.07401239390934349, 0.08430724488507586, 0.06216955048200734, 0.09268464562725678, 0.03967463814904212, 0.10754412847578602, 0.04580277281507984, 0.1182123877698068, 0.07063687242841005, 0.15168113162644142, 0.025325995895543452, 0.12378261427663503, 0.11848142789545772, 0.4281059426189756]
    sita_hessian = [0.16471642756017052, 0.37703807272681056, 0.2669296062294753, 0.280619880055184, 0.07192150856067434, 0.12733051867508172, 0.19656188251559956, 0.1396863415228546, 0.08161124748659379, 0.07037187608196936, 0.2691757775510773, 0.12860847143558368, 0.07032010921878985, 0.05824189538797418, 0.24455103672036063, 0.1355432804278672, 0.09121229995658382, 0.05037362030028039, 0.14155561524215743, 0.05878120603469259, 0.06025201422323511, 0.04301338145307761, 0.1209065041722117, 0.055967609636498675, 0.04942733843530334, 0.038744058290335996, 0.1844578689431991, 0.052479752534973924, 0.06541915587376337, 0.030472129164648955, 0.15762227719729446, 0.037278239105564, 0.056581356288052956, 0.04628033857150743, 0.1575758574294753, 0.034639664475995535, 0.04181781516300422, 0.03698485541798021, 0.0851281972428118, 0.026407393967599665, 0.05798196718889691, 0.034291354399930474, 0.1270577165446494, 0.051395675802022256, 0.07844756939060754, 0.02100678900426062, 0.09144228128021381, 0.0661321837088853, 0.19682371636714371]
    #####################################################

    if args.quant:
        # TODO:
        # Get calibration set
        # Case 0: PASQ-ViT
        if args.mode == 2:
            print("Generating data...")
            calibrate_data = generate_data(args)
            print("Calibrating with generated data...")
            model.model_open_calibrate()
            with torch.no_grad():
                model.model_open_last_calibrate()
                output = model(calibrate_data)
        # Case 1: Gaussian noise
        elif args.mode == 1:
            calibrate_data = torch.randn((args.calib_batchsize, 3, 224, 224)).to(device)
            print("Calibrating with Gaussian noise...")
            model.model_open_calibrate()
            with torch.no_grad():
                model.model_open_last_calibrate()
                output = model(calibrate_data)
        # Case 2: Real data (Standard)
        elif args.mode == 0:
            # Get calibration set.
            image_list = []
            for i, (data, target) in enumerate(train_loader):
                if i == args.calib_iter:
                    break
                data = data.to(device)
                image_list.append(data)

            print("Calibrating with real data...")
            model.model_open_calibrate()
            with torch.no_grad():
                # TODO:
                # for i, image in enumerate(image_list):
                #     if i == len(image_list) - 1:
                #         # This is used for OMSE method to
                #         # calculate minimum quantization error
                #         model.model_open_last_calibrate()
                #     output = model(image)
                # model.model_quant(flag='off')
                model.model_open_last_calibrate()
                output, FLOPs, global_distance = model(image_list[0])

        model.model_close_calibrate()
        model.model_quant()

    # FIXME:
    # #####################################################
    # print("Pareto Frontier.......")
    # assert len(FLOPs)-1 == len(global_distance) == len(mean_hessian) == len(sita_hessian)
    # bit_list = []
    # # model size constraint
    # # TODO:
    # model_constraint = 1.1*sum([FLOPs[i]*4 for i in range(len(FLOPs))])
    # for i in range(2**len(global_distance)):
    #     # bit_config = [random.choice([torch.Tensor([4]).cuda(),torch.Tensor([8]).cuda()]) for i in range(len(FLOPs))]
    #     # TODO:
    #     bit_choice = [4,8]
    #     # bit_config = [random.choice(bit_choice) for i in range(len(FLOPs))]
    #     bit_config = [random.choice(bit_choice) for i in range(len(FLOPs)//2-1)]
    #     new_bit_config = [max(bit_choice)] + [bit for bit in bit_config for i in range(2)] + [random.choice(bit_choice)]
    #     # new_bit_config = [7] + [bit for bit in bit_config for i in range(2)] + [6]
    #     model_size = sum([FLOPs[i]*new_bit_config[i] for i in range(len(FLOPs))])
    #     # FIXME:
    #     if not model_size > model_constraint and new_bit_config not in bit_list:
    #         bit_list.append(new_bit_config)
    #     if len(bit_list) > 50:
    #         break
    
    # # compute the omega 
    # omega_list = []
    # for bit_config in bit_list:
    #     select_diastance = []
    #     for i, bit in enumerate(bit_config):
    #         if i == 0:
    #             continue
    #         for k, choice in enumerate(bit_choice):
    #             if choice == bit:
    #                 select_diastance.append(global_distance[i-1][k])
    #                 break
    #         # if bit == 4:
    #         #     select_diastance.append(global_distance[i][0])
    #         # elif bit == 6:
    #         #     select_diastance.append(global_distance[i][1])
    #         # elif bit == 8:
    #         #     select_diastance.append(global_distance[i][2])
    #         # elif bit == 10:
    #         #     select_diastance.append(global_distance[i][3])
    #         # else:
    #         #     assert bit == 4 or bit == 6 or bit == 8
    #     # TODO:
    #     # omega = [(mean_hessian[i]+sita_hessian[i])*select_diastance[i] for i in range(len(FLOPs))]
    #     omega = [mean_hessian[i]*select_diastance[i] for i in range(len(FLOPs)-1)]
    #     omega_list.append([bit_config, sum(omega)])
    
    # # sort and selection
    # omega_list.sort(key = lambda x : x[-1])
    # #####################################################
    # print('Hessien-Based Validating...')
    # for i in range(5):
    #     # FIXME:
    #     bit_config = omega_list[i][0]
    #     # bit_config = [random.choice([5,6,7]) for i in range(len(FLOPs))]
    #     # bit_config = [6]*50
    #     # bit_config = [6, 7, 7, 7, 7, 6, 6, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 5, 5, 6, 6, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 6, 6, 5, 5, 5, 5, 5, 5, 6]
    #     # model_size = sum([FLOPs[i]*bit_config[i] for i in range(len(FLOPs))])
    #     # model_size = 0
    #     # FIXME:
    #     # if not model_size > model_constraint:
    #     print(bit_config)
    #     val_loss, val_prec1, val_prec5 = validate(args, val_loader, model,
    #                                         criterion, device, bit_config)
    #     print('')
    
    # exit()

    # ####################### Evolutionary search ###################
    # print('Start Evolutionary.......')
    # parent_popu = []
    # pop_size = 25
    # evo_iter = 8
    # mutate_size = 10
    # mutate_prob = 0.5
    # crossover_size = 10
    # crossover_prob = 0.5
    # for i in range(pop_size):
    #     bit_config = omega_list[i][0]
    #     val_loss, val_prec1, val_prec5 = validate(args, val_loader, model,
    #                                         criterion, device, bit_config)
    #     parent_popu.append([bit_config, val_prec1])
    # parent_popu.sort(key = lambda x : x[-1], reverse=True)
    
    # for evo in range(evo_iter):                                      
    #     # mutate
    #     children_list =[]
    #     mutate_bit_list =[]
    #     while True:
    #         old_bit = random.choice(parent_popu)[0]
    #         new_bit = [bit if random.random() < mutate_prob else random.choice(bit_choice) for bit in old_bit]
    #         model_size = sum([FLOPs[i]*new_bit[i] for i in range(len(FLOPs))])
    #         if not model_size > model_constraint and new_bit not in mutate_bit_list:
    #             val_loss, val_prec1, val_prec5 = validate(args, val_loader, model,
    #                                         criterion, device, new_bit)
    #         mutate_bit_list.append(new_bit)
    #         children_list.append([new_bit, val_prec1])   
    #         if len(mutate_bit_list) > mutate_size:
    #             break
        
    #     # crossover
    #     crossover_bit_list =[]
    #     while True:
    #         old_bit_1 = random.choice(parent_popu)[0]
    #         old_bit_2 = random.choice(parent_popu)[0]
    #         if old_bit_1 == old_bit_2:
    #             continue
    #         new_bit = [bit1 if random.random() < crossover_prob else bit2 for (bit1, bit2) in zip(old_bit_1, old_bit_2)]
    #         model_size = sum([FLOPs[i]*new_bit[i] for i in range(len(FLOPs))])
    #         if not model_size > model_constraint and new_bit not in crossover_bit_list:
    #             val_loss, val_prec1, val_prec5 = validate(args, val_loader, model,
    #                                         criterion, device, new_bit)
    #         crossover_bit_list.append(new_bit)
    #         children_list.append([new_bit, val_prec1])   
    #         if len(crossover_bit_list) > crossover_size:
    #             break
        
    #     # updation
    #     for child in children_list:
    #         if child[1] > parent_popu[-1][1]:
    #             parent_popu.append(child)

    #     parent_popu.sort(key = lambda x : x[-1], reverse=True)
    #     parent_popu = parent_popu[:pop_size]
    #     print("Evolotionary iteration: ", evo)
    #     print(parent_popu)
    #     print('')
    # ###############################################################

    bit_config = [8]*50
    print(bit_config)
    val_loss, val_prec1, val_prec5 = validate(args, val_loader, model,
                                            criterion, device, bit_config)


def validate(args, val_loader, model, criterion, device, bit_config=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    val_start_time = end = time.time()
    for i, (data, target) in enumerate(val_loader):
        data = data.to(device)
        target = target.to(device)
        if i == 0:
            plot_flag = True
        else:
            plot_flag = False
        with torch.no_grad():
            output, FLOPs, distance = model(data, bit_config, plot_flag)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), data.size(0))
        top1.update(prec1.data.item(), data.size(0))
        top5.update(prec5.data.item(), data.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     print('Test: [{0}/{1}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
        #           'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
        #               i,
        #               len(val_loader),
        #               batch_time=batch_time,
        #               loss=losses,
        #               top1=top1,
        #               top5=top5,
        #           ))
    val_end_time = time.time()
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Time {time:.3f}'.
          format(top1=top1, top5=top5, time=val_end_time - val_start_time))

    return losses.avg, top1.avg, top5.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def build_transform(input_size=224,
                    interpolation='bicubic',
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    crop_pct=0.875):

    def _pil_interp(method):
        if method == 'bicubic':
            return Image.BICUBIC
        elif method == 'lanczos':
            return Image.LANCZOS
        elif method == 'hamming':
            return Image.HAMMING
        else:
            return Image.BILINEAR

    resize_im = input_size > 32
    t = []
    if resize_im:
        size = int(math.floor(input_size / crop_pct))
        ip = _pil_interp(interpolation)
        t.append(
            transforms.Resize(
                size,
                interpolation=ip),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


if __name__ == '__main__':
    main()
