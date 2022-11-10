# from nets.unet import Unet
# from torch.utils.tensorboard import SummaryWriter
# import torch
# from torch import nn
#
# mioird = Unet()
# input = torch.ones(8, 3, 256, 256)
# output = mioird(input)
#
# writer = SummaryWriter("logs")
# writer.add_graph(mioird, input)
# writer.close()




import torch
import torchvision
from thop import profile
from nets.unet import Unet
# Model
print('==> Building model..')
model = Unet()

dummy_input = torch.ones(8, 3, 256, 256)
flops, params = profile(model, (dummy_input,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))