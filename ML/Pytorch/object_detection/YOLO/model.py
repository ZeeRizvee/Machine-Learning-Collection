from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import BatchNorm2d
from torch.nn import LeakyReLU
from torch.nn import MaxPool2d
from torch.nn import Linear
from torch.nn import Flatten
from torch.nn import Sequential
from torch import flatten
import torch

class Yolov1(Module):
    def __init__(self, in_channels=3, split_size=7, num_boxes=2, num_classes=20):
        super(Yolov1, self).__init__()

        self.conv_block1 = Sequential(Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
                            BatchNorm2d(64), LeakyReLU(0.1), MaxPool2d(kernel_size=2, stride=2))

        self.conv_block2 = Sequential(Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1, bias=False),
                            BatchNorm2d(192), LeakyReLU(0.1), MaxPool2d(kernel_size=2, stride=2))

        self.conv_block3 = Sequential(Conv2d(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
                            BatchNorm2d(128), LeakyReLU(0.1),
                            Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                            BatchNorm2d(256), LeakyReLU(0.1),
                            Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
                            BatchNorm2d(256), LeakyReLU(0.1),
                            Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
                            BatchNorm2d(512), LeakyReLU(0.1), MaxPool2d(kernel_size=2, stride=2))

        self.conv_block4 = Sequential(Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
                            BatchNorm2d(256), LeakyReLU(0.1),
                            Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
                            BatchNorm2d(512), LeakyReLU(0.1),
                            Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
                            BatchNorm2d(256), LeakyReLU(0.1),
                            Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
                            BatchNorm2d(512), LeakyReLU(0.1),
                            Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
                            BatchNorm2d(256), LeakyReLU(0.1),
                            Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
                            BatchNorm2d(512), LeakyReLU(0.1),
                            Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
                            BatchNorm2d(256), LeakyReLU(0.1),
                            Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
                            BatchNorm2d(512), LeakyReLU(0.1),
                            Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False),
                            BatchNorm2d(512), LeakyReLU(0.1),
                            Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
                            BatchNorm2d(1024), LeakyReLU(0.1), MaxPool2d(kernel_size=2, stride=2))

        self.conv_block5 = Sequential(Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False),
                            BatchNorm2d(512), LeakyReLU(0.1),
                            Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
                            BatchNorm2d(1024), LeakyReLU(0.1),
                            Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False),
                            BatchNorm2d(512), LeakyReLU(0.1),
                            Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
                            BatchNorm2d(1024), LeakyReLU(0.1),
                            Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
                            BatchNorm2d(1024), LeakyReLU(0.1),
                            Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1, bias=False),
                            BatchNorm2d(1024), LeakyReLU(0.1))

        self.conv_block6 = Sequential(Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
                            BatchNorm2d(1024), LeakyReLU(0.1),
                            Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
                            BatchNorm2d(1024), LeakyReLU(0.1))

        self.fc = self._create_fc_layers(split_size, num_boxes, num_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv_block6(x)
        x = flatten(x, start_dim=1)
        x = self.fc(x)
        return x

    def _create_fc_layers(self, grid_size, num_bbox, num_class):
        S, B, C = grid_size, num_bbox, num_class
        return Sequential(Flatten(), Linear(in_features=1024*S*S, out_features=4096),
                    LeakyReLU(0.1), Linear(in_features=4096, out_features=S*S*(C+B*5)))


##### Sanity check #####

# def test(in_channels=3, split_size=7, num_boxes=2, num_classes=30):
#     model_yolo = Yolov1(in_channels, split_size, num_boxes, num_classes)
#     X = torch.randn((2, 3, 448, 448))
#     #print(model)
#     print(model_yolo(X).shape)

# test()
