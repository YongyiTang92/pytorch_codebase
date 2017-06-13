import scipy.io as io
import torch
import torch.nn as nn
import numpy as np

"This script only tested on converting two-stream model from matconvnet to pytorch."


class matconv_convertor(object):
    def __init__(self, file_dir):
        self.net = io.loadmat(file_dir)['net']
        self.vars = self.net['vars'][0, 0]
        self.params = self.net['params'][0, 0]
        self.meta = self.net['meta'][0, 0]
        self.layers = self.net['layers'][0, 0]
        self.image_size = self.net['meta'][0][0]['normalization'][0][0]['imageSize'][0][0][0]
        self.mean_image = self.net['meta'][0][0]['normalization'][0][0]['averageImage'][0][0]

    def get_layer_seq(self):
        # Demo of extract layer type:
        layer_list = []
        for i in range(self.layers.shape[1]):
            current_layer_type = str(self.layers['type'][0, i][0]).split('.')[1]  # Conv/ReLU/Pooling/DropOut/Loss
            current_layer_name = str(self.layers['name'][0, i][0])
            current_layer_input = str(self.layers['inputs'][0, i][0])
            current_layer_output = str(self.layers['outputs'][0, i][0])
            current_layer_block = self.layers['block'][0, i][0]
            current_layer_params = self.layers['params'][0][i]

            layer = self.get_layer(current_layer_type, current_layer_block, current_layer_params)
            if layer is not None:
                layer_list.append(layer)

        layer_stack = nn.Sequential(*layer_list)
        return layer_stack

    def get_layer(self, layer_name, layer_block, layer_params):
        if layer_name == 'Conv':
            layer = self.get_conv(layer_block, layer_params)
        elif layer_name == 'ReLU':
            layer = self.get_ReLU()
        elif layer_name == 'Pooling':
            layer = self.get_pooling(layer_block)
        elif layer_name == 'DropOut':
            layer = self.get_DropOut(layer_block)
        elif layer_name == 'Loss':
            layer = None
        else:
            raise Exception('layer' + layer_name + 'not exist')
        return layer

    def get_conv(self, layer_block, layer_params=None):
        """
        Input:
            layer_block: Contain the detail info of this layer.
            layer_params: Params of this layer if is not None.
        Output:
            pytorch conv layer
        """
        in_channel = int(layer_block['size'][0][0, 2])
        out_channel = int(layer_block['size'][0][0, 3])
        kernel_size = (int(layer_block['size'][0][0, 0]), int(layer_block['size'][0][0, 1]))
        stride = (int(layer_block['stride'][0][0, 0]), int(layer_block['stride'][0][0, 1]))
        bias = bool(layer_block['hasBias'][0][0, 0])
        padding = (int(layer_block['pad'][0][0, 0]), int(layer_block['pad'][0][0, 3]))
        conv_layer = nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias)

        if layer_params is not None:
            # Copy the params from matconvnet layer to pytorch layer
            w_name = str(layer_params[0, 0][0])
            w_index = np.where(self.params['name'][0, :] == w_name)[0][0]
            w_params = self.params['value'][0, w_index]
            w_params = np.transpose(w_params, [3, 2, 0, 1])  # [w, h, in_channel, out_channel] -> [out_channel, in_channel, w, h]
            if bias:
                b_name = str(layer_params[0, 1][0])
                b_index = np.where(self.params['name'][0, :] == b_name)[0][0]
                b_params = self.params['value'][0, b_index]
                b_params = np.reshape(b_params, [-1])

            print ('w_params shape: ', w_params.shape)
            print ('b_params shape: ', b_params.shape)
            print ('conv weight shape', conv_layer.weight.data.size())
            print ('conv bias shape', conv_layer.bias.data.size())
            conv_layer.weight.data.copy_(torch.from_numpy(w_params))
            conv_layer.bias.data.copy_(torch.from_numpy(b_params))

        return conv_layer

    def get_pooling(self, layer_block):
        """
        Input:
            layer_block: Contain the detail info of this layer.
        Output:
            pytorch pooling layer
        """
        kernel_size = (int(layer_block['poolSize'][0][0, 0]), int(layer_block['poolSize'][0][0, 1]))
        stride = (int(layer_block['stride'][0][0, 0]), int(layer_block['stride'][0][0, 1]))
        if layer_block['method'][0][0] == 'max':
            pooling_layer = nn.MaxPool2d(kernel_size, stride)
        elif layer_block['method'][0][0] == 'mean':
            pooling_layer = nn.AvgPool2d(kernel_size, stride)
        else:
            raise Exception('Pooling method ' + layer_block['method'][0][0] + 'is not defined.')

        return pooling_layer

    def get_ReLU(self):
        """
        Output:
            pytorch relu layer
        """
        return nn.ReLU()

    def get_DropOut(self, layer_block):
        """
        Input:
            layer_block: Contain the detail info of this layer.
        Output:
            pytorch dropout layer
        """
        return nn.Dropout(float(layer_block['rate'][0][0]))


def main():
    file_dir = '/home/yongyi'
    file_name = 'ucf101-img-vgg16-split1.mat'
