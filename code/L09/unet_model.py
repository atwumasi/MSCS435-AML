import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderMiniBlock(nn.Module):
    """
    Two 3x3 convolutions (+ReLU, He-normal init) followed by BatchNorm, optional Dropout,
    and optional 2x2 max pooling.

    Returns (next_layer, skip_connection): `skip_connection` is the pre-pooling activation,
    used by the decoder; `next_layer` is what feeds the next encoder block.
    """

    def __init__(self, in_channels, n_filters=32, dropout_prob=0.3, max_pooling=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, n_filters, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding='same')
        self.bn = nn.BatchNorm2d(n_filters)
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else None
        self.pool = nn.MaxPool2d(kernel_size=2) if max_pooling else None
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        conv = F.relu(self.conv1(x))
        conv = F.relu(self.conv2(conv))

        # Batch Normalization will normalize the output of the last layer based on the batch's mean and standard deviation
        conv = self.bn(conv)

        # In case of overfitting, dropout will regularize the loss and gradient computation to shrink the influence of weights on output
        if self.dropout is not None:
            conv = self.dropout(conv)

        skip_connection = conv

        # Pooling reduces the size of the image while keeping the number of channels the same.
        # Pooling is optional since the last encoder layer does not use pooling.
        next_layer = self.pool(conv) if self.pool is not None else conv

        return next_layer, skip_connection


class DecoderMiniBlock(nn.Module):
    """
    Transpose convolution to upsample 2x, concatenation with the matching encoder skip
    connection, then two 3x3 convolutions (+ReLU). Mirrors the TensorFlow DecoderMiniBlock.
    """

    def __init__(self, in_channels, skip_channels, n_filters=32):
        super().__init__()
        # Transpose convolution that doubles the spatial size (equivalent to TF's
        # Conv2DTranspose(kernel=3, stride=2, padding='same') for even input sizes)
        self.up = nn.ConvTranspose2d(in_channels, n_filters, kernel_size=3,
                                      stride=2, padding=1, output_padding=1)
        self.conv1 = nn.Conv2d(n_filters + skip_channels, n_filters, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding='same')

        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, prev_layer_input, skip_layer_input):
        up = self.up(prev_layer_input)

        # Merge the skip connection from the encoder to prevent information loss
        merge = torch.cat([up, skip_layer_input], dim=1)

        conv = F.relu(self.conv1(merge))
        conv = F.relu(self.conv2(conv))
        return conv


class UNet(nn.Module):
    """
    Combines the encoder and decoder blocks per the U-Net paper.
    """

    def __init__(self, in_channels=3, n_filters=32, n_classes=3):
        super().__init__()

        # Encoder: filters increase as we go deeper, increasing the # channels of the feature map
        self.cblock1 = EncoderMiniBlock(in_channels, n_filters, dropout_prob=0, max_pooling=True)
        self.cblock2 = EncoderMiniBlock(n_filters, n_filters * 2, dropout_prob=0, max_pooling=True)
        self.cblock3 = EncoderMiniBlock(n_filters * 2, n_filters * 4, dropout_prob=0, max_pooling=True)
        self.cblock4 = EncoderMiniBlock(n_filters * 4, n_filters * 8, dropout_prob=0.3, max_pooling=True)
        self.cblock5 = EncoderMiniBlock(n_filters * 8, n_filters * 16, dropout_prob=0.3, max_pooling=False)

        # Decoder: filters decrease; skip connections come from the matching encoder block
        self.ublock6 = DecoderMiniBlock(n_filters * 16, n_filters * 8, n_filters * 8)
        self.ublock7 = DecoderMiniBlock(n_filters * 8, n_filters * 4, n_filters * 4)
        self.ublock8 = DecoderMiniBlock(n_filters * 4, n_filters * 2, n_filters * 2)
        self.ublock9 = DecoderMiniBlock(n_filters * 2, n_filters, n_filters)

        # One more 3x3 conv, then a 1x1 conv to map to `n_classes` output channels (raw logits)
        self.conv9 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding='same')
        self.conv10 = nn.Conv2d(n_filters, n_classes, kernel_size=1)
        nn.init.kaiming_normal_(self.conv9.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        next1, skip1 = self.cblock1(x)
        next2, skip2 = self.cblock2(next1)
        next3, skip3 = self.cblock3(next2)
        next4, skip4 = self.cblock4(next3)
        next5, _ = self.cblock5(next4)  # bottleneck has no pooling and no skip usage

        up6 = self.ublock6(next5, skip4)
        up7 = self.ublock7(up6, skip3)
        up8 = self.ublock8(up7, skip2)
        up9 = self.ublock9(up8, skip1)

        conv9 = F.relu(self.conv9(up9))
        logits = self.conv10(conv9)  # (N, n_classes, H, W) raw logits
        return logits
