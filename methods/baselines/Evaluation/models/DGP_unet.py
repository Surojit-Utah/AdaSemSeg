import torch.nn as nn
import torch


def double_conv(in_channels, out_channels, affine):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels, affine=affine),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels, affine=affine),
        nn.ReLU(inplace=True))


class Image_Encoder(nn.Module):

    def __init__(self, features=32, affine=True):
        super().__init__()

        self.dconv_down1 = double_conv(3, features, affine=affine)
        self.dconv_down2 = double_conv(features, 2*features, affine=affine)
        self.dconv_down3 = double_conv(2*features, 4*features, affine=affine)
        self.dconv_down4 = double_conv(4*features, 8*features, affine=affine)
        self.dconv_down5 = double_conv(8*features, 16*features, affine=affine)
        self.dconv_down6 = double_conv(16*features, 32*features, affine=affine)

        self.downsample_1 = nn.Conv2d(features, features, 3, stride=2, padding=1)
        self.downsample_2 = nn.Conv2d(2*features, 2*features, 3, stride=2, padding=1)
        self.downsample_3 = nn.Conv2d(4*features, 4*features, 3, stride=2, padding=1)
        self.downsample_4 = nn.Conv2d(8*features, 8*features, 3, stride=2, padding=1)
        self.downsample_5 = nn.Conv2d(16*features, 16*features, 3, stride=2, padding=1)

        self.last_conv = nn.Conv2d(32*features, 16*features, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        features = {}

        features['f1'] = conv1 = self.dconv_down1(x)        # (B, 32, H, W)
        x = self.downsample_1(conv1)                        # (B, 32, H//2, W//2)

        features['f2'] = conv2 = self.dconv_down2(x)        # (B, 64, H//2, W//2)
        x = self.downsample_2(conv2)                        # (B, 64, H//4, W//4)

        features['f3'] = conv3 = self.dconv_down3(x)        # (B, 128, H//4, W//4)
        x = self.downsample_3(conv3)                        # (B, 128, H//8, W//8)

        features['f4'] = conv4 = self.dconv_down4(x)        # (B, 256, H//8, W//8)
        x = self.downsample_4(conv4)                        # (B, 256, H//16, W//16)

        features['f5'] = conv5 = self.dconv_down5(x)        # (B, 512, H//16, W//16)
        x = self.downsample_5(conv5)                        # (B, 512, H//32, W//32)

        features['f6'] = self.dconv_down6(x)                # (B, 1024, H//32, W//32)

        return features


class Image_Decoder(nn.Module):

    def __init__(self, n_class=1, image_features=32, mask_features=8, covar_size=5, sigmoid=False, affine=True):
        super().__init__()

        # Instance of the Image Encoder class
        # For the skip connection in reconstructing images
        self.n_class = n_class
        self.image_features = image_features
        self.mask_features = mask_features
        self.covar_size = covar_size

        self.dconv_up1 = double_conv((self.image_features*16 + self.image_features*16), self.image_features*16, affine=affine)    # (B, 64, H//16, H//16)
        self.dconv_up2 = double_conv((self.image_features*8 + self.image_features*8), self.image_features*8, affine=affine)       # (B, 64, H//8, H//8)
        self.dconv_up3 = double_conv((self.image_features*4 + self.image_features*4), self.image_features*4, affine=affine)       # (B, 32, H//4, H//4)
        self.dconv_up4 = double_conv((self.image_features*2 + self.image_features*2), self.image_features*2, affine=affine)       # (B, 16, H//2, H//2)
        self.dconv_up5 = double_conv((self.image_features*1 + self.image_features*1), self.image_features*1, affine=affine)       # (B, 08, H//1, H//1)

        self.upsample1 = nn.ConvTranspose2d(self.image_features*32, self.image_features*16, kernel_size=2, stride=2)              # (B, 64, H//16, H//16)
        self.upsample2 = nn.ConvTranspose2d(self.image_features*16, self.image_features*8, kernel_size=2, stride=2)               # (B, 64, H//8, H//8)
        self.upsample3 = nn.ConvTranspose2d(self.image_features*8, self.image_features*4, kernel_size=2, stride=2)                # (B, 32, H//4, H//4)
        self.upsample4 = nn.ConvTranspose2d(self.image_features*4, self.image_features*2, kernel_size=2, stride=2)                # (B, 16, H//2, H//2)
        self.upsample5 = nn.ConvTranspose2d(self.image_features*2, self.image_features*1, kernel_size=2, stride=2)                # (B, 8, H//1, H//1)

        self.conv_last = nn.Conv2d(self.image_features, self.n_class, kernel_size=1)                                              # (B, 1, H//1, H//1)

        self.add_sigmoid = sigmoid

    def forward(self, encoded_image_features):

        x = encoded_image_features['f6']                                # (B, 1024, H//32, W//32)

        x = self.upsample1(x)                                           # (B, 512, H//16, H//16)
        x = torch.cat([x, encoded_image_features['f5']], dim=1)         # (B, 512+512, H//16, H//16)
        x = self.dconv_up1(x)                                           # (B, 512, H//16, H//16)

        x = self.upsample2(x)                                           # (B, 256, H//8, H//8)
        x = torch.cat([x, encoded_image_features['f4']], dim=1)         # (B, 256+256, H//8, H//8)
        x = self.dconv_up2(x)                                           # (B, 256, H//8, H//8)

        x = self.upsample3(x)                                           # (B, 128, H//4, H//4)
        x = torch.cat([x, encoded_image_features['f3']], dim=1)         # (B, 128+128, H//4, H//4)
        x = self.dconv_up3(x)                                           # (B, 128, H//4, H//4)

        x = self.upsample4(x)                                           # (B, 64, H//2, H//2)
        x = torch.cat([x, encoded_image_features['f2']], dim=1)         # (B, 64+64, H//2, H//2)
        x = self.dconv_up4(x)                                           # (B, 64, H//2, H//2)

        x = self.upsample5(x)                                           # (B, 32, H//1, H//1)
        x = torch.cat([x, encoded_image_features['f1']], dim=1)         # (B, 32+32, H//1, H//1)
        x = self.dconv_up5(x)                                           # (B, 32, H//1, H//1)

        x = self.conv_last(x)                                           # (B, 01, H//1, H//1)

        if self.add_sigmoid:
            out = nn.Sigmoid()(x)
        else:
            out = x

        return out


class FSSLearner(nn.Module):
    def __init__(self, image_encoder, upsampler):
        super().__init__()
        self.image_encoder = image_encoder
        self.upsampler = upsampler

    def forward(self, images):
        """
        Args:
            images (Tensor(B N 1 H W)):
            online_models:
        Returns:
            Tensor(B, N, C, H, W)
        """
        encoded_images_features = self.image_encoder(images)    # [B, D, H, W]
        segscores = self.upsampler(encoded_images_features)     # [B, 1, H, W]

        return segscores

    def __str__(self):
        return f"FSSlearner-{str(self.model)}"