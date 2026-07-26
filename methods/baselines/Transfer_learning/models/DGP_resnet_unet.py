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

    def __init__(self, resnet, freeze_bn=False):
        super().__init__()
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.freeze_bn = freeze_bn

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bn:
            # print("Freezing the batch normalizing layer!!!")
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def forward(self, x):
        # B, N, _, H, W = x.size()
        # x = x.reshape(B*N, 3, H, W)
        features = {}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features['f2'] = x
        x = self.maxpool(x)
        features['f4'] = x = self.layer1(x)
        features['f8'] = x = self.layer2(x)
        features['f16'] = x = self.layer3(x)
        features['f32'] = x = self.layer4(x)

        return features


class Image_Decoder(nn.Module):

    def __init__(self, image_features=64, affine=True):
        super().__init__()

        self.image_features = image_features

        self.dconv_up1 = double_conv((self.image_features*16 + self.image_features*16), self.image_features*16, affine=affine)    # (B, 64, H//16, H//16)
        self.dconv_up2 = double_conv((self.image_features*8 + self.image_features*8), self.image_features*8, affine=affine)       # (B, 64, H//8, H//8)
        self.dconv_up3 = double_conv((self.image_features*4 + self.image_features*4), self.image_features*4, affine=affine)       # (B, 32, H//4, H//4)
        self.dconv_up4 = double_conv((self.image_features*2 + self.image_features*1), self.image_features*2, affine=affine)       # (B, 16, H//2, H//2)

        self.upsample1 = nn.ConvTranspose2d(self.image_features*32, self.image_features*16, kernel_size=2, stride=2)              # (B, 64, H//16, H//16)
        self.upsample2 = nn.ConvTranspose2d(self.image_features*16, self.image_features*8, kernel_size=2, stride=2)               # (B, 64, H//8, H//8)
        self.upsample3 = nn.ConvTranspose2d(self.image_features*8, self.image_features*4, kernel_size=2, stride=2)                # (B, 32, H//4, H//4)
        self.upsample4 = nn.ConvTranspose2d(self.image_features*4, self.image_features*2, kernel_size=2, stride=2)                # (B, 16, H//2, H//2)
        self.upsample5 = nn.ConvTranspose2d(self.image_features*2, self.image_features*1, kernel_size=2, stride=2)                # (B, 8, H//1, H//1)


    def forward(self, encoded_image_features):

        x = encoded_image_features['f32']                               # (B, 2048, H//32, W//32)

        x = self.upsample1(x)                                           # (B, 1024, H//16, H//16)
        x = torch.cat([x, encoded_image_features['f16']], dim=1)        # (B, 1024+1024, H//16, H//16)
        x = self.dconv_up1(x)                                           # (B, 1024, H//16, H//16)

        x = self.upsample2(x)                                           # (B, 512, H//8, H//8)
        x = torch.cat([x, encoded_image_features['f8']], dim=1)         # (B, 512+512, H//8, H//8)
        x = self.dconv_up2(x)                                           # (B, 512, H//8, H//8)

        x = self.upsample3(x)                                           # (B, 256, H//4, H//4)
        x = torch.cat([x, encoded_image_features['f4']], dim=1)         # (B, 256+256, H//4, H//4)
        x = self.dconv_up3(x)                                           # (B, 256, H//4, H//4)

        x = self.upsample4(x)                                           # (B, 128, H//2, H//2)
        x = torch.cat([x, encoded_image_features['f2']], dim=1)         # (B, 128+128, H//2, H//2)
        x = self.dconv_up4(x)                                           # (B, 128, H//2, H//2)

        out = self.upsample5(x)                                           # (B, 64, H//1, H//1)

        return out



class Segmentation_Network(nn.Module):
    def __init__(self, image_encoder, upsampler):
        super().__init__()
        self.class_count_dict = {'parihaka': 6, 'penobscot': 7, 'f3': 6}
        self.image_encoder = image_encoder
        self.shared_decoder = upsampler
        self.parihaka_classifier = nn.Conv2d(self.shared_decoder.image_features, self.class_count_dict['parihaka'], kernel_size=1)
        self.penobscot_classifier = nn.Conv2d(self.shared_decoder.image_features, self.class_count_dict['penobscot'], kernel_size=1)
        self.f3_classifier = nn.Conv2d(self.shared_decoder.image_features, self.class_count_dict['f3'], kernel_size=1)

        self.class_name = None

    def forward(self, images):
        """
        Args:
            images (Tensor(B N 1 H W)):
            online_models:
        Returns:
            Tensor(B, N, C, H, W)
        """
        encoded_images_features = self.image_encoder(images)                            # [B, D, H//32, W//32]
        shared_decoded_features = self.shared_decoder(encoded_images_features)          # [B, 64, H, W]
        if 'parihaka' in self.class_name:
            segscores = self.parihaka_classifier(shared_decoded_features)               # [B, 6, H, W]
        elif 'penobscot' in self.class_name:
            segscores = self.penobscot_classifier(shared_decoded_features)              # [B, 7, H, W]
        elif 'f3' in self.class_name:
            segscores = self.f3_classifier(shared_decoded_features)                     # [B, 6, H, W]

        self.class_name = None

        return segscores

    def __str__(self):
        return f"FSSlearner-{str(self.model)}"