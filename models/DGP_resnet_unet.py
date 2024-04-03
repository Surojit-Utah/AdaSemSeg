import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch
import numpy as np


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
            print("Freezing the batch normalizing layer!!!")
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def forward(self, x):
        B, N, _, H, W = x.size()
        x = x.reshape(B*N, 3, H, W)
        features = {}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features['f2'] = x
        x = self.maxpool(x)
        features['f4'] = x = self.layer1(x)
        features['f8'] = x = self.layer2(x)
        features['gp_input_16'] = x = self.layer3(x)
        features['gp_input_32'] = x = self.layer4(x)
        features['gp_input_16'] = features['gp_input_16'].view(B, N, -1, H//16, W//16)
        features['gp_input_32'] = features['gp_input_32'].view(B, N, -1, H//32, W//32)

        return features


class Mask_Encoder(nn.Module):

    def __init__(self, features=8, affine=True):
        super().__init__()

        #Encoder
        self.dconv_down1 = double_conv(1, features, affine=affine)
        self.dconv_down2 = double_conv(features, 2*features, affine=affine)
        self.dconv_down3 = double_conv(2*features, 4*features, affine=affine)
        self.dconv_down4 = double_conv(4*features, 8*features, affine=affine)
        self.dconv_down5 = double_conv(8*features, 8*features, affine=affine)
        self.dconv_down6 = double_conv(8*features, 8*features, affine=affine)

        self.downsample_1 = nn.Conv2d(features, features, 3, stride=2, padding=1)
        self.downsample_2 = nn.Conv2d(2*features, 2*features, 3, stride=2, padding=1)
        self.downsample_3 = nn.Conv2d(4*features, 4*features, 3, stride=2, padding=1)
        self.downsample_4 = nn.Conv2d(8*features, 8*features, 3, stride=2, padding=1)
        self.downsample_5 = nn.Conv2d(8*features, 8*features, 3, stride=2, padding=1)

    def forward(self, x):
        features = {}
        B, N, C, H, W = x.size()
        x = x.reshape(B*N, C, H, W)

        features['f1'] = conv1 = self.dconv_down1(x)        # (B*N, 8, H, W)
        x = self.downsample_1(conv1)                        # (B*N, 8, H//2, W//2)

        features['f2'] = conv2 = self.dconv_down2(x)        # (B*N, 16, H//2, W//2)
        x = self.downsample_2(conv2)                        # (B*N, 16, H//4, W//4)

        features['f4'] = conv3 = self.dconv_down3(x)        # (B*N, 32, H//4, W//4)
        x = self.downsample_3(conv3)                        # (B*N, 32, H//8, W//8)

        features['f8'] = conv4 = self.dconv_down4(x)        # (B*N, 64, H//8, W//8)
        x = self.downsample_4(conv4)                        # (B*N, 64, H//16, W//16)

        conv5 = self.dconv_down5(x)                         # (B*N, 64, H//16, W//16)
        features['gp_input_16'] = conv5.view(B, N, -1, H//16, W//16)
        x = self.downsample_5(conv5)                        # (B*N, 64, H//32, W//32)

        x = self.dconv_down6(x)                             # (B*N, 64, H//32, W//32)
        features['gp_input_32'] = x.view(B, N, -1, H//32, W//32)

        return features


class FSS_Decoder(nn.Module):

    def __init__(self, n_class=2, image_features=64, mask_features=8, covar_size=5, sigmoid=False, affine=True):
        super().__init__()

        # Instance of the Image Encoder class
        # For the skip connection in reconstructing images
        self.n_class = n_class
        self.image_features = image_features
        self.mask_features = mask_features
        self.covar_size = covar_size

        self.gp_output_dconv1 = double_conv((self.mask_features*8 + self.covar_size**2), self.mask_features*8, affine=affine)         # (B, 2048, H//32, H//32)
        self.gp_output_dconv2 = double_conv((self.mask_features*8 + self.covar_size**2), self.mask_features*8, affine=affine)         # (B, 1024, H//32, H//32)
        self.dconv_up2 = double_conv((self.mask_features*8 + self.mask_features*8), self.mask_features*8, affine=affine)          # (B, 1024, H//16, H//16)
        self.dconv_up3 = double_conv((self.image_features*8 + self.mask_features*8), self.mask_features*8, affine=affine)             # (B, 512, H//8, H//8)
        self.dconv_up4 = double_conv((self.image_features*4 + self.mask_features*8), self.mask_features*8, affine=affine)             # (B, 256, H//4, H//4)
        self.dconv_up5 = double_conv((self.image_features*1 + self.mask_features*4), self.mask_features*4, affine=affine)      # (B, 32, H//2, H//2)

        self.upsample1 = nn.ConvTranspose2d(self.mask_features*8, self.mask_features*8, kernel_size=2, stride=2)              # (B, 1024, H//16, H//16)
        self.upsample2 = nn.ConvTranspose2d(self.mask_features*8, self.mask_features*8, kernel_size=2, stride=2)               # (B, 512, H//8, H//8)
        self.upsample3 = nn.ConvTranspose2d(self.mask_features*8, self.mask_features*8, kernel_size=2, stride=2)                # (B, 256, H//4, H//4)
        self.upsample4 = nn.ConvTranspose2d(self.mask_features*8, self.mask_features*4, kernel_size=2, stride=2)                # (B, 64, H//2, H//2)
        self.upsample5 = nn.ConvTranspose2d(self.mask_features*4, self.mask_features*4, kernel_size=2, stride=2)  # (B, 32, H//1, H//1)

        self.conv_last = nn.Conv2d(self.mask_features*4, self.n_class, kernel_size=1)                                     # (B, 32, H//1, H//1)

        self.add_sigmoid = sigmoid

    def forward(self, predicted_segmentation_encodings, encoded_image_features):

        # Convolving the output GP
        gp_output_32 = predicted_segmentation_encodings['gp_output_32']
        B, M, C, H, W = gp_output_32.size()
        gp_output_32 = gp_output_32.reshape(B*M, C, H, W)
        gp_output_32 = self.gp_output_dconv1(gp_output_32)              # (B*M, 2048, H//32, H//32)

        x = self.upsample1(gp_output_32)                                # (B*M, 1024, H//16, H//16)
        # Convolving the output GP
        gp_output_16 = predicted_segmentation_encodings['gp_output_16']
        B, M, C, H, W = gp_output_16.size()
        gp_output_16 = gp_output_16.reshape(B*M, C, H, W)
        gp_output_16 = self.gp_output_dconv2(gp_output_16)
        x = torch.cat([x, gp_output_16], dim=1)                         # (B*M, 1024+1024, H//16, H//16)
        x = self.dconv_up2(x)                                           # (B*M, 1024, H//16, H//16)

        x = self.upsample2(x)                                           # (B*M, 512, H//8, H//8)
        x = torch.cat([x, encoded_image_features['f8']], dim=1)         # (B*M, 512+512, H//8, H//8)
        x = self.dconv_up3(x)                                           # (B*M, 512, H//8, H//8)

        x = self.upsample3(x)                                           # (B*M, 256, H//4, H//4)
        x = torch.cat([x, encoded_image_features['f4']], dim=1)         # (B*M, 256+256, H//4, H//4)
        x = self.dconv_up4(x)                                           # (B*M, 256, H//4, H//4)

        x = self.upsample4(x)                                           # (B*M, 64, H//2, H//2)
        x = torch.cat([x, encoded_image_features['f2']], dim=1)         # (B*M, 64+64, H//2, H//2)
        x = self.dconv_up5(x)                                           # (B*M, 32, H//2, H//2)

        x = self.upsample5(x)                                           # (B*M, 32, H//1, H//1)

        x = self.conv_last(x)                                           # (B*M, 01, H//1, H//1)

        if self.add_sigmoid:
            out = nn.Sigmoid()(x)
        else:
            out = x

        _, C, H, W = out.size()
        out = out.view(B, M, C, H, W)

        return out


class DGPModel(nn.Module):

    def __init__(self, kernel, covariance_output_mode='none',
                 covar_size=5, sigma_noise=1e-1):
        super().__init__()

        self.kernel = kernel
        self.sigma_noise = sigma_noise
        self.covariance_output_mode = covariance_output_mode
        self.covar_size = covar_size

        assert covariance_output_mode in ['none', 'concatenate variance']
        if covariance_output_mode == 'concatenate variance':
            assert covar_size > 0, 'Covar neighbourhood size must be larger than 0 if using concatenate variance'

    def _sample_points_strided_grid(self, enc_images, enc_segmentations):
        """
        Reshape the input and its label for GP regression
        """
        B, N, C, H, W = enc_segmentations.size()
        B, N, D, H, W = enc_images.size()

        x_s = enc_images.permute(0, 1, 3, 4, 2).reshape(B, N*H*W, D)
        y_s = enc_segmentations.permute(0, 1, 3, 4, 2).reshape(B, N*H*W, C)

        return x_s, y_s

    def learn(self, enc_images, enc_segmentations):
        """
        Learnd the GP regression between the encoded support images and its encoded counterparts
        """

        # print("DGP Model before reshape contiguous check....")
        # print(enc_images.is_contiguous())
        # print(enc_segmentations.is_contiguous())

        x_s, y_s = self._sample_points_strided_grid(enc_images, enc_segmentations)
        # x_s = x_s.contiguous()
        # y_s = y_s.contiguous()

        # print("DGP Model contiguous check....")
        # print(x_s.is_contiguous())
        # print(y_s.is_contiguous())

        B, S, _ = x_s.size()

        sigma_noise = self.sigma_noise * torch.eye(S, device=x_s.device)[None, :, :]
        K_ss = self.kernel(x_s, x_s) + sigma_noise                                              # Kernel matrix between the encoded support images
        L = torch.linalg.cholesky(K_ss)                                                         # Cholesky decomposition
        alpha = self.tri_solve(L.permute(0, 2, 1), self.tri_solve(L, y_s), lower=False)         # solution for : ((K_ss + (sigma^2)I)^-1)y_s

        return L, alpha, x_s

    def tri_solve(self, L, b, lower=True):
        return torch.triangular_solve(b, L, upper=not lower)[0]

    def _get_covar_neighbours(self, v_q):
        """ Converts covariance of the form (B,H,W,H,W) to (B,H,W,K**2) where K is the covariance in a local neighbourhood around each point
        and M*M = Q
        """
        K = self.covar_size
        B, H, W, H, W = v_q.shape
        v_q = F.pad(v_q, 4 * (K // 2,))  # pad v_q
        delta = torch.stack(torch.meshgrid(torch.arange(-(K // 2), K // 2 + 1), torch.arange(-(K // 2), K // 2 + 1)),
                            dim=-1)
        positions = torch.stack(torch.meshgrid(torch.arange(K // 2, H + K // 2), torch.arange(K // 2, W + K // 2)),
                                dim=-1)
        neighbours = positions[:, :, None, None, :] + delta[None, :, :]
        points = torch.arange(H * W)[:, None].expand(H * W, K ** 2)
        v_q_neigbours = v_q.reshape(B, H * W, H + K - 1, W + K - 1)[:, points.flatten(), neighbours[..., 0].flatten(),
                        neighbours[..., 1].flatten()].reshape(B, H, W, K ** 2)
        return v_q_neigbours

    def _get_covariance(self, x_q, K_qs, L):
        """
        Returns:
            torch.Tensor (B, Q, Q)
        """
        B, Q, S = K_qs.size()
        K_qq = self.kernel(x_q, x_q)  # B, Q, Q
        v = self.tri_solve(L, K_qs.permute(0, 2, 1))
        v_q = K_qq - torch.einsum('bsq,bsk->bqk', v, v)
        return v_q

    def forward(self, encoded_images, online_model):
        """
        Get the mean and co-variance using the posterior analysis of GP
        """

        B, M, D, H, W = encoded_images.size()
        x_q = encoded_images.permute(0, 1, 3, 4, 2).reshape(B*M, H*W, D)  # B*M go together because we don't want covariance between different query images
        L, alpha, x_s = online_model

        K_qs = self.kernel(x_q, x_s)
        f_q = K_qs @ alpha
        B, Q, C = f_q.shape

        if self.covariance_output_mode == 'concatenate variance':
            v_q = self._get_covariance(x_q, K_qs, L)
            v_q = v_q.reshape(B * M, H, W, H, W)
            v_q_neighbours = self._get_covar_neighbours(v_q)
            out = torch.cat([f_q, v_q_neighbours.view(B, Q, self.covar_size ** 2)], dim=2)
            out = out.reshape(B, M, H, W, C+self.covar_size**2).permute(0, 1, 4, 2, 3).reshape(B, M, (C + self.covar_size**2), H, W)
        else:
            f_q = f_q.reshape(B, M, H, W, C).permute(0, 1, 4, 2, 3).reshape(B, M, C, H, W)
            out = f_q

        return out

    def __str__(self):
        return f"{str(self.kernel)}"


class FSSLearner(nn.Module):
    def __init__(self, image_encoder, anno_encoder, dgp_model, upsampler):
        super().__init__()
        self.image_encoder = image_encoder
        self.anno_encoder = anno_encoder
        self.dgp_model = dgp_model
        self.upsampler = upsampler

    def learn(self, images, segmaps):
        """
        Args:
            images (Tensor(B N 1 H W)):
            segmaps (LongTensor(B N C H W))
        Returns:
            online_model
        """
        # print("Few shot segmentation learn....")

        B, N, _, H, W = images.size()
        encoded_images_features = self.image_encoder(images)         # Shape : [B, N, D, H, W]

        encoded_segmentations_features = self.anno_encoder(segmaps)  # Shape : [B, N, C, H, W]

        online_models = dict()
        for feature_size in [16, 32]:
            input_key = 'gp_input_'+str(feature_size)
            output_key = 'gp_output_'+str(feature_size)
            online_models[output_key] = self.dgp_model.learn(encoded_images_features[input_key],
                                                      encoded_segmentations_features[input_key])

        return online_models

    def forward(self, images, online_models):
        """
        Args:
            images (Tensor(B N 1 H W)):
            online_models:
        Returns:
            Tensor(B, N, C, H, W)
        """
        encoded_images_features = self.image_encoder(images)  # (B, N, D, H, W)

        predicted_segmentation_encodings = dict()
        for key in online_models.keys():
            feature_size = key.split('gp_output_')[1]
            input_key = 'gp_input_'+str(feature_size)
            predicted_segmentation_encodings[key] = self.dgp_model(encoded_images_features[input_key],
                                                          online_models[key])

        segscores = self.upsampler(predicted_segmentation_encodings, encoded_images_features) # Shape : [B*M, 1, H, W], where M=1

        return segscores

    def __str__(self):
        return f"FSSlearner-{str(self.model)}"