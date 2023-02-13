import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from methods.module.base_model import BasicModelClass
from methods.module.conv_block import ConvBNReLU
from utils.builder import MODELS
from utils.ops import cus_sample
import tensorly as tl

tl.set_backend('pytorch')

###############  Multi-scale features Process Module  ##################

class ASPP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ASPP, self).__init__()
        self.conv1 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
        self.conv2 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=2, padding=2)
        self.conv3 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=5, padding=5)
        self.conv4 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=7, padding=7)
        self.conv5 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
        self.fuse = ConvBNReLU(5 * out_dim, out_dim, 3, 1, 1)
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv5 = self.conv5(cus_sample(x.mean((2, 3), keepdim=True), mode="size", factors=x.size()[2:]))
        return self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))

class TransLayer(nn.Module):
    def __init__(self, out_c, last_module=ASPP):
        super().__init__()
        self.c5_down = nn.Sequential(
            # ConvBNReLU(2048, 256, 3, 1, 1),
            last_module(in_dim=2048, out_dim=out_c),
        )
        self.c4_down = nn.Sequential(ConvBNReLU(1024, out_c, 3, 1, 1))
        self.c3_down = nn.Sequential(ConvBNReLU(512, out_c, 3, 1, 1))
        self.c2_down = nn.Sequential(ConvBNReLU(256, out_c, 3, 1, 1))
        self.c1_down = nn.Sequential(ConvBNReLU(64, out_c, 3, 1, 1))

    def forward(self, xs):
        assert isinstance(xs, (tuple, list))
        assert len(xs) == 5
        c1, c2, c3, c4, c5 = xs
        c5 = self.c5_down(c5)
        c4 = self.c4_down(c4)
        c3 = self.c3_down(c3)
        c2 = self.c2_down(c2)
        c1 = self.c1_down(c1)
        return c5, c4, c3, c2, c1
    
###############  Cross-View Attention Module  ##################

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CAMV(nn.Module):
    def __init__(self, in_dim, mm_size):
        super().__init__()
        self.conv_l_pre_down = ConvBNReLU(in_dim, in_dim, 5, stride=1, padding=2)
        self.conv_l_post_down = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_m = nn.Sequential(
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
        )
        self.conv_s_pre_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s_post_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.trans = nn.Sequential(
            ConvBNReLU(3 * in_dim, 3 * in_dim, 1),
            ConvBNReLU(3 * in_dim, in_dim, 3, 1, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
        )
        self.trans1 = nn.Sequential(
            ConvBNReLU(3 * in_dim, 3 * in_dim, 1),
            ConvBNReLU(3 * in_dim, in_dim, 3, 1, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
        )
       
        self.transa1 = nn.Sequential(
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
        )
        self.transa2 = nn.Sequential(
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
        )
        self.mm_size = mm_size
        self.coe_c_c1 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
        self.coe_h_c1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        self.coe_w_c1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
        self.coe_c_md = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
        self.coe_h_md = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        self.coe_w_md = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
        self.coe_c_c2 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
        self.coe_h_c2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        self.coe_w_c2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
        self.coe_c_c1.data.uniform_(-0.5,0.5)
        self.coe_h_c1.data.uniform_(-0.5,0.5)
        self.coe_w_c1.data.uniform_(-0.5,0.5)
        
        self.coe_c_md.data.uniform_(-0.5,0.5)
        self.coe_h_md.data.uniform_(-0.5,0.5)
        self.coe_w_md.data.uniform_(-0.5,0.5)
        
        self.coe_c_c2.data.uniform_(-0.5,0.5)
        self.coe_h_c2.data.uniform_(-0.5,0.5)
        self.coe_w_c2.data.uniform_(-0.5,0.5)
        
        self.coe_c_a1 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
        self.coe_h_a1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        self.coe_w_a1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
        self.coe_c_ma = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
        self.coe_h_ma = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        self.coe_w_ma = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
        self.coe_c_a2 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
        self.coe_h_a2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        self.coe_w_a2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
        self.coe_c_a1.data.uniform_(-0.5,0.5)
        self.coe_h_a1.data.uniform_(-0.5,0.5)
        self.coe_w_a1.data.uniform_(-0.5,0.5)
        
        self.coe_c_ma.data.uniform_(-0.5,0.5)
        self.coe_h_ma.data.uniform_(-0.5,0.5)
        self.coe_w_ma.data.uniform_(-0.5,0.5)
        
        self.coe_c_a2.data.uniform_(-0.5,0.5)
        self.coe_h_a2.data.uniform_(-0.5,0.5)
        self.coe_w_a2.data.uniform_(-0.5,0.5)
        self.channel_attn = ChannelAttention(64)
        self.spatial_attn = SpatialAttention()
        self.fuse = nn.Sequential(ConvBNReLU(128, 128, 1),ConvBNReLU(128, 64, 3,1,1),ConvBNReLU(64, 64, 3,1,1))
    def forward(self, c1, o, c2, a1, a2, return_feats=False):
        tgt_size = o.shape[2:]
        c1 = self.conv_l_pre_down(c1)
        c1 = F.adaptive_max_pool2d(c1, tgt_size) + F.adaptive_avg_pool2d(c1, tgt_size)
        c1 = self.conv_l_post_down(c1)
        m = self.conv_m(o)
        c2 = self.conv_s_pre_up(c2)
        c2 = F.adaptive_max_pool2d(c2, tgt_size) + F.adaptive_avg_pool2d(c2, tgt_size)
        c2 = self.conv_s_post_up(c2)
        attn = self.trans(torch.cat([c1, m, c2], dim=1))
        attn_c1 = tl.tenalg.mode_dot(attn,self.coe_c_c1,mode=1)
        attn_c1 = tl.tenalg.mode_dot(attn_c1,self.coe_h_c1,mode=2)
        attn_c1 = tl.tenalg.mode_dot(attn_c1,self.coe_w_c1,mode=3)
        attn_c1 = torch.softmax(attn_c1, dim=1)
        
        attn_md = tl.tenalg.mode_dot(attn,self.coe_c_md,mode=1)
        attn_md = tl.tenalg.mode_dot(attn_md,self.coe_h_md,mode=2)
        attn_md = tl.tenalg.mode_dot(attn_md,self.coe_w_md,mode=3)
        attn_md = torch.softmax(attn_md, dim=1)
        
        attn_c2 = tl.tenalg.mode_dot(attn,self.coe_c_c2,mode=1)
        attn_c2 = tl.tenalg.mode_dot(attn_c2,self.coe_h_c2,mode=2)
        attn_c2 = tl.tenalg.mode_dot(attn_c2,self.coe_w_c2,mode=3)
        attn_c2 = torch.softmax(attn_c2, dim=1)
        
        cmc = attn_c1 * c1 + attn_md * m + attn_c2 * c2

        a1 = self.transa1(a1)
        a2 = self.transa2(a2)
        attn1 = self.trans1(torch.cat([a1, m, a2], dim=1))
        
        attn_a1 = tl.tenalg.mode_dot(attn1,self.coe_c_a1,mode=1)
        attn_a1 = tl.tenalg.mode_dot(attn_a1,self.coe_h_a1,mode=2)
        attn_a1 = tl.tenalg.mode_dot(attn_a1,self.coe_w_a1,mode=3)
        attn_a1 = torch.softmax(attn_a1, dim=1)
        
        attn_ma = tl.tenalg.mode_dot(attn1,self.coe_c_ma,mode=1)
        attn_ma = tl.tenalg.mode_dot(attn_ma,self.coe_h_ma,mode=2)
        attn_ma = tl.tenalg.mode_dot(attn_ma,self.coe_w_ma,mode=3)
        attn_ma = torch.softmax(attn_ma, dim=1)
        
        attn_a2 = tl.tenalg.mode_dot(attn1,self.coe_c_a2,mode=1)
        attn_a2 = tl.tenalg.mode_dot(attn_a2,self.coe_h_a2,mode=2)
        attn_a2 = tl.tenalg.mode_dot(attn_a2,self.coe_w_a2,mode=3)
        attn_a2 = torch.softmax(attn_a2, dim=1)
        
        ama = attn_a1 * a1 + attn_ma * m + attn_a2 * a2
        ama = ama.mul(self.channel_attn(ama))
        ama = ama.mul(self.spatial_attn(ama))
        lms = self.fuse(torch.cat([ama,cmc],dim=1))
        return lms
class Progressive_Iteration(nn.Module):
    def __init__(self, input_channels):
        super(Progressive_Iteration, self).__init__()
        self.input_channels = input_channels
        self.channels_single = int(input_channels / 4)
        self.p1_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p1 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p1_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p2 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p3 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 5, 1, 2),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p4 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 7, 1, 3),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.fusion = nn.Sequential(nn.Conv2d(self.input_channels, self.input_channels, 1, 1, 0),
                                    nn.BatchNorm2d(self.input_channels), nn.ReLU())

    def forward(self, x):
        p1_input = self.p1_channel_reduction(x)
        p1 = self.p1(p1_input)
        p1_dc = self.p1_dc(p1)
        p2_input = self.p2_channel_reduction(x) + p1_dc
        p2 = self.p2(p2_input)
        p2_dc = self.p2_dc(p2)
        p3_input = self.p3_channel_reduction(x) + p2_dc
        p3 = self.p3(p3_input)
        p3_dc = self.p3_dc(p3)

        p4_input = self.p4_channel_reduction(x) + p3_dc
        p4 = self.p4(p4_input)
        p4_dc = self.p4_dc(p4)

        ce = self.fusion(torch.cat((p1_dc, p2_dc, p3_dc, p4_dc), 1))
        return ce

class CFU(nn.Module):
    def __init__(self, in_c, num_groups=4, hidden_dim=None):
        super().__init__()
        self.num_groups = num_groups
        hidden_dim = hidden_dim or in_c // 2
        expand_dim = hidden_dim * num_groups
        self.expand_conv = ConvBNReLU(in_c, expand_dim, 1)
        self.interact = nn.ModuleDict()
        self.interact["0"] = ConvBNReLU(hidden_dim, 2 * hidden_dim, 3, 1, 1)
        for group_id in range(1, num_groups - 1):
            self.interact[str(group_id)] = ConvBNReLU(2 * hidden_dim, 2 * hidden_dim, 3, 1, 1)
        self.interact[str(num_groups - 1)] = ConvBNReLU(2 * hidden_dim, 1 * hidden_dim, 3, 1, 1)
        self.fuse = nn.Sequential(nn.Conv2d(num_groups * hidden_dim, in_c, 3, 1, 1), nn.BatchNorm2d(in_c))
        self.final_relu = nn.ReLU(True)
        self.fp = Progressive_Iteration(192)

    def forward(self, x):
        xs = self.expand_conv(x).chunk(self.num_groups, dim=1)
        outs = []
        branch_out = self.interact["0"](xs[0])
        outs.append(branch_out.chunk(2, dim=1))

        for group_id in range(1, self.num_groups - 1):
            branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
            outs.append(branch_out.chunk(2, dim=1))

        group_id = self.num_groups - 1
        branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
        outs.append(branch_out.chunk(1, dim=1))
        out = torch.cat([o[0] for o in outs], dim=1)
        out = self.fp(out)
        out = self.fuse(out)
        return self.final_relu(out + x)

def get_coef(iter_percentage, method):
    if method == "linear":
        milestones = (0.3, 0.7)
        coef_range = (0, 1)
        min_point, max_point = min(milestones), max(milestones)
        min_coef, max_coef = min(coef_range), max(coef_range)
        if iter_percentage < min_point:
            ual_coef = min_coef
        elif iter_percentage > max_point:
            ual_coef = max_coef
        else:
            ratio = (max_coef - min_coef) / (max_point - min_point)
            ual_coef = ratio * (iter_percentage - min_point)
    elif method == "cos":
        coef_range = (0, 1)
        min_coef, max_coef = min(coef_range), max(coef_range)
        normalized_coef = (1 - np.cos(iter_percentage * np.pi)) / 2
        ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
    else:
        ual_coef = 1.0
    return ual_coef


def cal_ual(seg_logits, seg_gts):
    assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
    sigmoid_x = seg_logits.sigmoid()
    loss_map = 1 - (2 * sigmoid_x - 1).abs().pow(2)
    return loss_map.mean()


@MODELS.register()
class MFFN(BasicModelClass):
    def __init__(self):
        super().__init__()
        self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)
        self.translayer = TransLayer(out_c=64)  # [c5, c4, c3, c2, c1]
        dim = [64, 64, 64, 64, 64]
        size = [12, 24, 48, 96, 192]
        self.CAMV_layers = nn.ModuleList([CAMV(in_dim=in_c, mm_size=mm_s) for in_c, mm_s in zip(dim, size)])
        self.d5 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
        self.d4 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
        self.d3 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
        self.d2 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
        self.d1 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
        self.out_layer_00 = ConvBNReLU(64, 32, 3, 1, 1)
        self.out_layer_01 = nn.Conv2d(32, 1, 1)

    def encoder_translayer(self, x):
        en_feats = self.shared_encoder(x)
        trans_feats = self.translayer(en_feats)
        return trans_feats
    def body(self, c1_scale, o_scale, c2_scale, a1_scale, a2_scale):
        c1_trans_feats = self.encoder_translayer(c1_scale)
        o_trans_feats = self.encoder_translayer(o_scale)
        c2_trans_feats = self.encoder_translayer(c2_scale)
        a1_trans_feats = self.encoder_translayer(a1_scale)
        a2_trans_feats = self.encoder_translayer(a2_scale)
        feats = []
        for c1, o,c2,a1,a2, layer in zip(c1_trans_feats, o_trans_feats, c2_trans_feats, a1_trans_feats, a2_trans_feats, self.CAMV_layers):
            CAMV_outs = layer(c1=c1, o=o, c2=c2, a1=a1, a2=a2)
            feats.append(CAMV_outs)

        x = self.d5(feats[0])
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d4(x + feats[1])
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d3(x + feats[2])
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d2(x + feats[3])
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d1(x + feats[4])
        x = cus_sample(x, mode="scale", factors=2)
        logits = self.out_layer_01(self.out_layer_00(x))
        return dict(seg=logits)
    def train_forward(self, data, **kwargs):
        assert not {"image_c1", "image_o", "image_c2", "image_a1", "image_a2", "mask"}.difference(set(data)), set(data)
        output = self.body(
            c1_scale=data["image_c1"],
            o_scale=data["image_o"],
            c2_scale=data["image_c2"],
            a1_scale=data["image_a1"],
            a2_scale=data["image_a2"],
        )
        loss, loss_str = self.cal_loss(
            all_preds=output,
            gts=data["mask"],
            iter_percentage=kwargs["curr"]["iter_percentage"],
        )
        return dict(sal=output["seg"].sigmoid()), loss, loss_str

    def test_forward(self, data, **kwargs):
        output = self.body(
            l_scale=data["image_c1"],
            o_scale=data["image_o"],
            s_scale=data["image_c2"],
            a1_scale=data["image_a1"],
            a2_scale=data["image_a2"],
        )
        return output["seg"]

    def cal_loss(self, all_preds: dict, gts: torch.Tensor, method="cos", iter_percentage: float = 0):
        ual_coef = get_coef(iter_percentage, method)
        losses = []
        loss_str = []
        # for main
        for name, preds in all_preds.items():
            resized_gts = cus_sample(gts, mode="size", factors=preds.shape[2:])
            sod_loss = F.binary_cross_entropy_with_logits(input=preds, target=resized_gts, reduction="mean")
            losses.append(sod_loss)
            loss_str.append(f"{name}_BCE: {sod_loss.item():.5f}")
            ual_loss = cal_ual(seg_logits=preds, seg_gts=resized_gts)
            ual_loss *= ual_coef
            losses.append(ual_loss)
            loss_str.append(f"{name}_UAL_{ual_coef:.5f}: {ual_loss.item():.5f}")
        return sum(losses), " ".join(loss_str)

    def get_grouped_params(self):
        param_groups = {}
        for name, param in self.named_parameters():
            if name.startswith("shared_encoder.layer"):
                param_groups.setdefault("pretrained", []).append(param)
            elif name.startswith("shared_encoder."):
                param_groups.setdefault("fixed", []).append(param)
            else:
                param_groups.setdefault("retrained", []).append(param)
        return param_groups

