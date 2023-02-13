
def channel_attn(self, x):
    avg_out = self.conv(relu(conv(avgpool(x))))
    max_out = self.conv(relu1(fc1(maxpool(x))))
    out = avg_out + max_out
    return sigmoid(out)
def spatial_attn(self, x):
    avg_out=torch.mean(x, dim=1, keepdim=True)
    max_out=torch.max(x, dim=1, keepdim=True)
    x = torch.cat([avg_out, max_out], dim=1)
    x = self.conv(x)
    return sigmoid(x)
class CAMV(nn.Module):
    def __init__(self, in_dim, mm_size):
        super().__init__()
        self.conv_f1 = ConvBlock(in_dim, in_dim)
        self.conv_p1 = ConvBlock(in_dim, in_dim)
        self.conv_m  = ConvBlock(in_dim, in_dim)
        self.conv_f2 = ConvBlock(in_dim, in_dim)
        self.conv_p2 = ConvBlock(in_dim, in_dim)
        self.trans_D = ConvBlock(3*in_dim, in_dim)
        self.transa1 = ConvBlock(in_dim, in_dim)
        self.transa2 = ConvBlock(in_dim, in_dim)
        self.trans_A = ConvBlock(3*in_dim, in_dim)
        self.D_A_Fusion = ConvBlock(2*in_dim, in_dim)
        self.atten_channel = ChannelAttention(dim)
        self.atten_spatial = SpatialAttention()
    def forward(self,c1,c2,o,a1,a2):
        h,w = o.shape[2:]
        # close view 1 --> down sampling
        c1 = self.conv_f1(c1)
        c1 = F.adaptive_max_pool2d(c1,(h,w))+F.adaptive_avg_pool2d(c1,(h,w))
        c1 = self.conv_p1(c1)
        # original view
        m = self.conv_m(o)
        # close view 2 --> down sampling
        c2 = self.conv_f2(c1)
        c2 = F.adaptive_max_pool2d(c2,(h,w))+F.adaptive_avg_pool2d(c2,(h,w))
        c2 = self.conv_p2(c2)
        # calculate attention factor
        attn_D = self.trans_D(torch.cat([c1, m, c2], dim=1))
        u1 = mode_dot(attn_D,Parameter1,mode=(1,2,3)).softmax()
        u2 = mode_dot(attn_D,Parameter2,mode=(1,2,3)).softmax()
        u3 = mode_dot(attn_D,Parameter3,mode=(1,2,3)).softmax()
        view_D = u1 * c1 + u2 * m + u3 * c2
        # process angel view
        a1 = self.transa1(a1)
        a2 = self.transa2(a2)
        # calculate attention factor
        attn_A = self.trans_A(torch.cat([a1, m, a2], dim=1))
        u4 = mode_dot(attn_A,Parameter4,mode=(1,2,3)).softmax()
        u5 = mode_dot(attn_A,Parameter5,mode=(1,2,3)).softmax()
        u6 = mode_dot(attn_A,Parameter6,mode=(1,2,3)).softmax()
        view_A = u4 * a1 + u5 * m + u6 * a2
        view_A = view_A.mul(self.atten_flow_channel_0(view_A))
        view_A = temp.mul(self.atten_flow_spatial_0(view_A))
        result = self.D_A_Fusion(torch.cat([view_A, view_D], dim=1))
        return lms
class Context_Exploration_Block(nn.Module):
    def __init__(self, input_channels):
        super(Context_Exploration_Block, self).__init__()
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


class HMU(nn.Module):
    def __init__(self, in_c, num_groups=4, hidden_dim=None):
        super().__init__()
        self.num_groups = num_groups
        hidden_dim = hidden_dim or in_c // 2
        expand_dim = hidden_dim * num_groups
        self.expand_conv = ConvBNReLU(in_c, expand_dim, 1)
        self.gate_genator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(num_groups * hidden_dim, hidden_dim, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, num_groups * hidden_dim, 1),
            nn.Softmax(dim=1),
        )
        self.interact = nn.ModuleDict()
        self.interact["0"] = ConvBNReLU(hidden_dim, 3 * hidden_dim, 3, 1, 1)
        for group_id in range(1, num_groups - 1):
            self.interact[str(group_id)] = ConvBNReLU(2 * hidden_dim, 3 * hidden_dim, 3, 1, 1)
        self.interact[str(num_groups - 1)] = ConvBNReLU(2 * hidden_dim, 2 * hidden_dim, 3, 1, 1)
        self.fuse = nn.Sequential(nn.Conv2d(num_groups * hidden_dim, in_c, 3, 1, 1), nn.BatchNorm2d(in_c))
        self.final_relu = nn.ReLU(True)

    def forward(self, x):
        xs = self.expand_conv(x).chunk(self.num_groups, dim=1)
        outs = []
        branch_out = self.interact["0"](xs[0])
        outs.append(branch_out.chunk(3, dim=1))

        for group_id in range(1, self.num_groups - 1):
            branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
            outs.append(branch_out.chunk(3, dim=1))

        group_id = self.num_groups - 1
        branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
        outs.append(branch_out.chunk(2, dim=1))
        out = torch.cat([o[0] for o in outs], dim=1)
        gate = self.gate_genator(torch.cat([o[-1] for o in outs], dim=1))
        out = self.fuse(out * gate)
        return self.final_relu(out + x)


class my_HMU(nn.Module):
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
        self.fp = Context_Exploration_Block(192)

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
class ZoomNet(BasicModelClass):
    def __init__(self):
        super().__init__()
        self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)
        self.translayer = TransLayer(out_c=64)  # [c5, c4, c3, c2, c1]
        dim = [64, 64, 64, 64, 64]
        size = [12, 24, 48, 96, 192]
        self.merge_layers = nn.ModuleList([SIU(in_dim=in_c, mm_size=mm_s) for in_c, mm_s in zip(dim, size)])

        self.d5 = nn.Sequential(my_HMU(64, num_groups=6, hidden_dim=32))
        self.d4 = nn.Sequential(my_HMU(64, num_groups=6, hidden_dim=32))
        self.d3 = nn.Sequential(my_HMU(64, num_groups=6, hidden_dim=32))
        self.d2 = nn.Sequential(my_HMU(64, num_groups=6, hidden_dim=32))
        self.d1 = nn.Sequential(my_HMU(64, num_groups=6, hidden_dim=32))
        self.out_layer_00 = ConvBNReLU(64, 32, 3, 1, 1)
        self.out_layer_01 = nn.Conv2d(32, 1, 1)

    def encoder_translayer(self, x):
        en_feats = self.shared_encoder(x)
        trans_feats = self.translayer(en_feats)
        return trans_feats
    def body(self, l_scale, o_scale, s_scale, a1_scale, a2_scale):
        l_trans_feats = self.encoder_translayer(l_scale)
        o_trans_feats = self.encoder_translayer(o_scale)
        s_trans_feats = self.encoder_translayer(s_scale)
        a1_trans_feats = self.encoder_translayer(a1_scale)
        a2_trans_feats = self.encoder_translayer(a2_scale)
        feats = []
        for l, o,s,a1,a2, layer in zip(l_trans_feats, o_trans_feats, s_trans_feats, a1_trans_feats, a2_trans_feats, self.merge_layers):
            siu_outs = layer(l=l, o=o, s=s, a1=a1, a2=a2)
            feats.append(siu_outs)

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
        assert not {"image1.5", "image1.0", "image0.5", "imagea.1", "imagea.2", "mask"}.difference(set(data)), set(data)
        output = self.body(
            l_scale=data["image1.5"],
            o_scale=data["image1.0"],
            s_scale=data["image0.5"],
            a1_scale=data["imagea.1"],
            a2_scale=data["imagea.2"],
        )
        loss, loss_str = self.cal_loss(
            all_preds=output,
            gts=data["mask"],
            iter_percentage=kwargs["curr"]["iter_percentage"],
        )
        return dict(sal=output["seg"].sigmoid()), loss, loss_str

    def test_forward(self, data, **kwargs):
        output = self.body(
            l_scale=data["image1.5"],
            o_scale=data["image1.0"],
            s_scale=data["image0.5"],
            a1_scale=data["imagea.1"],
            a2_scale=data["imagea.2"],
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
@MODELS.register()
class ZoomNet_CK(ZoomNet):
    def __init__(self):
        super().__init__()
        self.dummy = torch.ones(1, dtype=torch.float32, requires_grad=True)

    def encoder(self, x, dummy_arg=None):
        assert dummy_arg is not None
        x0, x1, x2, x3, x4 = self.shared_encoder(x)
        return x0, x1, x2, x3, x4

    def trans(self, x0, x1, x2, x3, x4):
        x5, x4, x3, x2, x1 = self.translayer([x0, x1, x2, x3, x4])
        return x5, x4, x3, x2, x1

    def decoder(self, x5, x4, x3, x2, x1):
        x = self.d5(x5)
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d4(x + x4)
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d3(x + x3)
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d2(x + x2)
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d1(x + x1)
        x = cus_sample(x, mode="scale", factors=2)
        logits = self.out_layer_01(self.out_layer_00(x))
        return logits

    def body(self, l_scale, m_scale, s_scale):
        l_trans_feats = checkpoint(self.encoder, l_scale, self.dummy)
        m_trans_feats = checkpoint(self.encoder, m_scale, self.dummy)
        s_trans_feats = checkpoint(self.encoder, s_scale, self.dummy)
        l_trans_feats = checkpoint(self.trans, *l_trans_feats)
        m_trans_feats = checkpoint(self.trans, *m_trans_feats)
        s_trans_feats = checkpoint(self.trans, *s_trans_feats)
        feats = []
        for layer_idx, (l, m, s) in enumerate(zip(l_trans_feats, m_trans_feats, s_trans_feats)):
            siu_outs = checkpoint(self.merge_layers[layer_idx], l, m, s)
            feats.append(siu_outs)
        logits = checkpoint(self.decoder, *feats)
        return dict(seg=logits)
