class Context_Explor(nn.Module):
    def __init__(self, dim):
        super(Context_Explor, self).__init__()
        self.input_channels = input_channels
        self.channels_single = int(input_channels / 4)
        self.p1_channel_chunk = ConvBlock(n*dim,dim)
        self.p2_channel_chunk = ConvBlock(n*dim,dim)
        self.p3_channel_chunk = ConvBlock(n*dim,dim)
        self.p4_channel_chunk = ConvBlock(n*dim,dim)
        self.p1 = ConvBlock(dim,dim)
        self.p2 = ConvBlock(dim,dim)
        self.p3 = ConvBlock(dim,dim)
        self.p4 = ConvBlock(dim,dim)
        self.fusion = ConvBlock(n*dim,dim)
    def forward(self, x):
        p1_input = self.p1_channel_chunk(x)
        p1 = self.p1(p1_input)
        p2_input = self.p2_channel_chunk(x) + p1
        p2 = self.p2(p2_input)
        p3_input = self.p3_channel_chunk(x) + p2
        p3 = self.p3(p3_input)
        p4_input = self.p4_channel_chunk(x) + p3
        p4 = self.p4(p4_input)
        ce = self.fusion(torch.cat((p1, p2, p3, p4), 1))
        return ce
class CFU(nn.Module):
    def __init__(self, dim, groups):
        super().__init__()
        self.expand_conv = ConvBlock(dim, dim*n, 1)
        self.interact = nn.ModuleDict()
        self.interact["0"] = ConvBlock(hidden_dim, 2 * hidden_dim)
        for group_id in range(1,groups - 1):
            self.interact[str(group_id)] = ConvBlock(2 * hidden_dim, 2 * hidden_dim)
        self.interact[str(num_groups - 1)] = ConvBlock(2 * hidden_dim, 1 * hidden_dim)
        self.fuse = ConvBlock(groups * dim, dim)
        self.fp = Context_Explor(dim)
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
