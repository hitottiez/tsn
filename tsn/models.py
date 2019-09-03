import torch
from ops.basic_ops import ConsensusModule, Identity
from torch import nn
from torch.nn.init import constant, normal
from transforms import *
from models import TSN


class TSNCustom(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8,
                 crop_num=1, partial_bn=True,
                 does_use_global_img=False):
        super(TSNCustom, self).__init__()
        self.num_segments = num_segments
        self.modality = modality
        self.new_length = new_length
        self.base_model = base_model
        self.consensus_type = consensus_type
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self._enable_pbn = partial_bn
        self.does_use_global_img = does_use_global_img

        print('\033[93muse global:{} \033[0m'.format(does_use_global_img))
        if does_use_global_img:
            self.tsn_for_global = PartialTSN(
                num_class=num_class, num_segments=num_segments,
                modality=modality, base_model=base_model,
                new_length=new_length, consensus_type=consensus_type,
                before_softmax=before_softmax, dropout=dropout,
                crop_num=crop_num, partial_bn=partial_bn
            )
            self.tsn_for_global.consensus = None

        self.tsn_for_local = PartialTSN(
            num_class=num_class, num_segments=num_segments,
            modality=modality, base_model=base_model,
            new_length=new_length, consensus_type=consensus_type,
            before_softmax=before_softmax, dropout=dropout,
            crop_num=crop_num, partial_bn=partial_bn
        )
        self.tsn_for_local.consensus = None

        self.consensus = ConsensusModule(consensus_type)
        self._prepare_newfc(num_class)

        if does_use_global_img:
            self.tsn_for_global.new_fc = None
        self.tsn_for_local.new_fc = None

    def _prepare_newfc(self, num_class):
        feature_dim = self.tsn_for_local.new_fc.in_features
        std = 0.001
        self.new_fc = nn.Linear(feature_dim, num_class)
        normal(self.new_fc.weight, 0, std)
        constant(self.new_fc.bias, 0)

    def forward(self, input):

        if self.does_use_global_img:
            global_tensor, local_tensor = torch.chunk(input, chunks=2, dim=1)
            global_tensor = global_tensor.contiguous()
            local_tensor = local_tensor.contiguous()

            # out: [batch_size * num_segments, 1024 features]
            global_base_out = self.tsn_for_global(global_tensor)
            local_base_out = self.tsn_for_local(local_tensor)

            # out: [batch_size * num_segments * 2, 1024 features]
            # variable order: global, global, ..., local, local, ....
            base_out = torch.cat([global_base_out, local_base_out], dim=0)

            # order to <(global num_segments, local num_segments) * batch_size>  about dim=0
            # variable order: global, local, global, local, ....
            base_out = self._reorder(base_out)

            # used for mcf-tracker transition cost model
            before_fc_output = base_out.view(
                (-1, self.num_segments* 2) + base_out.size()[1:])

            # out: [batch_size * num_segments * 2, num_class]
            out = self.new_fc(base_out)

            if not self.before_softmax:
                out = self.softmax(out)

            # reshape to [batch_size, num_segments x 2(global,local), action_class_num]
            out = out.view((-1, self.num_segments * 2) + out.size()[1:])

        else:
            base_out = self.tsn_for_local(input)
            before_fc_output = base_out.view(
                (-1, self.num_segments) + base_out.size()[1:])
            out = self.new_fc(base_out)
            if not self.before_softmax:
                out = self.softmax(out)
            out = out.view((-1, self.num_segments) + out.size()[1:])

        # out = [batch_size, action_class_num]
        out = self.consensus(out)
        return out.squeeze(1), before_fc_output.mean(dim=1)

    def _reorder(self, base_out):
        """
        input: base_out. shape = [batch_size * numsetments, cass_num]
        input variable order : global, global, ..., local, local, ...
        output vairable order: global, local, global, local, ...
        """
        out = None
        split_size = base_out.size()[0] // 2  # global, local
        global_base_out, local_base_out = torch.split(base_out, split_size, dim=0)
        for i in range(split_size):
            tmp = torch.stack([global_base_out[i], local_base_out[i]], dim=0)
            if out is None:
                out = tmp
            else:
                out = torch.cat([out, tmp], dim=0)
        return out

    @property
    def input_mean(self):
        return self.tsn_for_local.input_mean

    @property
    def input_std(self):
        return self.tsn_for_local.input_std

    @property
    def input_size(self):
        return self.tsn_for_local.input_size

    @property
    def crop_size(self):
        return self.tsn_for_local.crop_size

    @property
    def scale_size(self):
        return self.tsn_for_local.scale_size

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        tsn_local_policies = self.tsn_for_local.get_optim_policies()
        if not self.does_use_global_img:
            self._append_new_fc_param(tsn_local_policies)
            return tsn_local_policies

        tsn_global_policies = self.tsn_for_global.get_optim_policies()

        for i in range(len(tsn_global_policies)):
            tsn_global_policies[i]['params'].extend(
                tsn_local_policies[i]['params'])
        self._append_new_fc_param(tsn_global_policies)
        return tsn_global_policies

    def _append_new_fc_param(self, policies):
        new_fc_params = list(self.new_fc.parameters())
        for dic in policies:
            if 'normal_weight' in dic['name']:
                dic['params'].append(new_fc_params[0])
            if 'normal_bias' in dic['name'] and len(new_fc_params) == 2:
                dic['params'].append(new_fc_params[1])


class PartialTSN(TSN):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8,
                 crop_num=1, partial_bn=True,
                 does_use_global_img=False):
        super(PartialTSN, self).__init__(
            num_class,
            num_segments,
            modality,
            base_model,
            new_length,
            consensus_type,
            before_softmax,
            dropout,
            crop_num,
            partial_bn
        )
        self.does_use_global_img = does_use_global_img

    def forward(self, input):
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)

        base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))

        # if self.dropout > 0:
        #     base_out = self.new_fc(base_out)

        # if not self.before_softmax:
        #     base_out = self.softmax(base_out)

        return base_out
