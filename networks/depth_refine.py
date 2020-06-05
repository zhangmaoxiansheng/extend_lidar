import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from layers import *
import torchvision.models as models
from collections import OrderedDict
from .resnet_encoder import ResnetEncoder,resnet_multiimage_input
from .deform_conv import DeformConv
class Simple_Propagate(nn.Module):
    def __init__(self,crop_h,crop_w,mode='c'):
        super(Simple_Propagate, self).__init__()
        self.crop_h = crop_h#[96,128,160,192]
        self.crop_w = crop_w#[192,256,384,640]
        self.model_ref0 = nn.Sequential(ConvBlock(16+1,32),
                            ConvBlock(32,64),
                            ConvBlock(64,32),
                            ConvBlock(32,16),
                            Conv3x3(16,1),nn.Sigmoid())
        self.model_ref1 = nn.Sequential(ConvBlock(16+1,32),
                            ConvBlock(32,64),
                            ConvBlock(64,32),
                            ConvBlock(32,16),
                            Conv3x3(16,1),nn.Sigmoid())
        self.model_ref2 = nn.Sequential(ConvBlock(16+1,32),
                            ConvBlock(32,64),
                            ConvBlock(64,32),
                            ConvBlock(32,16),
                            Conv3x3(16,1),nn.Sigmoid())
        self.model_ref3 = nn.Sequential(ConvBlock(16+1,32),
                            ConvBlock(32,64),
                            ConvBlock(64,32),
                            ConvBlock(32,16),
                            Conv3x3(16,1),nn.Sigmoid())
        self.models = nn.ModuleList([self.model_ref0,self.model_ref1,self.model_ref2,self.model_ref3])
        #self.sigmoid = nn.Sigmoid()
        #self.cspn = Affinity_Propagate()
        self.crop_mode = mode
    
    def crop(self,image,h=160,w=320):
        #mask = torch.zeros_like(image)
        origin_h = image.size(2)
        origin_w = image.size(3)
        if self.crop_mode=='c' or self.crop_mode=='s' or self.crop_mode=='r':
            h_start = max(int(round((origin_h-h)/2)),0)
            h_end = min(h_start + h,origin_h)
            w_start = max(int(round((origin_w-w)/2)),0)
            w_end = min(w_start + w,origin_w)
            output = image[:,:,h_start:h_end,w_start:w_end] 
        elif self.crop_mode=='b':
            origin_h = image.size(2)
            origin_w = image.size(3)
            h_start = max(int(round(origin_h-h)),0)
            w_start = max(int(round((origin_w-w)/2)),0)
            w_end = min(w_start + w,origin_w)
            output = image[:,:,h_start:,w_start:w_end] 
        #mask[:,:,h_start:h_end,w_start:w_end] = 1
        return output
    #small feature 256 -> fuse small gt -> conv -> larger area -> next stage 
    def stage_forward(self,features,rgbd,dep_last,stage):
        #dep_last is the padded depth
        model = self.models[stage]
        h = self.crop_h[stage]
        w = self.crop_w[stage]
        rgbd = self.crop(rgbd,h,w)
        feature_crop = self.crop(features,h,w)
        #feature_crop = rgbd[:,1:,:,:]
        dep = rgbd[:,0,:,:].unsqueeze(1)
        mask = dep_last.sign()
        dep_fusion = dep_last * mask + dep * (1-mask) 
        feature_stage = torch.cat((feature_crop,dep_fusion),1)
        x = model(feature_stage)
        x, scale = self.scale_adjust(dep_last,x)
        return x, feature_stage
    
    def stage_pad(self,depth,h,w):
        if self.crop_mode == 'b':
            hs = depth.size(2)
            ws = depth.size(3)
            pad_w = (w-ws) // 2
            pad_h= h-hs
            pad = nn.ZeroPad2d((pad_w,pad_w,pad_h,0))
            depth_pad = pad(depth)
        else:
            hs = depth.size(2)
            ws = depth.size(3)
            pad_w = (w-ws) // 2
            pad_h= (h-hs) // 2
            pad = nn.ZeroPad2d((pad_w,pad_w,pad_h,pad_h))
            depth_pad = pad(depth)
        return depth_pad

    def stage_block(self,features,rgbd,dep_last,stage,outputs):
        for index, i in enumerate(stage):
            if index == len(stage) - 1:#the last stage
                outputs[("disp",i)], outputs[("condition",i)] = self.stage_forward(features,rgbd,dep_last,i)
            else:
                #with torch.no_grad():
                outputs[("disp",i)], outputs[("condition",i)] = self.stage_forward(features,rgbd,dep_last,i)
                dep_last = self.stage_pad(outputs[("disp",i)],self.crop_h[i+1],self.crop_w[i+1])
        return outputs
    def scale_adjust(self,gt,dep):
        if torch.median(dep[gt>0])>0:
            scale = torch.median(gt[gt>0]) / torch.median(dep[gt>0])
        else:
            scale = 1
        dep_ref = dep * scale
        return dep_ref,scale
    
    def forward(self,features, blur_depth, gt, rgb, stage):

        outputs = {}

        #print(scale)
        all_scale = gt / blur_depth
        blur_depth_o, scale = self.scale_adjust(gt,blur_depth)
        
        
        gt[all_scale>1.2] = 0
        gt[all_scale<0.8]=0
        gt_mask = gt.sign()
        blur_depth = gt_mask * gt + (1-gt_mask) * blur_depth_o
        rgbd = torch.cat((rgb, blur_depth_o),1)
        dep_0 = self.crop(gt,self.crop_h[0],self.crop_w[0])

        self.stage_block(features,rgbd,dep_0, stage, outputs)
        # final_dep = outputs[("disp",stage[-1])]
        # if we use 
        # scale = torch.median(gt[gt>0]) / torch.median(final_dep[gt>0])
        # for key, dep in outputs.items():
        #     outputs[key] = dep * scale
        
        outputs["blur_disp"] = blur_depth_o
        outputs["disp_all_in"] = blur_depth
        if self.crop_mode == 'b':
            #outputs["dense_gt"] = self.crop(blur_depth,72,164)
            outputs["dense_gt"] = self.crop(blur_depth,128,164)
        else:
            outputs["dense_gt"] = self.crop(blur_depth,64,128)
        outputs['scale'] = scale
        return outputs

class Iterative_Propagate(Simple_Propagate):
    def __init__(self,crop_h,crop_w,mode='c'):
        super(Iterative_Propagate, self).__init__(crop_h, crop_w,mode='c')
        self.model_ref0 = nn.Sequential(ConvBlock(16+1,32),
                            ConvBlock(32,16),
                            ConvBlock(16,8),
                            Conv3x3(8,1),nn.Sigmoid())
        self.model_ref1 = nn.Sequential(ConvBlock(16+1,32),
                            ConvBlock(32,16),
                            ConvBlock(16,8),
                            Conv3x3(8,1),nn.Sigmoid())
        self.model_ref2 = nn.Sequential(ConvBlock(16+1,32),
                            ConvBlock(32,32),
                            ConvBlock(32,16),
                            Conv3x3(16,1),nn.Sigmoid())
        self.model_ref3 = nn.Sequential(ConvBlock(16+1,32),
                            ConvBlock(32,32,2),
                            ConvBlock(32,16),
                            Conv3x3(16,1),nn.Sigmoid())
        self.model_ref4 = nn.Sequential(ConvBlock(16+1,32),
                            ConvBlock(32,32,4),
                            ConvBlock(32,16,2),
                            Conv3x3(16,1),nn.Sigmoid())
        self.models = nn.ModuleList([self.model_ref0,self.model_ref1,self.model_ref2,self.model_ref3])
        if len(self.crop_h) > 4:
            self.models.append(self.model_ref4)
        self.propagate_time = 3
        
    
    def stage_forward(self,features,rgbd,dep_last,stage):
        model = self.models[stage]
        h = self.crop_h[stage]
        w = self.crop_w[stage]
        #dep_last is the padded depth
        rgbd = self.crop(rgbd,h,w)
        feature_crop = self.crop(features,h,w)
        #feature_crop = rgbd[:,1:,:,:]
        dep = rgbd[:,0,:,:].unsqueeze(1)
        if torch.median(dep[dep_last>0]) > 0:
            scale = torch.median(dep_last[dep_last>0]) / torch.median(dep[dep_last>0])
        else:
            scale = 1
        dep = dep * scale
        mask = dep_last.sign()
        for i in range(self.propagate_time): 
            dep_fusion = dep_last * mask + dep * (1-mask)
            feature_stage = torch.cat((feature_crop,dep_fusion),1)
            dep = model(feature_stage)
            if torch.median(dep[dep_last>0]) > 0:
                scale = torch.median(dep_last[dep_last>0]) / torch.median(dep[dep_last>0])
            else:
                scale = 1
            dep = dep * scale
        return dep, feature_stage

class Iterative_Propagate_deep(Iterative_Propagate):
    def __init__(self,crop_h,crop_w,mode='c'):
        super(Iterative_Propagate, self).__init__(crop_h, crop_w,mode='c')
        self.model_ref0 = nn.Sequential(ConvBlock(16+1,32),
                            ConvBlock(32,64),
                            ConvBlock(64,32),
                            ConvBlock(32,16),
                            Conv3x3(16,1),nn.Sigmoid())
        self.model_ref1 = nn.Sequential(ConvBlock(16+1,32),
                            ConvBlock(32,64),
                            ConvBlock(64,32),
                            ConvBlock(32,16),
                            Conv3x3(16,1),nn.Sigmoid())
        self.model_ref2 =nn.Sequential(ConvBlock(16+1,32),
                            ConvBlock(32,64),
                            ConvBlock(64,32),
                            ConvBlock(32,16),
                            Conv3x3(16,1),nn.Sigmoid())
        self.model_ref3 = nn.Sequential(ConvBlock(16+1,32),
                            ConvBlock(32,32,2),
                            ConvBlock(32,32,2),
                            ConvBlock(32,16),
                            Conv3x3(16,1),nn.Sigmoid())
        self.model_ref4 = nn.Sequential(ConvBlock(16+1,32),
                            ConvBlock(32,32,4),
                            ConvBlock(32,32,2),
                            ConvBlock(32,16),
                            Conv3x3(16,1),nn.Sigmoid())
        self.models = nn.ModuleList([self.model_ref0,self.model_ref1,self.model_ref2,self.model_ref3])
        if len(self.crop_h) > 4:
            self.models.append(self.model_ref4)
        self.propagate_time = 3

class Iterative_Propagate_meta(Iterative_Propagate):
    def __init__(self,crop_h,crop_w,mode='c'):
        super(Iterative_Propagate, self).__init__(crop_h, crop_w,mode='c')
        self.model_ref0_2 = nn.Sequential(ConvBlock(16+1,32),
                            ConvBlock(32,16),
                            ConvBlock(16,8),
                            Conv3x3(8,1),nn.Sigmoid())
        self.model_ref3_4 = nn.Sequential(ConvBlock(16+1,32),
                            ConvBlock(32,32,4),
                            ConvBlock(32,16,2),
                            Conv3x3(16,1),nn.Sigmoid())
        #self.models = nn.ModuleList([self.model_ref0_2,self.model_ref0_2,self.model_ref0_2,self.model_ref3_4,self.model_ref3_4])
        # if len(self.crop_h) > 4:
        #     self.models.append(self.model_ref4)
        self.propagate_time = 4
    
    def stage_forward(self,features,rgbd,dep_last,stage):
        if stage < 3:
            model = self.model_ref0_2
        else:
            model = self.model_ref3_4
        h = self.crop_h[stage]
        w = self.crop_w[stage]
        #dep_last is the padded depth
        rgbd = self.crop(rgbd,h,w)
        feature_crop = self.crop(features,h,w)
        #feature_crop = rgbd[:,1:,:,:]
        dep = rgbd[:,0,:,:].unsqueeze(1)
        if torch.median(dep[dep_last>0]) > 0:
            scale = torch.median(dep_last[dep_last>0]) / torch.median(dep[dep_last>0])
        else:
            scale = 1
        dep = dep * scale
        mask = dep_last.sign()
        for i in range(self.propagate_time): 
            dep_fusion = dep_last * mask + dep * (1-mask)
            feature_stage = torch.cat((feature_crop,dep_fusion),1)
            dep = model(feature_stage)
            if torch.median(dep[dep_last>0]) > 0:
                scale = torch.median(dep_last[dep_last>0]) / torch.median(dep[dep_last>0])
            else:
                scale = 1
            dep = dep * scale
        return dep, feature_stage

class Iterative_Propagate_deform(Iterative_Propagate):
    def __init__(self,crop_h,crop_w,mode='c'):
        super(Iterative_Propagate_deform, self).__init__(crop_h, crop_w,mode='c')
        self.model_ref0 = nn.Sequential(ConvBlock(16+1,32),
                            ConvBlock(32,16),
                            Deformable_Conv(16,8),
                            Conv3x3(8,1),nn.Sigmoid())
        self.model_ref1 = nn.Sequential(ConvBlock(16+1,32),
                            ConvBlock(32,16),
                            Deformable_Conv(16,8),
                            Conv3x3(8,1),nn.Sigmoid())
        self.model_ref2 = nn.Sequential(ConvBlock(16+1,32),
                            ConvBlock(32,32),
                            Deformable_Conv(32,16),
                            Conv3x3(16,1),nn.Sigmoid())
        self.model_ref3 = nn.Sequential(ConvBlock(16+1,32),
                            ConvBlock(32,32,2),
                            Deformable_Conv(32,16),
                            Conv3x3(16,1),nn.Sigmoid())
        self.model_ref4 = nn.Sequential(ConvBlock(16+1,32),
                            ConvBlock(32,32,4),
                            ConvBlock(32,16,2),
                            Deformable_Conv(16,16),
                            Conv3x3(16,1),nn.Sigmoid())
        self.models = nn.ModuleList([self.model_ref0,self.model_ref1,self.model_ref2,self.model_ref3])
        if len(self.crop_h) > 4:
            self.models.append(self.model_ref4)
        self.propagate_time = 3

class Deformable_Conv(nn.Module):
    def __init__(self,inC,outC):
        super(Deformable_Conv,self).__init__()
        num_deformable_groups = 2
        kH, kW = 3,3
        self.offset_conv = nn.Conv2d(inC, num_deformable_groups * 2 * kH * kW, (kH,kW),1,1,bias=False)
        self.deform_conv = DeformConv(inC, outC, (kH,kW), 1, 1, deformable_groups = num_deformable_groups)
        nn.init.constant_(self.offset_conv.weight, 0.)
        #nn.init.constant_(self.offset_conv.bias, 0.)
    def forward(self, x):
        offset = self.offset_conv(x)
        x = self.deform_conv(x,offset)
        return x



class Iterative_Propagate_fc(Iterative_Propagate):
    def __init__(self,crop_h,crop_w,mode='c'):
        super(Iterative_Propagate_fc, self).__init__(crop_h, crop_w,mode='c')
        self.fc1 = nn.ModuleList([ConvBlock(16+1,2),nn.Linear(crop_h[0]*crop_w[0]*2,8)])
        self.fc2 = nn.ModuleList([ConvBlock(16+1,2),nn.Linear(crop_h[1]*crop_w[1]*2,8)])
        self.fc3 = nn.ModuleList([ConvBlock(16+1,2),nn.Linear(crop_h[2]*crop_w[2]*2,8)])
        self.fc4 = nn.ModuleList([ConvBlock(16+1,2),nn.Linear(crop_h[3]*crop_w[3]*2,8)])
        self.fc5 = nn.ModuleList([ConvBlock(16+1,2),nn.Linear(crop_h[4]*crop_w[4]*2,8)])
        
        self.fc = nn.Sequential(nn.Linear(8,4),
                                nn.Linear(4,4),
                                nn.Linear(4,1))
        self.fc_scale = nn.ModuleList([self.fc1,self.fc2,self.fc3,self.fc4,self.fc5])
        self.scale_list = []

    
    def stage_forward(self,features,rgbd,dep_last,stage):
        model = self.models[stage]
        h = self.crop_h[stage]
        w = self.crop_w[stage]
        fc_scale = self.fc_scale[stage]
        fc_conv = fc_scale[0]
        fc_fully = fc_scale[1]
        #dep_last is the padded depth
        rgbd = self.crop(rgbd,h,w)
        feature_crop = self.crop(features,h,w)
        #feature_crop = rgbd[:,1:,:,:]
        dep = rgbd[:,0,:,:].unsqueeze(1)
        scale = torch.median(dep_last[dep_last>0]) / torch.median(dep[dep_last>0])
        dep = dep * scale
        mask = dep_last.sign()
        for i in range(self.propagate_time): 
            dep_fusion = dep_last * mask + dep * (1-mask)
            feature_stage = torch.cat((feature_crop,dep_fusion),1)
            dep = model(feature_stage)
            scale_feat = fc_conv(feature_stage)
            scale_feat = scale_feat.view(scale_feat.shape[0],-1)
            scale_fc = self.fc(fc_fully(scale_feat))
            #print(scale_fc.shape)
            if torch.median(dep[dep_last>0])>0:
                scale = torch.median(dep_last[dep_last>0]) / torch.median(dep[dep_last>0])
            else:
                scale = 1
            dep = dep * scale
            for ii in range(scale_fc.shape[0]):
                dep[ii] *= scale_fc[ii]
            self.scale_list.append(scale_fc.mean())
        return dep, feature_stage


class U_refine(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers=18, pretrained=False, num_input_images=2):
        super(U_refine, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        self.num_output_channels = 1
        self.use_skips = True
        self.upsample_mode = 'nearest'
        self.scales = range(4)
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

         # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, blur_depth, gt, rgb):
        scale = torch.median(gt[gt>0]) / torch.median(blur_depth[gt>0])
        all_scale = gt / blur_depth
        blur_depth_o = blur_depth
        blur_depth_s = blur_depth * scale
        
        gt[all_scale>1.2] = 0
        gt[all_scale<0.8]=0
        gt_mask = gt.sign()
        blur_depth = gt_mask * gt + (1-gt_mask) * blur_depth_s
        x = rgb
        x = torch.cat((rgb,blur_depth,blur_depth,blur_depth),1)
        
        self.features = []
        x = self.encoder.conv1(x)
        
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        self.outputs = {}

        # decoder
        input_features = self.features
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
                #mask = self.sigmoid(self.convs[("maskconv", i)](x))
        #depth_ref = self.outputs[("disp",0)]# * mask + (1-mask) * blur_depth_o
        depth_ref = self.outputs[("disp", 0)]
        scale_ref = torch.median(gt[gt>0]) / torch.median(depth_ref[gt>0])
        self.outputs["blur_disp"] = blur_depth_s
        self.outputs["disp_all_in"] = blur_depth
        self.outputs['scale'] = scale
        return self.outputs#, blur_depth, scale_ref








