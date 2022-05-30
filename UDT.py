from os.path import join, isdir
from os import makedirs
import argparse
import json
import numpy as np
import torch
import ops
import cv2
import time as time
from util import crop_chw, gaussian_shaped_labels, cxy_wh_2_rect1, rect1_2_cxy_wh, cxy_wh_2_bbox
from net import DCFNet

from got10k.trackers import Tracker


class TrackerConfig(object):
    # These are the default hyper-params for DCFNet
    # OTB2013 / AUC(0.665)
    feature_path = '/home/airlab/PycharmProjects/pythonProject5/UDT_GOT10K/param.pth'
    crop_sz = 125

    lambda0 = 1e-4
    padding = 2
    output_sigma_factor = 0.1
    interp_factor = 0.01
    num_scale = 3
    scale_step = 1.0275
    scale_factor = scale_step ** (np.arange(num_scale) - num_scale / 2)
    min_scale_factor = 0.2
    max_scale_factor = 5
    scale_penalty = 0.9925
    scale_penalties = scale_penalty ** (np.abs((np.arange(num_scale) - num_scale / 2)))

    net_input_size = [crop_sz, crop_sz]
    net_average_image = np.array([104, 117, 123]).reshape(-1, 1, 1).astype(np.float32)
    output_sigma = crop_sz / (1 + padding) * output_sigma_factor
    y = gaussian_shaped_labels(output_sigma, net_input_size)
    yf = torch.fft.rfft(torch.Tensor(y).view(1, 1, crop_sz, crop_sz).cuda(),2)
    cos_window = torch.Tensor(np.outer(np.hanning(crop_sz), np.hanning(crop_sz))).cuda()


class UDTracker(Tracker):
    def __init__(self, config=TrackerConfig(), gpu=True):
        super(UDTracker,self).__init__(name='UDTracker')
        self.gpu = gpu
        self.config = config
        self.net = DCFNet(config)
        self.net.load_param(config.feature_path)
        self.net.eval()
        if gpu:
            self.net.cuda()




    def init(self,image,box):
        init_rect = box
        target_pos, target_sz = rect1_2_cxy_wh(init_rect)  # OTB label is 1-indexed
        # confine results
        self.min_sz = np.maximum(self.config.min_scale_factor * target_sz, 4)
        self.max_sz = np.minimum(image.shape[:2], self.config.max_scale_factor * target_sz)

        # crop template
        window_sz = target_sz * (1 + self.config.padding)
        bbox = cxy_wh_2_bbox(target_pos, window_sz)
        patch = crop_chw(image, bbox, self.config.crop_sz)

        target = patch - self.config.net_average_image
        self.net.update(torch.Tensor(np.expand_dims(target, axis=0)).cuda())
        self.target_pos, self.target_sz = target_pos, target_sz
        self.patch_crop = np.zeros((self.config.num_scale, patch.shape[0], patch.shape[1], patch.shape[2]),
                                   np.float32)  # buff

        # confine results
        min_sz = np.maximum(self.config.min_scale_factor * target_sz, 4)
        max_sz = np.minimum(image.shape[:2], self.config.max_scale_factor * target_sz)

        # crop template
        window_sz = target_sz * (1 + self.config.padding)
        bbox = cxy_wh_2_bbox(target_pos, window_sz)
        patch = crop_chw(image, bbox, self.config.crop_sz)

        target = patch - self.config.net_average_image
        self.net.update(torch.Tensor(np.expand_dims(target, axis=0)).cuda())

        res = [cxy_wh_2_rect1(target_pos, target_sz)]  # save in .txt
        patch_crop = np.zeros((self.config.num_scale, patch.shape[0], patch.shape[1], patch.shape[2]), np.float32)


    def update(self,im):
        for i in range(self.config.num_scale):  # crop multi-scale search region
            window_sz = self.target_sz * (self.config.scale_factor[i] * (1 + self.config.padding))
            bbox = cxy_wh_2_bbox(self.target_pos, window_sz)
            self.patch_crop[i, :] = crop_chw(im, bbox, self.config.crop_sz)

        search = self.patch_crop - self.config.net_average_image

        if self.gpu:
            response = self.net(torch.Tensor(search).cuda())
        else:
            response = self.net(torch.Tensor(search))
        peak, idx = torch.max(response.view(self.config.num_scale, -1), 1)
        peak = peak.data.cpu().numpy() * self.config.scale_penalties
        best_scale = np.argmax(peak)
        r_max, c_max = np.unravel_index(idx[best_scale].detach().cpu().numpy(), self.config.net_input_size)

        if r_max > self.config.net_input_size[0] / 2:
            r_max = r_max - self.config.net_input_size[0]
        if c_max > self.config.net_input_size[1] / 2:
            c_max = c_max - self.config.net_input_size[1]
        window_sz = self.target_sz * (self.config.scale_factor[best_scale] * (1 + self.config.padding))

        self.target_pos = self.target_pos + np.array([c_max, r_max]) * window_sz / self.config.net_input_size
        self.target_sz = np.minimum(np.maximum(window_sz / (1 + self.config.padding), self.min_sz), self.max_sz)

        # model update
        window_sz = self.target_sz * (1 + self.config.padding)
        bbox = cxy_wh_2_bbox(self.target_pos, window_sz)
        patch = crop_chw(im, bbox, self.config.crop_sz)
        target = patch - self.config.net_average_image
        self.net.update(torch.Tensor(np.expand_dims(target, axis=0)).cuda(), lr=self.config.interp_factor)

        return cxy_wh_2_rect1(self.target_pos, self.target_sz)  # 1-index

    def track(self, img_files, box, visualize=False):
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        for f, img_file in enumerate(img_files):
            img = ops.read_image(img_file)

            begin = time.time()
            if f == 0:
                self.init(img, box)
            else:
                boxes[f, :] = self.update(img)
            times[f] = time.time() - begin


        return boxes, times
