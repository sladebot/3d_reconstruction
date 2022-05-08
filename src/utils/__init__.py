import os
import shutil
import torch
from pathlib import Path
import collections

from src.lib.models.with_mobilenet import PoseEstimationWithMobileNet

from .convert import generate_rect, mp4_to_frames


def load_state(net, checkpoint):
    source_state = checkpoint['state_dict']
    target_state = net.state_dict()
    new_target_state = collections.OrderedDict()
    for target_key, target_value in target_state.items():
        if target_key in source_state and source_state[target_key].size() == target_state[target_key].size():
            new_target_state[target_key] = source_state[target_key]
        else:
            new_target_state[target_key] = target_state[target_key]
            print('[WARNING] Not found pre-trained parameters for {}'.format(target_key))

    net.load_state_dict(new_target_state)


def _load_mobile_net(ckpt):
    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(ckpt, map_location='cpu')
    load_state(net, checkpoint)
    return net


def preprocess(input_video, fps=24, mobilenet_ckpt="checkpoints/checkpoint_iter_370000.pth"):
    
    if Path(input_video).suffix != '.mp4':
        raise Exception("Please provide a valid mp4 file")
    
    vid2frames_dir = "__temp_vid_frames_input"
    if os.path.exists(vid2frames_dir):
        shutil.rmtree(vid2frames_dir)
    
    os.makedirs(vid2frames_dir)

    mp4_to_frames(input_video, vid2frames_dir, fps)
    
    net = _load_mobile_net(mobilenet_ckpt)

    for filename in os.listdir(vid2frames_dir):
        print(f"processing - {filename}")
        generate_rect(net.cuda(), [f"{vid2frames_dir}/{filename}"], 512)

    return vid2frames_dir

    
__all__ = [
    'mp4_to_frames',
    'generate_rect',
    'preprocess'
]