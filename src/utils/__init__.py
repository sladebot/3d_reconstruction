from pathlib import Path
import os
import shutil

from src.models.with_mobilenet import PoseEstimationWithMobileNet

from .convert import mp4_to_frames, generate_rect

def _load_mobile_net(ckpt="checkpoints/checkpoint_iter_370000.pth"):
    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(ckpt, map_location='cpu')
    load_state(net, checkpoint)
    return net



def preprocess(input_video, mobilenet_ckpt):
    
    if Path(input_video).suffix != '.mp4':
        raise Exception("Please provide a valid mp4 file")
    
    vid2frames_dir = "__temp_vid_frames_input"
    if os.path.exists(vid2frames_dir):
        shutil.rmtree(vid2frames_dir)
    
    os.makedirs(vid2frames_dir)

    mp4_to_frames(input_video, vid2frames_dir, fps)
    
    net = _load_mobile_net(mobilenet_ckpt)

    for filename in os.listdir(vid2frames_dir):
        generate_rect(net.cuda(), [f"{vid2frames_dir}/{filename}"], 512)

    return __temp_vid_frames_input

    
__all__ = [
    'mp4_to_frames',
    'generate_rect'
]