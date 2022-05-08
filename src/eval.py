import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.lib.data import EvalDataset, EvalPoseDataset
from src.lib.models import HGPIFuMRNet, HGPIFuNetwNML
from src.utils.mesh_utils import reconstruction, save_obj_mesh, save_obj_mesh_with_color


def generate(
        ckpt_path,
        dataroot,
        resolution,
        results_path,
        load_size,
        start_id,
        end_id,
        gpu_id,
        use_rect=False):
    
    start_id = start_id
    end_id = end_id

    cuda = torch.device('cuda:%d' % gpu_id if torch.cuda.is_available() else 'cpu')

    state_dict = None

    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=cuda)
        opt = state_dict['opt']
        opt.dataroot = dataroot
        opt.resolution = resolution
        opt.results_path = results_path
        opt.loadSize = load_size
    else:
        raise Exception("Failed to load checkpoint at - ", ckpt_path)

    if use_rect:
        test_dataset = EvalDataset(opt)
    else:
        test_dataset = EvalPoseDataset(opt)
    
    projection_mode = test_dataset.projection_mode
    print(state_dict.keys())
    opt_netG = state_dict["opt_netG"]
    netG = HGPIFuNetwNML(opt_netG, projection_mode).to(device=cuda)
    netMR = HGPIFuMRNet(opt, netG, projection_mode).to(device=cuda)

    def set_eval():
        netG.eval()

    netMR.load_state_dict(state_dict['model_state_dict'])

    # TODO: Check
    # os.makedirs(ckpt_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    os.makedirs('%s/%s/recon' % (results_path, opt.name), exist_ok=True)

    
    with torch.no_grad():
        set_eval()
        for i in tqdm(range(len(test_dataset))):
            if i >= len(test_dataset):
                break

            # TODO: Try multi-object
            test_data = test_dataset[i]

            save_path = '%s/%s/recon/result_%s_%d.obj' % (opt.results_path, opt.name, test_data['name'], opt.resolution)

            print(f"Saving to {save_path}")
            print(f"Model Mode:{netG.training}")
            gen_mesh(resolution, netMR, cuda, test_data, save_path, components=opt.use_compose)


def gen_mesh(res, net, cuda, data, save_path, thresh=0.5, use_octree=True, components=False):
    image_tensor_global = data['img_512'].to(device=cuda)
    image_tensor = data['img'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda)

    net.filter_global(image_tensor_global)
    net.filter_local(image_tensor[:,None])

    try:
        if net.netG.netF is not None:
            image_tensor_global = torch.cat([image_tensor_global, net.netG.nmlF], 0)
        if net.netG.netB is not None:
            image_tensor_global = torch.cat([image_tensor_global, net.netG.nmlB], 0)
    except:
        pass

    b_min = data['b_min']
    b_max = data['b_max']
    try:
        save_img_path = save_path[:-4] + '.png'
        save_img_list = []
        for v in range(image_tensor_global.shape[0]):
            save_img = (np.transpose(image_tensor_global[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
            save_img_list.append(save_img)
        save_img = np.concatenate(save_img_list, axis=1)
        cv2.imwrite(save_img_path, save_img)

        verts, faces, _, _ = reconstruction(
            net, cuda, calib_tensor, res, b_min, b_max, thresh, use_octree=use_octree, num_samples=50000)
        print(f"Predicted vertex shape {np.shape(verts)}")
        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()
        # if 'calib_world' in data:
        #     calib_world = data['calib_world'].numpy()[0]
        #     verts = np.matmul(np.concatenate([verts, np.ones_like(verts[:,:1])],1), inv(calib_world).T)[:,:3]

        color = np.zeros(verts.shape)
        interval = 50000
        for i in range(len(color) // interval + 1):
            left = i * interval
            if i == len(color) // interval:
                right = -1
            else:
                right = (i + 1) * interval
            net.calc_normal(verts_tensor[:, None, :, left:right], calib_tensor[:,None], calib_tensor)
            nml = net.nmls.detach().cpu().numpy()[0] * 0.5 + 0.5
            color[left:right] = nml.T

        save_obj_mesh_with_color(save_path, verts, faces, color)
    except Exception as e:
        print(e)