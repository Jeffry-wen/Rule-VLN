import os
import cv2
import torch
import numpy as np
from PIL import Image

from lib import *
from utils.misc import *
from data.dataloader import *
from data.custom_transforms import *

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def get_format(source_list):
    img_count = len([i for i in source_list if i.lower().endswith(('.jpg', '.png', '.jpeg'))])
    vid_count = len([i for i in source_list if i.lower().endswith(('.mp4', '.avi', '.mov'))])
    if img_count * vid_count != 0:
        return ''
    elif img_count != 0:
        return 'Image'
    elif vid_count != 0:
        return 'Video'
    else:
        return ''


def segment_image(
    config_file:str,
    source:str,
    type:str='mask',
    gpu:bool=False,
    jit:bool=False,
    verbose:bool=False,
    checkpoint:str=None,
    debug_dir:str=None
):
    """Return an InSPyReNet mask or visualization for one image."""
    if checkpoint is None:
        raise ValueError("checkpoint is required")
    opt = load_config(config_file)

    model = eval(opt.Model.name)(**opt.Model)
    model.load_state_dict(
        torch.load(checkpoint, map_location=torch.device('cuda' if gpu else 'cpu')),
        strict=True
    )

    if gpu:
        model = model.cuda()
    model.eval()

    # jit
    if jit:
        jit_file = os.path.join(opt.Test.Checkpoint.checkpoint_dir, 'jit.pt')
        if not os.path.isfile(jit_file):
            model = Simplify(model)
            model = torch.jit.trace(model, torch.rand(1, 3, *opt.Test.Dataset.transforms.static_resize.size).cuda(), strict=False)
            torch.jit.save(model, jit_file)
        else:
            del model
            model = torch.jit.load(jit_file)

    sample = next(iter(eval('ImageLoader')([source], opt.Test.Dataset.transforms)))
    if gpu:
        sample = to_cuda(sample)

    with torch.no_grad():
        if jit:
            out = model(sample['image'])
        else:
            out = model(sample)

    pred = to_numpy(out['pred'], sample['shape'])
    img = np.array(sample['original'])

    if type == 'map':
        img = (np.stack([pred]*3, axis=-1) * 255).astype(np.uint8)

    elif type == 'mask':
        pred_bin = (pred > 0.5).astype(np.uint8) * 255
        img = pred_bin

    elif type == 'rgba':
        r,g,b = cv2.split(img)
        alpha = (pred * 255).astype(np.uint8)
        img = cv2.merge([r,g,b,alpha])

    elif type == 'green':
        pred_bin = (pred > 0.5).astype(np.uint8)
        bg = np.ones_like(img) * np.array([120,255,155], dtype=np.uint8)
        img = img * pred_bin[...,None] + bg * (1 - pred_bin[...,None])
        img = img.astype(np.uint8)

    elif type == 'blur':
        pred_bin = (pred > 0.5).astype(np.uint8)
        blur = cv2.GaussianBlur(img,(0,0),15)
        img = img * pred_bin[...,None] + blur * (1 - pred_bin[...,None])
        img = img.astype(np.uint8)

    elif type == 'overlay':
        pred_bin = (pred > 0.5).astype(np.uint8)
        bg = ((np.ones_like(img) * np.array([120,255,155],dtype=np.uint8) + img)//2)
        img = bg * pred_bin[...,None] + img * (1 - pred_bin[...,None])
        border = cv2.Canny(pred_bin*255,50,100)
        img[border != 0] = [120,255,155]
        img = img.astype(np.uint8)

    elif type.lower().endswith(('.jpg','.jpeg','.png')):
        background = cv2.imread(type)
        background = cv2.cvtColor(background,cv2.COLOR_BGR2RGB)
        background = cv2.resize(background,img.shape[:2][::-1])
        pred_bin = (pred > 0.5).astype(np.uint8)
        img = img * pred_bin[...,None] + background * (1 - pred_bin[...,None])
        img = img.astype(np.uint8)

    elif type == 'debug':
        debs = []
        for k in opt.Train.Debug.keys:
            debs.extend(out[k])
        os.makedirs(debug_dir or "debug_output", exist_ok=True)
        for i,j in enumerate(debs):
            log = torch.sigmoid(j).cpu().detach().numpy().squeeze()
            log = ((log - log.min()) / (log.max()-log.min())*255).astype(np.uint8)
            log = cv2.cvtColor(log,cv2.COLOR_GRAY2RGB)
            log = cv2.resize(log, img.shape[:2][::-1])
            Image.fromarray(log).save(
                os.path.join(debug_dir or "debug_output", f"{sample['name']}_{i}.png")
            )
        img = img

    else:
        raise ValueError(f"Unknown type: {type}")

    return img
