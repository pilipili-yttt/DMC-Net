import torch.nn.functional as F
import math


def padding(scale,img):
    _,_,H,W = img.shape
    out_H = math.ceil(H / scale) * scale
    out_W = math.ceil(W / scale) * scale
    pad_H = out_H - H
    pad_W = out_W - W
    pad = (pad_W // 2, pad_W - (pad_W // 2),pad_H // 2, pad_H - (pad_H // 2))
    pad_img = F.pad(img,pad,mode="reflect")
    return pad_img,pad

def inv_padding(pad,img):
    _, _, H, W = img.shape
    out_img = img[:,:,pad[2]:H-pad[3],pad[0]:W-pad[1]]
    return out_img
