import os
import os.path
import json
import numpy as np
import random
import h5py
import torch
from PIL import Image
import glob
import torch.utils.data as udata
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF


def _safe_minmax_norm(data, min_val=None, max_val=None, eps=1e-6):
    if min_val is None:
        min_val = float(np.min(data))
    if max_val is None:
        max_val = float(np.max(data))
    denom = max_val - min_val
    if denom < eps:
        return np.zeros_like(data, dtype=np.float32), max_val, min_val
    return (data - min_val) / denom, max_val, min_val


class NYU_DC(udata.Dataset):
    def __init__(self, win, path, aug_mode,  train=True):
        """
        初始化 Dataset_NYU 类。

        :param win: 裁剪窗口大小
        :param path: 数据路径，包含保存的npy文件
        :param aug_mode: 是否进行数据增强
        :param scale: 用于处理图像的缩放因子
        :param train: 是否为训练集
        """
        super(NYU_DC, self).__init__()

        self.train = train
        self.win = win
        self.aug_mode = aug_mode

        rgb_dir = os.path.join(path, "train_rgb.npy" if train else "val_rgb.npy")
        raw_dir = os.path.join(path, "train_raw.npy" if train else "val_raw.npy")
        gt_dir = os.path.join(path, "train_gt.npy" if train else "val_gt.npy")

        self.rgb_list = self.totensor(np.load(rgb_dir))
        self.raw_list = self.totensor(np.expand_dims(np.load(raw_dir), axis=-1))  # 可能需要增加维度
        self.gt_list = self.totensor(np.expand_dims(np.load(gt_dir), axis=-1))  # 同上

        if not train:
            minmax_dir = os.path.join(path, "test_minmax.npy")
            self.minmax_list = torch.from_numpy(np.load(minmax_dir))

    def argument(self, rgb, raw, gt):
        """
        对输入数据进行随机水平翻转和垂直翻转。

        :param rgb: RGB图像
        :param raw: 原始深度图
        :param gt: Ground truth深度图
        :return: 翻转后的数据
        """
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5

        if hflip:
            rgb = torch.flip(rgb, dims=[2]).clone()
            raw = torch.flip(raw, dims=[2]).clone()
            gt = torch.flip(gt, dims=[2]).clone()
        if vflip:
            rgb = torch.flip(rgb, dims=[1]).clone()
            raw = torch.flip(raw, dims=[1]).clone()
            gt = torch.flip(gt, dims=[1]).clone()

        return rgb, raw, gt

    def get_patch_random(self, rgb, raw, gt):
        """
        随机裁剪图像块。

        :param rgb: RGB图像
        :param raw: 原始深度图
        :param gt: Ground truth深度图
        :return: 裁剪后的图像块
        """
        h, w = rgb.shape[1:3]
        win_h = min(self.win, h)
        win_w = min(self.win, w)
        if win_h == h and win_w == w:
            return rgb, raw, gt

        x = random.randrange(0, w - win_w + 1)
        y = random.randrange(0, h - win_h + 1)
        rgb = rgb[:, y:y + win_h, x:x + win_w]
        raw = raw[:, y:y + win_h, x:x + win_w]
        gt = gt[:, y:y + win_h, x:x + win_w]
        return rgb, raw, gt

    def totensor(self, data):
        """
        将数据转为tensor格式，并调整维度。

        :param data: 输入数据
        :return: 转换后的tensor
        """
        data = np.ascontiguousarray(data)
        data = data.transpose(0, 3, 1, 2)  # 转换为CHW格式
        data_tensor = torch.from_numpy(data).float()
        return data_tensor

    def __len__(self):
        """
        返回数据集的大小。
        """
        return self.rgb_list.shape[0]

    def _resize_and_center_crop(self, tensor, target_hw=(240, 320), crop_hw=(228, 304)):
        """
        先将tensor resize到target_hw (H, W)，再中心裁剪到crop_hw (H, W)。

        :param tensor: 形状为 (C, H, W) 的torch.Tensor
        :param target_hw: (H, W)
        :param crop_hw: (H, W)
        :return: 处理后的tensor
        """
        if tensor.dim() != 3:
            raise ValueError("Expected tensor shape (C, H, W)")

        target_h, target_w = target_hw
        crop_h, crop_w = crop_hw

        tensor = tensor.unsqueeze(0)
        tensor = F.interpolate(tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)
        tensor = tensor.squeeze(0)

        _, h, w = tensor.shape
        top = max((h - crop_h) // 2, 0)
        left = max((w - crop_w) // 2, 0)
        tensor = tensor[:, top:top + crop_h, left:left + crop_w]
        return tensor

    def __getitem__(self, idx):
        """
        获取指定索引的数据项。

        :param idx: 索引
        :return: 字典，包含RGB图像、原始深度图、Ground truth深度图
        """
        rgb_data = self.rgb_list[idx]
        raw_data = self.raw_list[idx]
        gt_data = self.gt_list[idx]

        rgb_data = self._resize_and_center_crop(rgb_data)
        raw_data = self._resize_and_center_crop(raw_data)
        gt_data = self._resize_and_center_crop(gt_data)

        if self.train:
            if self.aug_mode:
                rgb_data, raw_data, gt_data = self.argument(rgb_data, raw_data, gt_data)
            rgb_data, raw_data, gt_data = self.get_patch_random(rgb_data, raw_data, gt_data)
            minmax_data = torch.tensor([0, 0])  # 训练集时不使用 minmax
            gt_data = gt_data / 10.0  # 训练时归一化
        else:
            if self.minmax_list.dim() == 2 and self.minmax_list.shape[1] == 2:
                minmax_data = self.minmax_list[idx]
            else:
                minmax_data = self.minmax_list[:, idx]
            

        return {
            'Guide': rgb_data/10,
            'LQ': raw_data/10,
            'HQ': gt_data,
            'minmax': minmax_data
        }

class nyu_ma(udata.Dataset):
    """NYU-MA dataloader handling sparse depth supervision."""

    class ToNumpy:
        def __call__(self, sample):
            return np.array(sample)

    def __init__(self, path, mode='train', num_sample=500, mul_factor=1., num_mask=8,
                 scale_kcam=False, rand_scale=True):
        super(nyu_ma, self).__init__()
        if mode not in ('train', 'val'):
            raise ValueError("mode must be 'train' or 'val'")

        self.mode = mode
        self.num_sample = num_sample
        self.num_mask = num_mask
        self.mul_factor = mul_factor
        self.scale_kcam = scale_kcam
        self.rand_scale = rand_scale
        
        self.height = 240
        self.width = 320
        self.crop_size = (228, 304)
        self.numpy_converter = self.ToNumpy()

        self.Kcam = torch.from_numpy(np.array([
            [5.1885790117450188e+02, 0, 3.2558244941119034e+02],
            [0, 5.1946961112127485e+02, 2.5373616633400465e+02],
            [0, 0, 1.0],
        ], dtype=np.float32))

        pattern = os.path.join(path, mode, "**", "*.h5")
        self.sample_list = sorted(glob.glob(pattern, recursive=True))
        if not self.sample_list:
            raise RuntimeError(f"No h5 files found for mode '{mode}' under {path}")

    def __len__(self):
        if self.mode == 'train':
            return len(self.sample_list)
        if self.mode == 'val':
            return self.num_mask * len(self.sample_list)
        raise NotImplementedError

    def __getitem__(self, idx):
        if self.mode == 'val':
            seed = idx % self.num_mask
            data_idx = idx // self.num_mask
        else:
            data_idx = idx
            seed = None

        path_file = self.sample_list[data_idx]
        with h5py.File(path_file, 'r') as file:
            rgb_h5 = file['rgb'][:].transpose(1, 2, 0)
            dep_h5 = file['depth'][:]

        rgb = Image.fromarray(rgb_h5.astype(np.uint8), mode='RGB')
        dep = Image.fromarray(dep_h5.astype('float32'), mode='F')
        Kcam = self.Kcam.clone()

        if self.mode == 'train':
            scale = int(self.height * (np.random.uniform(1.0, 1.5) if self.rand_scale else 1.0))
            degree = np.random.uniform(-5.0, 5.0)
            flip = np.random.rand()

            if flip > 0.5:
                rgb = TF.hflip(rgb)
                dep = TF.hflip(dep)
                Kcam[0, 2] = rgb.width - 1 - Kcam[0, 2]

            rgb = TF.rotate(rgb, angle=degree)
            dep = TF.rotate(dep, angle=degree)

            rgb_transform = T.Compose([
                T.Resize(scale),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                T.CenterCrop(self.crop_size),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            depth_transform = T.Compose([
                T.Resize(scale),
                T.CenterCrop(self.crop_size),
                self.numpy_converter,
                T.ToTensor(),
            ])

            rgb = rgb_transform(rgb)
            dep = depth_transform(dep)

            if self.scale_kcam:
                Kcam[:2] = Kcam[:2] * (scale / self.height)
        else:
            rgb_transform = T.Compose([
                T.Resize(self.height),
                T.CenterCrop(self.crop_size),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            depth_transform = T.Compose([
                T.Resize(self.height),
                T.CenterCrop(self.crop_size),
                self.numpy_converter,
                T.ToTensor(),
            ])
            rgb = rgb_transform(rgb)
            dep = depth_transform(dep)
            seed = seed if seed is not None else 0

        if self.mode == 'train':
            dep_sp = self.get_sparse_depth(dep, self.num_sample)
        else:
            dep_sp = self.mask_sparse_depth(dep, self.num_sample, seed)


        Kcam[:2] = Kcam[:2] / 2.0
        Kcam[0, 2] += 0
        Kcam[1, 2] += 8

        dep_sp = dep_sp * self.mul_factor
        dep = dep * self.mul_factor
        valid = dep > 1e-6
        min_val = torch.tensor(0.0, device=dep.device)
        max_val =torch.tensor(10.0, device=dep.device)
        dep_sp_norm = torch.zeros_like(dep_sp)
        valid_sp = dep_sp > 1e-6  # 找到稀疏采样点的位置
        if valid_sp.any():
            dep_sp_norm[valid_sp] = (dep_sp[valid_sp] - min_val) / (max_val - min_val)

        if self.mode == 'train':
            dep = (dep - min_val) / (max_val - min_val)

        if self.mode == 'val':
            min_val = dep[valid].min()
            max_val = dep[valid].max()


        dep_sp = dep_sp_norm
        return {
            'Guide': rgb,
            'LQ': dep_sp,
            'HQ': dep,
            'Kcam': Kcam,
            'minmax': torch.tensor([max_val, min_val], dtype=torch.float32)
        }

    def get_sparse_depth(self, dep, num_sample):
        channel, height, width = dep.shape
        idx_nnz = torch.nonzero(dep.view(-1) > 0.0001, as_tuple=False)
        num_idx = len(idx_nnz)
        if num_idx == 0:
            dep_clone = dep.clone()
            dep_clone[:, 20:-20:10, 20:-20:10] = 3.0
            return dep_clone

        num_sample = min(num_sample, num_idx)
        idx_sample = torch.randperm(num_idx)[:num_sample]
        idx_nnz = idx_nnz[idx_sample]

        mask = torch.zeros((channel * height * width))
        mask[idx_nnz] = 1.0
        mask = mask.view(channel, height, width)

        return dep * mask.type_as(dep)

    def mask_sparse_depth(self, dep, num_sample, seed):
        channel, height, width = dep.shape
        flat = dep.cpu().numpy().reshape(-1)
        num_sample = min(num_sample, flat.size)
        rng = np.random.default_rng(seed)
        index = rng.choice(flat.size, num_sample, replace=False)
        dep_sp = np.zeros_like(flat)
        dep_sp[index] = flat[index]
        dep_sp = dep_sp.reshape(channel, height, width)
        return torch.from_numpy(dep_sp.astype(np.float32))
class Dataset_TOFDC(udata.Dataset):
    def __init__(self, path, win=256, aug_mode=False, train=False):
        super(Dataset_TOFDC, self).__init__()
        self.train = train
        self.win = win
        self.aug_mode = aug_mode
        self.HQ_list = []
        self.Guide_list = []
        self.LQ_list = []

        suffix = "train" if self.train else "test"
        folder_path = os.path.join(path, suffix)

        if not self.train:
            rgb_folder_path = os.path.join(folder_path, 'rgb')
            hq_folder_path = os.path.join(folder_path, 'tgv_gt')
            lq_folder_path = os.path.join(folder_path, 'tof')

            for file in os.listdir(rgb_folder_path):
                name = file[:-13]
                hq_name = name + '_tgv_gt_crop.png'
                lq_name = name + '_depth_crop.png'

                self.Guide_list.append(os.path.join(rgb_folder_path, file))
                self.HQ_list.append(os.path.join(hq_folder_path, hq_name))
                self.LQ_list.append(os.path.join(lq_folder_path, lq_name))

        else:
            sub_folders = sorted(os.listdir(folder_path))
            for sub in sub_folders:
                sub_folder = os.path.join(folder_path, sub)
                if not os.path.isdir(sub_folder):
                    continue

                rgb_folder_path = os.path.join(sub_folder, 'rgb')
                hq_folder_path = os.path.join(sub_folder, 'tgv_gt')
                lq_folder_path = os.path.join(sub_folder, 'tof')

                for file in os.listdir(rgb_folder_path):
                    name = file[:-13]
                    hq_name = name + '_tgv_gt_crop.png'
                    lq_name = name + '_depth_crop.png'

                    self.Guide_list.append(os.path.join(rgb_folder_path, file))
                    self.HQ_list.append(os.path.join(hq_folder_path, hq_name))
                    self.LQ_list.append(os.path.join(lq_folder_path, lq_name))


    def totensor(self, img):
        img = np.ascontiguousarray(img)
        img = img.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img).float()
        return img_tensor

    def load_depth(self, depth_data):
        return np.array(Image.open(depth_data)).astype(np.float32)

    def load_color(self, color_data):
        return np.array(Image.open(color_data).convert("RGB")).astype(np.float32)

    def argument(self, hq, lq, guide):
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5

        if hflip:
            hq = torch.flip(hq, dims=[2]).clone()
            lq = torch.flip(lq, dims=[2]).clone()
            guide = torch.flip(guide, dims=[2]).clone()
        if vflip:
            hq = torch.flip(hq, dims=[1]).clone()
            lq = torch.flip(lq, dims=[1]).clone()
            guide = torch.flip(guide, dims=[1]).clone()

        return hq, lq, guide

    def get_patch_random(self, hq, lq, guide):
        win = self.win
        h, w = hq.shape[1:3]
        x = random.randrange(0, w - win + 1)
        y = random.randrange(0, h - win + 1)

        hq = hq[:, y:y+win, x:x+win]
        lq = lq[:, y:y+win, x:x+win]
        guide = guide[:, y:y+win, x:x+win]
        return hq, lq, guide

    def __len__(self):
        return len(self.Guide_list)

    def __getitem__(self, idx):
        Guide_data = self.load_color(self.Guide_list[idx])
        HQ_data = self.load_depth(self.HQ_list[idx])

        h, w = HQ_data.shape[:2]

        LQ_data = np.array(Image.open(self.LQ_list[idx])).astype(np.float32)

        maxx = 6000.0
        minn = 0.0
        LQ_data, _, _ = _safe_minmax_norm(LQ_data, minn, maxx)

        Guide_data, _, _ = _safe_minmax_norm(Guide_data)

        if self.train:
            HQ_data, _, _ = _safe_minmax_norm(HQ_data, minn, maxx)

        HQ_data = self.totensor(np.expand_dims(HQ_data, axis=-1))
        LQ_data = self.totensor(np.expand_dims(LQ_data, axis=-1))
        Guide_data = self.totensor(Guide_data)

        minmax = torch.tensor([maxx, minn])

        if self.train:
            if self.aug_mode:
                HQ_data, LQ_data, Guide_data = self.argument(HQ_data, LQ_data, Guide_data)
            HQ_data, LQ_data, Guide_data = self.get_patch_random(HQ_data, LQ_data, Guide_data)

        return {
            'HQ': HQ_data,
            'Guide': Guide_data,
            'LQ': LQ_data,
            'minmax': minmax
        }


class Dataset_DIML(udata.Dataset):
    """
    DIML depth completion dataset.
    Directory structure:
        path/train/XXXX/XXXX_color.png   (288, 512, 3) uint8
        path/train/XXXX/XXXX_depth.png   (270, 480) int32   <-- different size!
        path/train/XXXX/XXXX_gt.png      (288, 512) int32
    Depth values in mm, use ratio=1000 to convert to meters.
    Preprocessing: Resize short-edge to 192, then CenterCrop to (192, 320).
    """

    def __init__(self, path, win=256, aug_mode=False, train=False):
        super(Dataset_DIML, self).__init__()
        self.train = train
        self.win = win
        self.aug_mode = aug_mode

        size_diml = 192
        norm_size = [192, 320]

        self.depth_transform = T.Compose([
            T.Resize(size_diml),
            T.CenterCrop(norm_size),
            T.ConvertImageDtype(torch.float),
        ])
        self.rgb_transform = T.Compose([
            T.Resize(size_diml),
            T.CenterCrop(norm_size),
            T.ConvertImageDtype(torch.float),
        ])

        self.HQ_list = []
        self.Guide_list = []
        self.LQ_list = []

        suffix = "train" if self.train else "test"
        folder_path = os.path.join(path, suffix)

        sub_folders = sorted(os.listdir(folder_path))
        for sub in sub_folders:
            sub_dir = os.path.join(folder_path, sub)
            if not os.path.isdir(sub_dir):
                continue
            color_path = os.path.join(sub_dir, f"{sub}_color.png")
            depth_path = os.path.join(sub_dir, f"{sub}_depth.png")
            gt_path = os.path.join(sub_dir, f"{sub}_gt.png")
            if os.path.isfile(color_path) and os.path.isfile(depth_path) and os.path.isfile(gt_path):
                self.Guide_list.append(color_path)
                self.LQ_list.append(depth_path)
                self.HQ_list.append(gt_path)

    def load_depth_tensor(self, path, ratio=1.0):
        """Load depth as (1, H, W) float tensor, divided by ratio."""
        img = Image.open(path)
        tensor = torch.from_numpy(np.array(img).astype(np.float32)).unsqueeze(0)  # (1, H, W)
        return tensor / ratio

    def load_rgb_tensor(self, path):
        """Load RGB as (3, H, W) float tensor, normalized by self min/max to [0, 1]."""
        data = np.array(Image.open(path).convert("RGB")).astype(np.float32)  # (H, W, 3) [0,255]
        data, _, _ = _safe_minmax_norm(data)  # 用自身 min/max 归一化到 [0, 1]
        data = np.ascontiguousarray(data.transpose(2, 0, 1))  # (3, H, W)
        return torch.from_numpy(data)  # float32

    def argument(self, hq, lq, guide):
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        if hflip:
            hq = torch.flip(hq, dims=[2]).clone()
            lq = torch.flip(lq, dims=[2]).clone()
            guide = torch.flip(guide, dims=[2]).clone()
        if vflip:
            hq = torch.flip(hq, dims=[1]).clone()
            lq = torch.flip(lq, dims=[1]).clone()
            guide = torch.flip(guide, dims=[1]).clone()
        return hq, lq, guide

    def get_patch_random(self, hq, lq, guide):
        win = self.win
        h, w = hq.shape[1:3]
        if h <= win or w <= win:
            return hq, lq, guide
        x = random.randrange(0, w - win + 1)
        y = random.randrange(0, h - win + 1)
        return hq[:, y:y+win, x:x+win], lq[:, y:y+win, x:x+win], guide[:, y:y+win, x:x+win]

    def __len__(self):
        return len(self.Guide_list)

    def __getitem__(self, idx):
        ratio = 10000.0
        maxx = ratio
        minn = 0.0

        Guide_data = self.load_rgb_tensor(self.Guide_list[idx])     # (3, 288, 512)
        HQ_data = self.load_depth_tensor(self.HQ_list[idx], ratio)  # (1, 288, 512)

        Guide_data = self.rgb_transform(Guide_data)      # (3, 192, 320)
        HQ_data = self.depth_transform(HQ_data)           # (1, 192, 320)


        raw_depth = self.load_depth_tensor(self.LQ_list[idx], ratio)  # (1, 270, 480)
        raw_depth = self.depth_transform(raw_depth)                    # (1, 192, 320)
        mask = (raw_depth <= 0.001).float()
        LQ_data = HQ_data * (1.0 - mask)

        if not self.train:
            HQ_data = HQ_data * ratio

        minmax = torch.tensor([maxx, minn])

        if self.train:
            if self.aug_mode:
                HQ_data, LQ_data, Guide_data = self.argument(HQ_data, LQ_data, Guide_data)
            HQ_data, LQ_data, Guide_data = self.get_patch_random(HQ_data, LQ_data, Guide_data)

        return {
            'HQ': HQ_data,
            'Guide': Guide_data,
            'LQ': LQ_data,
            'minmax': minmax
        }


class Dataset_SUN(udata.Dataset):
    def __init__(self, path, win=256, aug_mode=False, train=False):
        super(Dataset_SUN, self).__init__()
        self.train = train
        self.win = win
        self.aug_mode = aug_mode
        self.scale = 1

        size_sun = (240, 320)
        crop_size = (228, 304)

        self.depth_transform = T.Compose([
            T.Resize(size_sun),
            T.CenterCrop(crop_size),
            T.ConvertImageDtype(torch.float),
        ])
        self.rgb_transform = T.Compose([
            T.Resize(size_sun),
            T.CenterCrop(crop_size),
            T.ConvertImageDtype(torch.float),
        ])

        self.HQ_list = []
        self.Guide_list = []
        self.LQ_list = []

        suffix = "train" if self.train else "test"
        folder_path = os.path.join(path, suffix)

        if os.path.isdir(folder_path):
            sub_folders = sorted(os.listdir(folder_path))
            for sub in sub_folders:
                sub_dir = os.path.join(folder_path, sub)
                if not os.path.isdir(sub_dir):
                    continue
                color_path = os.path.join(sub_dir, f"{sub}_color.jpg")
                depth_path = os.path.join(sub_dir, f"{sub}_gt.png")
                gt_path = os.path.join(sub_dir, f"{sub}_depth.png")
                if os.path.isfile(color_path) and os.path.isfile(depth_path) and os.path.isfile(gt_path):
                    self.Guide_list.append(color_path)
                    self.LQ_list.append(depth_path)
                    self.HQ_list.append(gt_path)

    def load_depth_tensor(self, path, ratio=1.0):
        """Load depth as (1, H, W) float tensor, divided by ratio."""
        img = Image.open(path)
        tensor = torch.from_numpy(np.array(img).astype(np.float32)).unsqueeze(0)  # (1, H, W)
        return tensor / ratio

    def load_rgb_tensor(self, path):
        """Load RGB as (3, H, W) float tensor, normalized by self min/max to [0, 1]."""
        data = np.array(Image.open(path).convert("RGB")).astype(np.float32)  # (H, W, 3) [0,255]
        data, _, _ = _safe_minmax_norm(data)  # 用自身 min/max 归一化到 [0, 1]
        data = np.ascontiguousarray(data.transpose(2, 0, 1))  # (3, H, W)
        return torch.from_numpy(data)  # float32

    def argument(self, hq, lq, guide):
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        if hflip:
            hq = torch.flip(hq, dims=[2]).clone()
            lq = torch.flip(lq, dims=[2]).clone()
            guide = torch.flip(guide, dims=[2]).clone()
        if vflip:
            hq = torch.flip(hq, dims=[1]).clone()
            lq = torch.flip(lq, dims=[1]).clone()
            guide = torch.flip(guide, dims=[1]).clone()
        return hq, lq, guide

    def get_patch_random(self, hq, lq, guide):
        win = self.win
        h, w = hq.shape[1:3]
        if h <= win or w <= win:
            return hq, lq, guide
        x = random.randrange(0, w - win + 1)
        y = random.randrange(0, h - win + 1)
        return hq[:, y:y+win, x:x+win], lq[:, y:y+win, x:x+win], guide[:, y:y+win, x:x+win]

    def __len__(self):
        return len(self.Guide_list)

    def __getitem__(self, idx):
        ratio = 65535.0
        maxx = ratio
        minn = 0.0

        Guide_data = self.load_rgb_tensor(self.Guide_list[idx])
        HQ_data = self.load_depth_tensor(self.HQ_list[idx], ratio)
        raw_depth = self.load_depth_tensor(self.LQ_list[idx], ratio)

        Guide_data = self.rgb_transform(Guide_data)
        HQ_data = self.depth_transform(HQ_data)
        raw_depth = self.depth_transform(raw_depth)
        
        mask = (raw_depth <= 0.001).float()
        LQ_data = HQ_data * (1.0 - mask)

        if not self.train:
            HQ_data = HQ_data * ratio

        minmax = torch.tensor([maxx, minn])

        if self.train:
            if self.aug_mode:
                HQ_data, LQ_data, Guide_data = self.argument(HQ_data, LQ_data, Guide_data)
            HQ_data, LQ_data, Guide_data = self.get_patch_random(HQ_data, LQ_data, Guide_data)

        return {
            'HQ': HQ_data,
            'Guide': Guide_data,
            'LQ': LQ_data,
            'minmax': minmax
        }
