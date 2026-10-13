import torch.utils.data as Data
import Dataset
import argparse
from Runner import make_model, Runner
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import torch
import random
import numpy as np
from torch.utils.data.distributed import DistributedSampler
import warnings

class WorkerInit:
    def __init__(self, rank):
        self.rank = rank
    
    def __call__(self, worker_id):
        seed = 42 + self.rank * 100 + worker_id
        np.random.seed(seed)
        random.seed(seed)

def setup(rank, world_size, port=12365):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size, args, port):
    setup(rank, world_size, port)
    set_seed(42)

    worker_init_fn = WorkerInit(rank)

    model, config = make_model(args)

    runner = Runner(model, args, config, rank)    


    train_target = str(config.get("train_target", "filled")).lower()
    TOFDC_data = config["datasets"]["test"]["TOFDC"]
    nyuma_data = config["datasets"]["test"]["NYU_MA"]
    nyudc_data = config["datasets"]["test"]["NYUDC"]
    diml_data = config["datasets"]["test"]["DIML"]
    sun_data = config["datasets"]["test"]["SUN"]


    if rank == 0:
        print('Loading dataset ...\n')

    kitti_cfg = config["datasets"].get("kittidc", {})

    def make_kittidc_dataset(mode, data_path=None, augment=False):
        data_path = data_path or kitti_cfg.get("path") or config["datasets"]["test"].get("KITTIDC")
        split_json = (
            kitti_cfg.get("split_json")
            or config["datasets"]["train"].get("split_json")
            or config["datasets"]["test"].get("KITTIDC_split_json")
        )
        if not data_path:
            raise ValueError("KITTIDC requires datasets.kittidc.path or datasets.test.KITTIDC in config.")
        if not split_json:
            raise ValueError("KITTIDC requires datasets.kittidc.split_json or datasets.train.split_json in config.")

        return Dataset.KITTIDC(
            path=data_path,
            split_json=split_json,
            mode=mode,
            patch_height=kitti_cfg.get("patch_height", config["datasets"]["train"].get("patch_height", 240)),
            patch_width=kitti_cfg.get("patch_width", config["datasets"]["train"].get("patch_width", 1216)),
            top_crop=kitti_cfg.get("top_crop", 100),
            augment=augment,
            num_sample=kitti_cfg.get("num_sample", 0),
            max_depth=kitti_cfg.get("max_depth", 90.0),
            test_crop=kitti_cfg.get("test_crop", False),
        )

    if train_target == "kittidc":
        kitti_test_mode = "val" if args.mode == "train" else kitti_cfg.get("test_mode", "test")
        dataset_kittidc_test = make_kittidc_dataset(kitti_test_mode, augment=False)
        kittidc_test_sampler = DistributedSampler(dataset_kittidc_test, num_replicas=world_size, rank=rank, shuffle=False)
        test_kittidc_loader = Data.DataLoader(
            dataset=dataset_kittidc_test,
            num_workers=4,
            batch_size=1,
            sampler=kittidc_test_sampler,
            worker_init_fn=worker_init_fn
        )
    else:
        dataset_tofdc_test = Dataset.Dataset_TOFDC(path=TOFDC_data, win = 256,aug_mode=False,train=False)
        tofdc_test_sampler = DistributedSampler(dataset_tofdc_test, num_replicas=world_size, rank=rank, shuffle=False)
        test_tofdc_loader = Data.DataLoader(dataset=dataset_tofdc_test, num_workers=4, batch_size=1, sampler=tofdc_test_sampler, worker_init_fn=worker_init_fn)

        dataset_nyuma_test = Dataset.nyu_ma(path=nyuma_data,mode="val",num_sample=500)
        nyuma_test_sampler = DistributedSampler(dataset_nyuma_test, num_replicas=world_size, rank=rank, shuffle=False)
        test_nyuma_loader = Data.DataLoader(dataset=dataset_nyuma_test, num_workers=4, batch_size=1, sampler=nyuma_test_sampler, worker_init_fn=worker_init_fn)
        dataset_nyudc_test = Dataset.NYU_DC(path=nyudc_data, win = 256,aug_mode=False,train=False)
        nyudc_test_sampler = DistributedSampler(dataset_nyudc_test, num_replicas=world_size, rank=rank, shuffle=False)
        test_nyudc_loader = Data.DataLoader(dataset=dataset_nyudc_test, num_workers=4, batch_size=1, sampler=nyudc_test_sampler, worker_init_fn=worker_init_fn)

        dataset_diml_test = Dataset.Dataset_DIML(path=diml_data, win=256, aug_mode=False, train=False)
        diml_test_sampler = DistributedSampler(dataset_diml_test, num_replicas=world_size, rank=rank, shuffle=False)
        test_diml_loader = Data.DataLoader(dataset=dataset_diml_test, num_workers=4, batch_size=1, sampler=diml_test_sampler, worker_init_fn=worker_init_fn)

        dataset_sun_test = Dataset.Dataset_SUN(path=sun_data, win=256, aug_mode=False, train=False)
        sun_test_sampler = DistributedSampler(dataset_sun_test, num_replicas=world_size, rank=rank, shuffle=False)
        test_sun_loader = Data.DataLoader(dataset=dataset_sun_test, num_workers=4, batch_size=1, sampler=sun_test_sampler, worker_init_fn=worker_init_fn)


    if args.mode == "train":

        train_data = config["datasets"]["train"]["path"]
        patch_size = config["datasets"]["train"]["patch_size"]
        aug_mode = config["datasets"]["train"]["aug_mode"]

        print("train_target:",train_target)
        if train_target == "tofdc":       
            dataset_train = Dataset.Dataset_TOFDC(path=train_data, win = patch_size,aug_mode=aug_mode,train=True)
        elif train_target == "nyu_ma":
            dataset_train = Dataset.nyu_ma(path = train_data,num_sample=500,mode="train")
        elif train_target == "nyu_dc":
            dataset_train = Dataset.NYU_DC(path=train_data, win = patch_size,aug_mode=aug_mode,train=True)
        elif train_target == "diml":
            dataset_train = Dataset.Dataset_DIML(path=train_data, win=patch_size, aug_mode=aug_mode, train=True)
        elif train_target == "sun":
            dataset_train = Dataset.Dataset_SUN(path=train_data, win=patch_size, aug_mode=aug_mode, train=True)
        elif train_target == "kittidc":
            dataset_train = make_kittidc_dataset("train", data_path=train_data, augment=aug_mode)

        else:    
            raise ValueError("error: wrong dataset!")


        train_sampler = DistributedSampler(dataset_train, num_replicas=world_size, rank=rank,
                                           shuffle=config["datasets"]["train"]["use_shuffle"])

        train_loader = Data.DataLoader(dataset=dataset_train, num_workers=4, batch_size=config["datasets"]["train"]["batch_size"],
                                       sampler=train_sampler, worker_init_fn=worker_init_fn)

        if train_target == "kittidc":
            test_loader_dict = {"KITTIDC": test_kittidc_loader}
        else:
            test_loader_dict = {
                                "nyu_ma":test_nyuma_loader,
                                "TOFDC": test_tofdc_loader,
                                "DIML": test_diml_loader,
                                "SUN": test_sun_loader
                                }
        if rank == 0:
            print("# of training samples: %d\n" % int(len(dataset_train)))

        runner.train(args,train_loader,test_loader_dict,train_target)

    else:


        if args.path == None and rank == 0:
            raise ValueError("The folder where the test weights are located needs to be provided!")
        if len((args.cuda).split(',')) > 1 and rank == 0:
            warnings.warn("Using more than one graphics card for testing may cause deviations in the average indicators.")
        if train_target == "kittidc":
            runner.test_kittidc(test_kittidc_loader, args.recode)
        else:
            runner.test_nyuma(test_nyudc_loader, args.recode)

    cleanup()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cuda", type=str, default="0", help="gpu to train")
    parser.add_argument("-r", "--recode",default = True, help="choose whether to recode", action="store_true")
    parser.add_argument("-p", "--path", type=str, default=None, help="pre weight path")
    parser.add_argument("-cfg", "--config", type=str, default=None, help="config file path")
    parser.add_argument("-m", "--mode", type=str, default="train", choices=["train","test"], help="choose to train or test")

    args = parser.parse_args()
    cuda_ids = [item.strip() for item in args.cuda.split(",") if item.strip() != ""]
    if len(cuda_ids) == 0:
        raise ValueError("--cuda must specify at least one GPU id, e.g. -c 0")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 按PCI总线ID排序，确保GPU顺序与它们在物理硬件中的位置一致。
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_ids)

    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count == 0:
        raise RuntimeError("No CUDA devices available. Check your drivers and CUDA_VISIBLE_DEVICES.")
    if visible_gpu_count < len(cuda_ids):
        raise RuntimeError(
            f"Requested {len(cuda_ids)} GPU(s) via --cuda but only {visible_gpu_count} are visible."
        )

    WORLD_SIZE = len(cuda_ids)
    
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        port = s.getsockname()[1]

    mp.spawn(main, args=(WORLD_SIZE, args, port), nprocs=WORLD_SIZE, join=True)
