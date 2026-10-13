import cv2
import time
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import logging
import yaml
import shutil
from util import padding, inv_padding
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch
import torch.nn.functional as F
import numpy as np


import Loss.loss as loss
import network.my_model_v10 as My_model

def remove_module_dict(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        
        if key.startswith("module."):
            k = key.replace("module.", "")
        else:
            k = key
        new_state_dict[k] = state_dict[key]
    return new_state_dict

def make_model(args):
    if args.config:
        config_path = args.config
    elif args.path:
        config_path = os.path.join(args.path, "config.yml")

    else:
        config_path = "config.yml"
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    Train = True if args.mode == "train" else False
    model = My_model.UnifiedDepthNet(config["network"], Train)
    return model,config


class Runner:
    def __init__(self, model, args, config, rank):
        device_count = torch.cuda.device_count()
        if device_count == 0:
            raise RuntimeError("CUDA not available. Check your drivers and CUDA_VISIBLE_DEVICES.")
        if rank >= device_count:
            raise RuntimeError(
                f"Invalid local rank {rank} for {device_count} visible GPU(s). "
                "Check --cuda and WORLD_SIZE settings."
            )
        torch.cuda.set_device(rank)
        self.config = config
        self.rank = rank

        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(self.rank)

        if args.path:
            print("Loading checkpoint...")
            self.load_checkpoint(args)
        else:
            print("Dont load checkpoint...")
        self.model = DDP(self.model, find_unused_parameters=True)
        if args.recode:
            self.initial_train_recode(args) if args.mode == "train" else self.initial_test_recode(args)
        
        self.best_checkpoints = [] # List of (score, path) for top 5 checkpoints

    def make_optimizer(self):
        lr = self.config["train"]["optim"]["init_lr"]
        wd = self.config["train"]["optim"]["weight_decay"]
        bs = self.config["train"]["optim"]["betas"]
        if self.config["train"]["optim"]["type"] == "AdamW":
            return torch.optim.AdamW(self.model.parameters(), lr = lr, weight_decay = wd,betas = bs)
        else:
            return torch.optim.Adam(self.model.parameters(), lr = lr, weight_decay = wd,betas = bs)
        
    def make_loss(self):
        type = self.config["train"]["loss_type"]
        if type == "mse":
            return nn.MSELoss()
        elif type == "mix1":
            return loss.MIX1Loss()
        elif type == "mix2":
            return loss.MIX2Loss()
        elif type == "mix3":
            return loss.MIX3Loss()#.to(self.rank)
        elif type == "Charbonnier":
            return loss.CharbonnierLoss()
        elif type == "l1":
            return nn.L1Loss()

    def make_scheduler(self,optimizer,current_idx,train_loader):
        type = self.config["train"]["optim"]["scheduler_type"]
        if type == "linear":
            end_factor = self.config["train"]["optim"]["final_lr"] / self.config["train"]["optim"]["init_lr"]
            return torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=end_factor,
                                                     total_iters=len(train_loader) * self.config["train"]["epoch"],
                                                     last_epoch=current_idx - 1)
        elif type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * self.config["train"]["epoch"],
                                                              eta_min=self.config["train"]["optim"]["final_lr"],last_epoch = current_idx - 1)


    def initial_test_recode(self,args):
        self.save_path = os.path.join(args.path, ("Test-" + str(self.checkpoints["current_idx"])))
        if self.rank == 0:
            os.makedirs(self.save_path,exist_ok=True)
            logging.basicConfig(filename=os.path.join(self.save_path, "test_recode.log"),
                                format='%(asctime)s %(message)s', level=logging.INFO)
        else:
            logging.disable(logging.CRITICAL)


    def initial_train_recode(self,args):
        timestr = time.strftime("%Y%m%d-%H%M%S")[4:]

        if args.path:
            self.save_path = args.path
            recode_path = os.path.join(self.save_path, ("finetune-" + timestr))
            if self.rank == 0:
                os.makedirs(recode_path, exist_ok=True)
                shutil.copy(os.path.join(self.save_path, "config.yml"),
                            os.path.join(recode_path, "config.yml"))
        else:
            self.save_path = os.path.join("Recode", ("recode-" + self.config["Version"] + "-" + timestr))
            recode_path = os.path.join(self.save_path, "First_time")
            if self.rank == 0:
                os.makedirs(recode_path, exist_ok=True)
                shutil.copy("config.yml", os.path.join(self.save_path, "config.yml"))
                shutil.copy("config.yml", os.path.join(recode_path, "config.yml"))

        if self.rank == 0:
            self.writer = SummaryWriter(log_dir=recode_path)
            logging.basicConfig(filename=os.path.join(recode_path, "train_recode.log"),
                                format='%(asctime)s %(message)s',level=logging.INFO)
            logging.info(self.config)

        else:
            self.writer = None
            logging.disable(logging.CRITICAL)

    def load_checkpoint(self, args):
        path = args.path
        recoder = "finetune : {}".format(path)
        self.checkpoints = torch.load(os.path.join(path, self.config["train"]["load"]["model"]))
        if self.rank == 0:
            print(f"Loading checkpoint from {os.path.join(path, self.config['train']['load']['model'])}")
            self.model.load_state_dict(remove_module_dict(self.checkpoints["model_state_dict"]))
            print("Model weights loaded successfully!")

    def save_checkpoint(self,*args):
        if self.rank == 0:
            checkpoint = {"current_epoch": args[0],
                          "current_idx": args[1],
                          "model_state_dict": args[2],
                          "optimizer_state_dict": args[3]}
            torch.save(checkpoint, args[4])

    def update_best_checkpoints(self, score, epoch, current_idx, optimizer):
        ckpt_name = os.path.join(self.save_path, 'model_{}_rmse_{:.4f}.pth'.format(current_idx, score))
        
        qualified = False
        if len(self.best_checkpoints) < 5:
            qualified = True
        elif score < self.best_checkpoints[-1][0]:
            qualified = True
            
        if qualified:
            logging.info(f"Saving new best checkpoint: {ckpt_name} (RMSE: {score:.4f})")
            self.save_checkpoint(epoch, current_idx, self.model.state_dict(), optimizer.state_dict(), ckpt_name)
            self.best_checkpoints.append((score, ckpt_name))
            self.best_checkpoints.sort(key=lambda x: x[0])
            
            if len(self.best_checkpoints) > 5:
                worst_score, worst_path = self.best_checkpoints.pop(-1)
                if os.path.exists(worst_path):
                    try:
                        os.remove(worst_path)
                        logging.info(f"Removed old checkpoint: {worst_path} (RMSE: {worst_score:.4f})")
                    except OSError as e:
                        logging.error(f"Error removing checkpoint {worst_path}: {e}")

    def save_img(self,img, folder, name):
            folder = os.path.join(self.save_path, folder)
            os.makedirs(folder, exist_ok=True)
            cv2.imwrite(os.path.join(folder, name), img)


    def average_loss(self, loss):
        loss = loss.float() 
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss /= dist.get_world_size() # 浮点数除法
        return loss.item()

    def average_metrice(self, tmp_results):
        all_results = {}
        for key in tmp_results:
            local_data = torch.tensor(tmp_results[key], dtype=torch.float32, device=self.rank)
            all_data = [torch.zeros_like(local_data) for _ in range(dist.get_world_size())]

            dist.all_gather(all_data, local_data)
            gathered_data = torch.cat(all_data)
            global_mean = gathered_data.mean().item()
            all_results[key] = global_mean

        return all_results


    def train(self,args,train_loader,test_loader_dict,train_target):
        optimizer = self.make_optimizer()
        loss_function = self.make_loss()

        if args.path and self.config["train"]["load"]["inherit"]:
            optimizer.load_state_dict(self.checkpoints["optimizer_state_dict"])
            current_epoch = self.checkpoints["current_epoch"]
            current_idx = self.checkpoints["current_idx"]
        else:
            current_epoch = 0
            current_idx = 0

        scheduler = self.make_scheduler(optimizer,current_idx,train_loader)
        last_epoch = self.config["train"]["epoch"] - current_epoch


        self.model.train()
        for epoch_idx in range(last_epoch):
            epoch = epoch_idx + current_epoch
            total_loss = 0
            total_loss_1 = 0
            total_loss_2 = 0
            total_loss_mask = 0

            train_loader.sampler.set_epoch(epoch)
            batch_list = tqdm(train_loader) if self.rank == 0 else train_loader

            for input_data in batch_list:
                current_lr = round(optimizer.param_groups[0]["lr"], 7)

                output,maskloss,loss_2 = self.model(input_data["LQ"].to(self.rank), input_data["Guide"].to(self.rank))
                loss_2 = torch.mean(loss_2)
                optimizer.zero_grad()
                loss_1 = loss_function(output, input_data["HQ"].to(self.rank))

                loss = loss_1 + 0.01 * loss_2 + 0.1 * maskloss.to(self.rank)
                if not torch.isfinite(loss):
                    if self.rank == 0:
                        logging.warning("Non-finite loss detected. Skipping this batch.")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                total_loss += loss.data.item()
        
                total_loss_1 += loss_1.data.item()
                total_loss_mask += maskloss.data.item()
                total_loss_2 += loss_2.data.item()

                loss.backward()

                if self.config["train"]["clip_grad"]:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.01)
                optimizer.step()
                scheduler.step()
                current_idx += 1

                if self.rank == 0:
                    batch_list.set_description("epoch:%d iter:%d loss:%.4f loss_1:%.4f mask:%.4f loss_2:%.4f lr:%.6f"%
                                               (epoch + 1,current_idx, loss.data.item(),loss_1.data.item(),maskloss.data.item(),loss_2.data.item(),current_lr))

                if (current_idx) % self.config["val"]["freq"] == 0:
                    results =self.test_SUN(test_loader_dict["SUN"],True)

                    if args.recode and self.rank == 0:
                         if "RMSE" in results:
                             avg_rmse = sum(results["RMSE"])/len(results["RMSE"])
                             self.update_best_checkpoints(avg_rmse, epoch, current_idx, optimizer)


                if args.recode and self.rank == 0:
                    self.writer.add_scalar('loss', loss.data.item(), current_idx)
                    if current_idx % self.config["save"]["auto_freq"] == 0:
                        name = os.path.join(self.save_path, 'model_current.pth')
                        self.save_checkpoint(epoch, current_idx, self.model.state_dict(), optimizer.state_dict(), name)

            if args.recode:
                avg_loss = self.average_loss(torch.tensor(total_loss).cuda(self.rank)) / len(train_loader)
                avg_loss_1 = self.average_loss(torch.tensor(total_loss_1).cuda(self.rank)) / len(train_loader)
                avg_loss_mask = self.average_loss(torch.tensor(total_loss_mask).cuda(self.rank)) / len(train_loader)
                avg_loss_2 = self.average_loss(torch.tensor(total_loss_2).cuda(self.rank)) / len(train_loader)

                logging.info("[idx %d][epoch %d] ave_loss: %.4f ave_loss_1: %.4f ave_loss_mask: %.4f ave_loss_2: %.4f learning_rate: %.6f" % (
                        current_idx + 1, epoch + 1, avg_loss, avg_loss_1, avg_loss_mask, avg_loss_2,  current_lr))


    def test_iter(self,input_data):
        LQ, pad = padding(8, input_data["LQ"])
        Guide, _ = padding(8, input_data["Guide"])

        start_time = time.perf_counter()
        output,_,_ = self.model(LQ.to(self.rank), Guide.to(self.rank))
        end_time = time.perf_counter()
        output = inv_padding(pad,output)
        cal_time = end_time - start_time
        return output,cal_time

    def test_TOFDC(self, test_loader, save_img=False):
        if self.rank == 0:
            print("======= test_TOFDC ========")
            logging.info("======= test_TOFDC ========")

        self.model.eval()

        
        metric_names = ['RMSE', 'MAE', 'iRMSE', 'iMAE', 'REL', 'D^1', 'D^2', 'D^3']

        tmp_results = {m: [] for m in metric_names}
        tmp_results['Time'] = []

        with torch.no_grad():
            batch_list = tqdm(test_loader) if self.rank == 0 else test_loader

            for batch_idx, input_data in enumerate(batch_list):
                output, cal_time = self.test_iter(input_data)
                HQ = input_data["HQ"].to(self.rank)
                LQ = input_data["LQ"].to(self.rank)
                minmax = input_data["minmax"].squeeze(0).to(self.rank)

                if save_img:
                    idx = batch_idx * dist.get_world_size() + self.rank
                    self.save_img(
                        img=np.uint8(output[0, 0].cpu().numpy() * 255),
                        folder="TOFDC/output",
                        name=f"{idx:03d}.png"
                    )

                    self.save_img(
                        img=((HQ[0, 0] - minmax[1]) / (minmax[0] - minmax[1]) * 255).cpu().numpy().astype(np.uint8),
                        folder="TOFDC/gt",
                        name=f"{idx:03d}.png"
                    )
                    self.save_img(
                        img=np.uint8(LQ[0, 0].cpu().numpy() * 255),
                        folder="TOFDC/LQ",
                        name=f"{idx:03d}.png"
                    )

                pred = output[0, 0] * (minmax[0] - minmax[1]) + minmax[1]
                gt = HQ[0, 0]
                

                metrics = self.evaluate_batch(gt, pred,0.001).squeeze(0)

                tmp_results['Time'].append(cal_time)
                for i, m in enumerate(metric_names):
                    tmp_results[m].append(metrics[i].item())

                if metrics[4] > 0.1:  # 或你觉得异常的阈值
                    print(f"REL explosion at image {batch_idx}")

                    print(f"gt min/max: {gt.min().item():.6f} / {gt.max().item():.6f}")
                    print(f"pred min/max: {pred.min().item():.6f} / {pred.max().item():.6f}")

                    rel_map = torch.abs(pred - gt) / gt
                    print(f"REL map max: {rel_map.max().item():.2f}")

                if self.rank == 0:
                    batch_list.set_description(f"REL:{metrics[4].item():.3f}")

        if self.rank == 0:
            print("\n======= TOFDC Results (Average) =======")
            for m in metric_names:
                avg_v = torch.tensor(tmp_results[m]).mean().item()
                print(f"{m}: {avg_v:.4f}")
                logging.info(f'{m} metric value: {avg_v:.4f}')

            avg_t = sum(tmp_results["Time"]) / len(tmp_results["Time"])
            print(f"Time: {avg_t:.4f}")

        return tmp_results

    def test_DIML(self, test_loader, save_img=False):
        if self.rank == 0:
            print("======= test_DIML ========")
            logging.info("======= test_DIML ========")

        self.model.eval()

        metric_names = ['RMSE', 'MAE', 'iRMSE', 'iMAE', 'REL', 'D^1', 'D^2', 'D^3']
        tmp_results = {m: [] for m in metric_names}
        tmp_results['Time'] = []

        with torch.no_grad():
            batch_list = tqdm(test_loader) if self.rank == 0 else test_loader

            for batch_idx, input_data in enumerate(batch_list):
                output, cal_time = self.test_iter(input_data)
                HQ = input_data["HQ"].to(self.rank)
                LQ = input_data["LQ"].to(self.rank)
                minmax = input_data["minmax"].squeeze(0).to(self.rank)

                if save_img:
                    idx = batch_idx * dist.get_world_size() + self.rank
                    self.save_img(
                        img=np.uint8(output[0, 0].cpu().numpy() * 255),
                        folder="DIML/output",
                        name=f"{idx:03d}.png"
                    )
                    self.save_img(
                        img=((HQ[0, 0] - minmax[1]) / (minmax[0] - minmax[1]) * 255).cpu().numpy().astype(np.uint8),
                        folder="DIML/gt",
                        name=f"{idx:03d}.png"
                    )
                    self.save_img(
                        img=np.uint8(LQ[0, 0].cpu().numpy() * 255),
                        folder="DIML/LQ",
                        name=f"{idx:03d}.png"
                    )

                pred = output[0, 0] * (minmax[0] - minmax[1]) + minmax[1]
                gt = HQ[0, 0]

                metrics = self.evaluate_batch(gt, pred, 1).squeeze(0)

                tmp_results['Time'].append(cal_time)
                for i, m in enumerate(metric_names):
                    tmp_results[m].append(metrics[i].item())

                if self.rank == 0:
                    batch_list.set_description(f"RMSE:{metrics[0].item():.3f}")

        if self.rank == 0:
            print("\n======= DIML Results (Average) =======")
            for m in metric_names:
                avg_v = torch.tensor(tmp_results[m]).mean().item()
                print(f"{m}: {avg_v:.4f}")
                logging.info(f'{m} metric value: {avg_v:.4f}')

            avg_t = sum(tmp_results["Time"]) / len(tmp_results["Time"])
            print(f"Time: {avg_t:.4f}")

        return tmp_results

    def test_SUN(self, test_loader, save_img=False):
        if self.rank == 0:
            print("======= test_SUN ========")
            logging.info("======= test_SUN ========")

        self.model.eval()

        metric_names = ['RMSE', 'MAE', 'iRMSE', 'iMAE', 'REL', 'D^1', 'D^2', 'D^3']
        tmp_results = {m: [] for m in metric_names}
        tmp_results['Time'] = []

        with torch.no_grad():
            batch_list = tqdm(test_loader) if self.rank == 0 else test_loader

            for batch_idx, input_data in enumerate(batch_list):
                output, cal_time = self.test_iter(input_data)
                HQ = input_data["HQ"].to(self.rank)
                LQ = input_data["LQ"].to(self.rank)
                minmax = input_data["minmax"].squeeze(0).to(self.rank)

                if save_img:
                    idx = batch_idx * dist.get_world_size() + self.rank
                    self.save_img(
                        img=np.uint8(output[0, 0].cpu().numpy() * 255),
                        folder="SUN/output",
                        name=f"{idx:03d}.png"
                    )
                    self.save_img(
                        img=((HQ[0, 0] - minmax[1]) / (minmax[0] - minmax[1]) * 255).cpu().numpy().astype(np.uint8),
                        folder="SUN/gt",
                        name=f"{idx:03d}.png"
                    )
                    self.save_img(
                        img=np.uint8(LQ[0, 0].cpu().numpy() * 255),
                        folder="SUN/LQ",
                        name=f"{idx:03d}.png"
                    )

                pred = (output[0, 0] * (minmax[0] - minmax[1]) + minmax[1])/10
                gt = HQ[0, 0]/10

                metrics = self.evaluate_batch(gt, pred, 1).squeeze(0)

                tmp_results['Time'].append(cal_time)
                for i, m in enumerate(metric_names):
                    tmp_results[m].append(metrics[i].item())

                if self.rank == 0:
                    batch_list.set_description(f"RMSE:{metrics[0].item():.3f}")

        if self.rank == 0:
            print("\n======= SUN Results (Average) =======")
            for m in metric_names:
                avg_v = torch.tensor(tmp_results[m]).mean().item()
                print(f"{m}: {avg_v:.4f}")
                logging.info(f'{m} metric value: {avg_v:.4f}')

            avg_t = sum(tmp_results["Time"]) / len(tmp_results["Time"])
            print(f"Time: {avg_t:.4f}")

        return tmp_results

    def test_nyuma(self, test_loader, save_img=False):
        if self.rank == 0:
            print("======= test_nyuma ========")
            logging.info("======= test_nyuma ========")

        self.model.eval()

        metric_names = ['RMSE', 'MAE', 'iRMSE', 'iMAE', 'REL', 'D^1', 'D^2', 'D^3']

        tmp_results = {m: [] for m in metric_names}
        tmp_results['Time'] = []
        tmp_results['RMSE_on_M'] = []
        tmp_results['RMSE_off_M'] = []
        tmp_results['RMSE_ratio'] = []

        with torch.no_grad():
            batch_list = tqdm(test_loader) if self.rank == 0 else test_loader

            for batch_idx, input_data in enumerate(batch_list):
                output, cal_time = self.test_iter(input_data)
                HQ = input_data["HQ"].to(self.rank)
                LQ = input_data["LQ"].to(self.rank)
                minmax = input_data["minmax"].squeeze(0).to(self.rank)


                pred = output[0, 0]*10

                gt = HQ[0, 0]

                M_mask = (LQ[0, 0] > 0.001).float()
                M_mask_hr = F.interpolate(M_mask.unsqueeze(0).unsqueeze(0),
                                          size=gt.shape[-2:], mode='nearest')[0, 0]
                sq_err = (pred - gt) ** 2
                n_on = M_mask_hr.sum().clamp(min=1)
                n_off = (1 - M_mask_hr).sum().clamp(min=1)
                rmse_on_M = torch.sqrt((sq_err * M_mask_hr).sum() / n_on).item()
                rmse_off_M = torch.sqrt((sq_err * (1 - M_mask_hr)).sum() / n_off).item()
                rmse_ratio = rmse_on_M / (rmse_off_M + 1e-8)

                tmp_results['RMSE_on_M'].append(rmse_on_M)
                tmp_results['RMSE_off_M'].append(rmse_off_M)
                tmp_results['RMSE_ratio'].append(rmse_ratio)

                if save_img:
                    idx = batch_idx * dist.get_world_size() + self.rank
                    self.save_img(
                        img=np.uint8(output[0, 0].cpu().numpy() * 255),
                        folder="NYU_MA/output",
                        name=f"{idx:03d}.png"
                    )

                    self.save_img(
                        img=np.uint8(HQ[0, 0].cpu().numpy() * 255/10),
                        folder="NYU_MA/gt",
                        name=f"{idx:03d}.png"
                    )
                    lq_val =  LQ[0, 0]
                    lq_norm = lq_val  * 255
                    lq_norm = torch.clamp(lq_norm, 0, 255)
                    self.save_img(
                        
                        img=lq_norm.cpu().numpy().astype(np.uint8),
                        folder="NYU_MA/LQ",
                        name=f"{idx:03d}.png"
                    )
                metrics = self.evaluate_batch(gt, pred, 1e-4).squeeze(0)

                tmp_results['Time'].append(cal_time)
                for i, m in enumerate(metric_names):
                    tmp_results[m].append(metrics[i].item())

                if self.rank == 0:
                    batch_list.set_description(f"RMSE:{metrics[0].item():.3f}")

        if self.rank == 0:
            print("\n======= NYU_MA Results (Average) =======")
            for m in metric_names:
                avg_v = torch.tensor(tmp_results[m]).mean().item()
                print(f"{m}: {avg_v:.4f}")
                logging.info(f'{m} metric value: {avg_v:.4f}')

            avg_t = sum(tmp_results["Time"]) / len(tmp_results["Time"])
            print(f"Time: {avg_t:.4f}")

            avg_rmse_on = torch.tensor(tmp_results['RMSE_on_M']).mean().item()
            avg_rmse_off = torch.tensor(tmp_results['RMSE_off_M']).mean().item()
            avg_ratio = torch.tensor(tmp_results['RMSE_ratio']).mean().item()
            print(f"RMSE_on_M (avg): {avg_rmse_on:.4f}")
            print(f"RMSE_off_M (avg): {avg_rmse_off:.4f}")
            print(f"RMSE_on/off ratio (avg): {avg_ratio:.4f}")
            logging.info(f'RMSE_on_M (avg): {avg_rmse_on:.4f}')
            logging.info(f'RMSE_off_M (avg): {avg_rmse_off:.4f}')
            logging.info(f'RMSE_on/off ratio (avg): {avg_ratio:.4f}')

        return tmp_results


    def evaluate_batch(self, gt, pred,min):
        if not isinstance(gt, torch.Tensor):
            gt = torch.from_numpy(gt)
        if not isinstance(pred, torch.Tensor):
            pred = torch.from_numpy(pred)

        mask = (gt > min)
        num_valid = mask.sum()

        pred = pred[mask]
        gt = gt[mask]

        diff = pred - gt
        diff_abs = torch.abs(diff)
        diff_sqr = diff ** 2

        rmse = torch.sqrt(diff_sqr.sum() / (num_valid + 1e-8))
        mae = diff_abs.sum() / (num_valid + 1e-8)

        inv_pred = 1.0 / (pred + 1e-8)
        inv_gt = 1.0 / (gt + 1e-8)

        inv_diff = inv_pred - inv_gt
        inv_diff_abs = torch.abs(inv_diff)
        inv_diff_sqr = inv_diff ** 2

        irmse = torch.sqrt(inv_diff_sqr.sum() / (num_valid + 1e-8))
        imae = inv_diff_abs.sum() / (num_valid + 1e-8)

        rel = diff_abs / (gt + 1e-8)
        rel = rel.sum() / (num_valid + 1e-8)

        r1 = gt / (pred + 1e-8)
        r2 = pred / (gt + 1e-8)
        ratio = torch.max(r1, r2)

        del_1 = (ratio < 1.25).float().sum() / (num_valid + 1e-8)
        del_2 = (ratio < 1.25**2).float().sum() / (num_valid + 1e-8)
        del_3 = (ratio < 1.25**3).float().sum() / (num_valid + 1e-8)

        result = [
            rmse,      # 0
            mae,       # 1
            irmse,     # 2
            imae,      # 3
            rel,       # 4
            del_1,     # 5
            del_2,     # 6
            del_3      # 7
        ]

        result = torch.stack(result)
        result = result.unsqueeze(0).detach()

        return result
