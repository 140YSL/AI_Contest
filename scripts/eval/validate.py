import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import os
from torchvision.io import read_image  # 用来读取图片的宽和高
import cv2

from datasets.dataset_rgbd_text import RGBDTextDataset
from models.tracker_model import MultiModalTracker
from torch.utils.data import DataLoader
from eval.evaluate import Evaluate
from utils.load_save import Load_and_Save
from show.draw import Show_to_User

class Validate:

    @staticmethod
    def validate(model_path, validation_dataset=None, validation_path=None, seq_used=None, seq_start=None, pred_save=False, batch_size=1):
        '''
        进入验证模式评估模型的性能，输入一个模型的存储地址，会从该地址的pls文件中读取模型，
        两种输入模式：1.需要验证的数据集dataset
        2.需要验证的文件的路径，使用的序列开端和序列长度
        :param model_path: 模型路径
        :param validation_dataset: 模式一输入数据集类型dataset
        :param validation_path: 模式二指定需要哦验证的文件路径
        :param seq_used: 模式二指定使用这个路径中多少的序列
        :param seq_start: 模式二指定使用这个路径中的从第几个开始的序列
        :param pred_save: 是否保存这次预测的结果，也会储存在pred_bbox.txt中，需要注意有覆盖的问题，启用后可能覆盖训练过程中保存的结果
        :param batch_size: 指定一次验证的批大小，没什么卵用
        :return: 返回预测方框的误差，总体平均的IoU
        '''
        print(f"使用模型：{model_path}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = MultiModalTracker()
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model = model.to(device)  #加载模型并且移动到设备上，默认设备是gpu

        model.eval()
        loss_list = []
        iou_list = []
        if validation_dataset is not None:
            val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
        elif validation_path is not None:
            validation_dataset = RGBDTextDataset(root_dir=validation_path, seq_used=seq_used, seq_used_start=seq_start)  # 加载数据集
            val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
        else:
            raise RuntimeError("验证模式输入不属于任何一种")

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                rgb = batch["rgb"].to(device)          # [B,T,3,H,W]
                depth = batch["depth"].to(device)      # [B,T,1,H,W]
                text = batch["text"]
                gt_boxes = batch["gt_boxes"].to(device) # [B,T,4]
                mask = batch["cls"].to(device)         # [B,T]
                init_box = batch["init_box"].to(device)  # [B,4] in resized pixels
                seq_dir = batch["seq_dir"]
                name = [s.split("\\")[-1] for s in seq_dir]
                B, T, _, H, W = rgb.shape

                # 构造标签
                cls_target = mask  # [B,T,1]
                bbox_target = gt_boxes  # [B,T,4]

                # === 模型前向 ===
                cls_pred, bbox_pred = model(rgb, depth, text, init_box) # bbox_pred[B,T,4]
                # Show_to_User.draw_cls_preatar_batch(cls_target, cls_pred, name=name, show_time=1)
                if pred_save: Load_and_Save.save_all_bbox_pred(seq_dir, bbox_pred)

                loss, bbox_loss, cls_loss = Evaluate.cal_smooth_loss_weight( cls_target, cls_pred, bbox_pred, bbox_target)
                if B == 1:
                    loss_list.append(loss.item())
                else:
                    loss_list.extend(loss.detach().cpu().tolist())

                # === IoU ===
                for b in range(B):
                    iou_frame = Evaluate.cal_IoU_actual(bbox_pred[b], bbox_target[b])
                    # show_num = 10
                    # color_path = os.path.join(batch["seq_dir"][b], "color")
                    # pic_path = os.path.join(color_path, os.listdir(color_path)[show_num])
                    # draw_train_vali_pic(pic_path, bbox_target=bbox_target[b][show_num], bbox_pred=bbox_pred[b][show_num])
                    iou_list.extend(iou_frame.detach().cpu().tolist())

        avg_loss = sum(loss_list) / len(loss_list)
        avg_iou = sum(iou_list) / len(iou_list)

        print(f"验证结果：Loss={avg_loss:.4f} | IoU={avg_iou:.4f}")
        return avg_loss, avg_iou

# # # ===== 测试部分 =====
if __name__ == "__main__":
    num_epochs = 50  # 训练轮数
    batch_size = 1  # 每次训练样本数量
    learning_rate = 1e-4  # 学习率
    train_seq_used = 22
    train_seq_start = 138
    train_seq_len = 61
    validation_seq_used = 1
    epochs_stop_rate = None
    save_pred = True

    save_dir = Path(os.path.abspath(__file__)).resolve().parent.parent.parent.joinpath("checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ============ 数据集 ============
    BASE_DATA_DIR = Path(os.path.abspath(__file__)).resolve().parent.parent.parent.joinpath("data")
    MODEL_PATH = save_dir.joinpath("epoch_50.pth")
    BASE_TRAINSET_DIR = str(BASE_DATA_DIR.joinpath("TrainSet"))
    BASE_VALIDATIONSET_DIR = str(BASE_DATA_DIR.joinpath("ValidationSet"))  # 训练集地址
    train_dataset = RGBDTextDataset(root_dir=BASE_TRAINSET_DIR, seq_used=train_seq_used, seq_used_start=train_seq_start,
                                    seq_len=train_seq_len)  # 加载数据集
    # validation_dataset = RGBDTextDataset(root_dir=BASE_VALIDATIONSET_DIR, seq_used=validation_seq_used,
    #                                      seq_len=None)  # 加载数据集
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 划分训练批次
#     validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
#     model = MultiModalTracker()
#     state_dict = torch.load(MODEL_PATH, map_location=device)
#     model.load_state_dict(state_dict)
#     model = model.to(device)
#
    # Validate.validate(os.path.join(save_dir, f"epoch_50.pth"), validation_dataset, batch_size=1, pred_save=True)
    from show.draw import Show_to_User
    # Vali_AUC = []
    # for i in validation_dataset:
    #     Show_to_User.draw_seq_pic_local(i["seq_dir"], show_time=10)
    #     Vali_AUC.append(Evaluate.cal_AUC_local(i["seq_dir"]))
    #     print(i["seq_dir"].split("data")[-1], ":", Vali_AUC[-1])
    # Vali_AUC = np.array(Vali_AUC)
#     AUC = []
#     for i in validation_dataset:
#         draw_seq_pic(i["seq_dir"], show_time=10)
#         AUC.append(cal_Success_Rate(i["seq_dir"]))
#         print(i["seq_dir"], ":", AUC[-1])
#     AUC = np.array(AUC)
#     print("验证集成功率：", AUC.mean())
#
    AUC = []
    for i in train_dataset:
        AUC.append(Evaluate.cal_SR_local(i["seq_dir"]))
        Show_to_User.draw_seq_pic_local(i["seq_dir"], show_time=10)
        print(i["seq_dir"], ":", AUC[-1])
    AUC = np.array(AUC)
    print("训练集成功率：", AUC.mean())

