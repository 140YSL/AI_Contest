'''
评估
'''

import numpy as np
from utils.load_save import Load_and_Save
from show.draw import Show_to_User
import torch
import torch.nn as nn # 提供神经网络层，损失函数等功能
import torch.nn.functional as F

class Evaluate:
    @staticmethod
    def cal_SR_local(seq_dir, save_local=False, show_time=None):
        '''
        利用本地之前的运行结果计算SR值
        :param seq_dir: 序列路径
        :param save_local: 保存到本地seq_dir
        :param show_time: 展示时间，None:不展示，0：一直展示，正数:显示时间
        :return:
        '''
        traget_bbox = Load_and_Save.load_target_norm_center(seq_dir)
        pred_bbox = Load_and_Save.load_pred_norm_center(seq_dir)
        IoU = Evaluate.cal_IoU_actual(traget_bbox, pred_bbox)
        IoU = np.array(IoU, dtype=np.float32)
        SR = Evaluate.cal_ToU2SR_actual(IoU)

        if save_local:
            Show_to_User.draw_SR(SR, save_path=seq_dir, show_time=show_time)
        else:
            Show_to_User.draw_SR(SR, show_time=show_time)
        return SR

    @staticmethod
    def cal_AUC_local(seq_dir, SR_save_local=False, SR_show_time=None):
        '''
        利用本地之前的运行结果计算AUC值
        :param seq_dir: 序列路径
        :param SR_save_local: 保存到本地seq_dir
        :param SR_show_time: 展示时间，None:不展示，0：一直展示，正数:显示时间
        :return:
        '''
        SR = Evaluate.cal_SR_local(seq_dir, save_local=SR_save_local, show_time=SR_show_time)
        AUC = SR[:-1].mean()
        return AUC

    @staticmethod
    def cal_ToU2SR_actual(IoU):
        '''
        通过IoU,[T]实时计算SR
        :param IoU: [T]
        :return: np.array,[21]
        '''
        theta = np.linspace(0, 1, 21)
        SR = []

        T = len(IoU)
        for th in theta:
            u_t = (IoU > th).astype(np.float32)  # 逐帧比
            SR.append((np.sum(u_t) / T))  # 平均值
        SR = np.array(SR)
        return SR

    @staticmethod
    def cal_SR_actual(target, pred, mask=None, eps=1e-6):
        '''
        根据target和pred直接计算SR
        :param target: 真实目标值[T, 4]
        :param pred:  预测值[T, 4]
        :param mask:  目标是否存在[T], True or False,若为None没输入会自动由target判断
        :param eps: 零值
        :return: [T],0-1
        '''
        IoU = Evaluate.cal_IoU_actual(target, pred, mask=mask, eps=eps)
        SR = Evaluate.cal_ToU2SR_actual(IoU)
        return SR

    @staticmethod
    def cal_IoU_actual(target, pred, mask=None, eps=1e-6):
        '''
        根据target和pred直接计算IoU
        :param target: 真实目标值[T, 4]
        :param pred:  预测值[T, 4]
        :param mask:  目标是否存在[T], True or False,若为None没输入会自动由target判断
        :param eps: 零值
        :return: [T],0-1
        '''
        if not torch.is_tensor(target):
            target = torch.as_tensor(target, dtype=torch.float32)
        if not torch.is_tensor(pred):
            pred = torch.as_tensor(pred, dtype=torch.float32)

            # 支持在 CPU 或 GPU 上计算
        device = target.device if target.is_cuda else 'cpu'
        pred = pred.to(device)

        if mask == None:
            mask = ~(target.abs().sum(dim=1) < 1e-6)  # 预测是否全零
        pred_mask = ~(pred.abs().sum(dim=1) < 1e-6)

        x1_min = target[:, 0] - 0.5 * target[:, 2]
        y1_min = target[:, 1] - 0.5 * target[:, 3]
        x1_max = target[:, 0] + 0.5 * target[:, 2]
        y1_max = target[:, 1] + 0.5 * target[:, 3]

        x2_min = pred[:, 0] - 0.5 * pred[:, 2]
        y2_min = pred[:, 1] - 0.5 * pred[:, 3]
        x2_max = pred[:, 0] + 0.5 * pred[:, 2]
        y2_max = pred[:, 1] + 0.5 * pred[:, 3]

        inter_xmin = torch.max(x1_min, x2_min)
        inter_ymin = torch.max(y1_min, y2_min)
        inter_xmax = torch.min(x1_max, x2_max)
        inter_ymax = torch.min(y1_max, y2_max)

        inter_w = torch.clamp(inter_xmax - inter_xmin, min=0)
        inter_h = torch.clamp(inter_ymax - inter_ymin, min=0)
        inter_area = inter_w * inter_h

        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - inter_area + eps
        IoU = inter_area / union
        IoU[~mask & ~pred_mask] = 1.0
        IoU[(mask & ~pred_mask) | (~mask & pred_mask)] = 0
        return IoU

    @staticmethod
    def cal_AUC_actual(target, pred, mask=None, eps=1e-6):
        SR = Evaluate.cal_SR_actual(target, pred, mask=mask, eps=eps)
        AUC = SR[:-1].mean()
        return AUC

    @staticmethod
    def cal_smooth_loss(cls_target, cls_pred, bbox_pred, bbox_target, null_penalty=10):
        '''
        :param cls_target: 目标是否存在
        :param bbox_pred: 预测方框
        :param bbox_target: 实际方框
        :param null_penalty: 错误检测时添加一个较大损失
        :return: 方框损失，已经将预测失误概率叠加到bbox_loss上
        '''
        # cls loss
        cls_pred = torch.sigmoid(cls_pred).squeeze(-1)
        cls_lamp = (nn.functional.binary_cross_entropy(
        cls_pred,
        cls_target.squeeze(-1),
        reduction='none'
        ))

        # bbox loss
        s1 = nn.SmoothL1Loss(reduction='none')
        pred_zero = (bbox_pred.abs().sum(dim=2) < 1e-6)  # [B,T] bool
        target_zero = ~cls_target.bool()

        bbox_lmap = s1(bbox_pred, bbox_target).mean(dim=2)  # per-frame avg over 4 coords -> [B,T]
        # cls_lamp[pred_zero ^ target_zero] = null_penalty
        # bbox_lmap[pred_zero ^ target_zero] = null_penalty  # 错误诊断时强制给一个较大损失
        cls_loss = cls_lamp.mean(dim=1)

        bbox_loss = bbox_lmap.mean(dim=1)
        loss = bbox_loss + cls_loss
        return loss.mean(), bbox_loss.mean(), cls_loss.mean()

    @staticmethod
    def cal_smooth_loss_weight(cls_target, cls_pred, bbox_pred, bbox_target, lambda_box=2.0, lambda_iou=1.0):
        """
        cls_pred: [B,T,1], sigmoided
        cls_target: [B,T,1]
        bbox_pred: [B,T,4] (cx,cy,w,h) normalized
        bbox_target: [B,T,4]
        """

        # ---- 分类 LOSS (BCE) ----
        cls_pred = cls_pred.squeeze(-1)    # [B,T]
        cls_target = cls_target.squeeze(-1).float()  # [B,T]

        cls_loss = F.binary_cross_entropy(cls_pred, cls_target, reduction='mean')

        # ---- 回归 LOSS（只对有目标帧计算） ----
        obj_mask = (cls_target > 0.5).float()  # [B,T]

        if obj_mask.sum() > 0:
            # SmoothL1
            reg_l1 = F.smooth_l1_loss(
                bbox_pred[obj_mask.bool()],
                bbox_target[obj_mask.bool()],
                reduction='mean'
            )

            # IoU Loss —— 强化几何一致性
            pred_xyxy = Evaluate.cxcywh_to_xyxy(bbox_pred[obj_mask.bool()])
            target_xyxy = Evaluate.cxcywh_to_xyxy(bbox_target[obj_mask.bool()])

            iou = Evaluate.bbox_iou(pred_xyxy, target_xyxy)
            iou_loss = 1 - iou.mean()

            reg_loss = lambda_box * reg_l1 + lambda_iou * iou_loss
        else:
            print("所给的实例中所有的帧都没有目标")
            reg_loss = torch.tensor(0.0, device=bbox_pred.device)

        total = cls_loss + reg_loss
        return total, cls_loss, reg_loss
    @staticmethod
    def cxcywh_to_xyxy(b):
        cx, cy, w, h = b.unbind(-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return torch.stack([x1, y1, x2, y2], dim=-1)
    @staticmethod
    def bbox_iou(b1, b2, eps=1e-6):
        x1 = torch.max(b1[..., 0], b2[..., 0])
        y1 = torch.max(b1[..., 1], b2[..., 1])
        x2 = torch.min(b1[..., 2], b2[..., 2])
        y2 = torch.min(b1[..., 3], b2[..., 3])
        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

        area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
        area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])

        union = area1 + area2 - inter + eps
        return inter / union

