import os
import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms as T


class RGBDTextDataset(Dataset):
    """
    多模态视频目标跟踪数据集
    输入: RGB + Depth + Text + 初始标注框
    输出: (rgb_tensor, depth_tensor, text_string, bbox_tensor)
    """

    def __init__(self, root_dir, seq_len=61, img_size=(90, 160), transform=None, seq_used=3, seq_used_start=0):
        """
        Args:
            root_dir (str): 数据集根目录 ,训练集目录或者验证集目录
            transform (callable): 图像变换函数
            max_frames (int): 限制帧数 (调试时可设较小),从一个视频序列中只选取前 N 帧参与训练或验证
        """
        self.root_dir = root_dir
        self.transform = transform
        self.sequences = []
        self.categories = []

        for root, dirs, files in os.walk(root_dir):
            # 检查该目录是否为一个完整数据序列
            if 'color' in dirs and 'depth' in dirs:
                category = os.path.basename(os.path.dirname(root))
                if category == os.path.basename(root_dir):
                    category = "Unknown"
                self.sequences.append(root)
                self.categories.append(category)

        # 测试的时候用前20个数据
        seq_used_end = len(self.sequences)
        if seq_used:
            seq_used_end = min(seq_used+seq_used_start, len(self.sequences))
        self.sequences = self.sequences[seq_used_start:seq_used_end]
        self.categories = self.categories[seq_used_start:seq_used_end]
        self.img_size = img_size
        self.seq_len = seq_len  #一个序列的长度，主要是训练用到
        # print("一共使用", len(self.sequences),"个数据训练")

    def __len__(self):
        return len(self.sequences)

    # 读取文本模态信息
    def _read_text(self, seq_dir):
        text_path = os.path.join(seq_dir, "nlp.txt")
        with open(text_path, "r", encoding="utf-8") as f:
            return f.readline().strip()

    # 读取第一帧的时候目标位置
    def _read_init_box(self, seq_dir, scale_x, scale_y):
        init_path = os.path.join(seq_dir, "groundtruth_rect.txt")
        with open(init_path, "r", encoding="utf-8") as f:
            box = list(map(float, f.readline().split(",")))
            box[0] *= scale_x
            box[1] *= scale_y
            box[2] *= scale_x
            box[3] *= scale_y
            return np.array(box, dtype=np.float32)

    # 读取标注出的真实位置(gt_boxes),以及是否存在(mask)
    def _read_gt_boxes_mask(self, seq_dir, scale_x, scale_y, seq_len_use):
        gt_path = os.path.join(seq_dir, "groundtruth_rect.txt")
        gt_list = []
        with open(gt_path, "r", encoding="utf-8") as f:
            mask_list = [True] * seq_len_use
            for num, line in enumerate(f):
                if num >= seq_len_use:
                    break
                if line == "0,0,0,0\n":
                    mask_list[num] = False
                box = list(map(float, line.strip().split(",")))
                box[0] = box[0] + 0.5*box[2]
                box[1] = box[1] + 0.5*box[3]  # 由左上角转化为图像中心
                box[0] /= scale_x  # 全部放缩到[0, 1]
                box[1] /= scale_y
                box[2] /= scale_x
                box[3] /= scale_y
                gt_list.append(box)
        init_box = gt_list[0]
        gt_seq = torch.from_numpy(np.stack(gt_list, axis=0))
        mask_seq = torch.tensor(mask_list, dtype=torch.float32)
        return gt_seq, mask_seq, torch.tensor(np.array(init_box, dtype=np.float32), dtype=torch.float32)

    def _load_and_preproc_rgb(self, path):
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]))  # (W,H)
        img = img.astype(np.float32) / 255.0
        # normalize
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        # HWC -> CHW
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img)

    def _load_and_preproc_depth(self, path):
        d = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # HxW, uint16 maybe
        if d is None:
            d = np.zeros(self.img_size, dtype=np.uint16)
        d = cv2.resize(d, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)
        d = d.astype(np.float32)
        maxv = d.max()
        if maxv > 0:
            d = d / maxv
        else:
            d = d * 0.0
        d = np.expand_dims(d, axis=0)  # 1,H,W
        return torch.from_numpy(d)

    def __getitem__(self, idx): # 定义pytorch所用的类似列表的访问方法
        seq_dir = self.sequences[idx]
        rgb_dir = os.path.join(seq_dir, "color")
        depth_dir = os.path.join(seq_dir, "depth")
        if self.seq_len == None:
            seq_len_use = len(os.listdir(rgb_dir))
        else:
            seq_len_use = self.seq_len
        # print(os.path.join(rgb_dir, os.listdir(rgb_dir)[0]))
        first_img = cv2.imread(os.path.join(rgb_dir, os.listdir(rgb_dir)[0]))
        orig_h, orig_w = first_img.shape[:2]

        # 读取所有帧文件
        rgb_files = sorted([os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir) if f.endswith((".jpg", ".png"))])
        depth_files = sorted([os.path.join(depth_dir, f) for f in os.listdir(depth_dir) if f.endswith(".png")])
        n = len(rgb_files)

        sel_rgb = rgb_files[0:seq_len_use]
        sel_depth = depth_files[0:seq_len_use]

        # 读取并预处理
        rgb_tensors = [self._load_and_preproc_rgb(p) for p in sel_rgb]  # List of [3,H,W]
        depth_tensors = [self._load_and_preproc_depth(p) for p in sel_depth]  # List of [1,H,W]
        rgb_seq = torch.stack(rgb_tensors, dim=0)  # [T,3,H,W]
        depth_seq = torch.stack(depth_tensors, dim=0)  # [T,1,H,W]

        text = self._read_text(seq_dir)
        # init_box = self._read_init_box(seq_dir, orig_w, orig_h)
        gt_boxes, masks, init_box = self._read_gt_boxes_mask(seq_dir, orig_w, orig_h, seq_len_use)

        sample = {
            "rgb": rgb_seq,             # Tensor, shape [T, 3, H, W]
            "depth": depth_seq,         # Tensor, shape [T, 1, H, W]
            "text": text,           # str
            "init_box": init_box,       # Tensor, [x, y, w, h], reduce to [0, 1]
            "gt_boxes": gt_boxes,   # np.ndarray, optional, reduce to [0, 1]
            "cls": masks,
            "seq_dir": seq_dir,
        }
        return sample


# ===== 测试部分 =====
if __name__ == "__main__":
    BASE_DATA_DIR = Path(os.path.abspath(__file__)).resolve().parent.parent.parent.joinpath("data")
    BASE_TRAINSET_DIR = str(BASE_DATA_DIR.joinpath("TrainSet"))
    dataset = RGBDTextDataset(root_dir=BASE_TRAINSET_DIR, img_size=(90, 160))

    print(f"共加载 {len(dataset)} 个序列。")
    sample = dataset[1]

    print("样本字段：", sample.keys())
    print("RGB张量形状:", sample["rgb"].shape)
    print("Depth张量形状:", sample["depth"].shape)
    print("文本描述:", sample["text"])
    print("初始框:", sample["init_box"])
    print(sample["cls"])

