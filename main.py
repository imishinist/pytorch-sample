#!/usr/bin/env python

import torch #基本モジュール
from torch.autograd import Variable #自動微分用
import torch.nn as nn #ネットワーク構築用
import torch.optim as optim #最適化関数
import torch.nn.functional as F #ネットワーク用の様々な関数
import torch.utils.data #データセット読み込み関連


if __name__ == '__main__':
    x = torch.Tensor(5, 3) #5x3のTensorの定義
    y = torch.rand(5, 3) #5x3の乱数で初期化したTensorの定義
    z = x + y #普通に演算も可能
    print(z)
