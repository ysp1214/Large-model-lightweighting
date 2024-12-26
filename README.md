# Large-model-lightweighting
## 大模型轻量化技术分类 
### 模型剪枝
#### 非结构化剪枝
#### 结构化剪枝
#### 动态剪枝
### 知识蒸馏

**知识蒸馏的开山之作**
**Distilling the Knowledge in a Neural Network**<br>
*Geoffrey Hinton、 Oriol Vinyals、 Jeff Dean ·*<br>
arXiv 2015  ·  [[PDF](https://arxiv.org/pdf/1503.02531v1)]

**将超参数温度动态化，网络会自动学习温度，无需手动设置**
**Curriculum Temperature for Knowledge Distillation**<br>
* Zheng Li, Xiang Li, Lingfeng Yang, Borui Zhao, RenJie Song, Lei Luo, Jun Li, Jian Yang · *<br>
arXiv 2022  ·  [[PDF](https://arxiv.org/pdf/2211.16231v3)]

**探讨了师生温度弊端问题，并提出了Logit标准化来解决**
**Logit Standardization in Knowledge Distillation**<br>
* Shangquan Sun, Wenqi Ren, Jingzhi Li, Rui Wang, Xiaochun Cao · *<br>
CVPR 2024  ·  [[PDF](https://arxiv.org/pdf/2403.01427v1)]

**首次提出用中间层蒸馏**
**FitNets: Hints for Thin Deep Nets**<br>
* Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta, Yoshua Bengio · *<br>
arXiv 2014  ·  [[PDF](https://arxiv.org/pdf/1412.6550v4)]

**多个阶段的中间层进行了融合，并设计了空间金字塔池化来匹配不同维度的教师学生**
**Distilling Knowledge via Knowledge Review**<br>
* Pengguang Chen, Shu Liu, Hengshuang Zhao, Jiaya Jia · *<br>
CVPR 2021  ·  [[PDF](https://arxiv.org/pdf/2104.09044v1)]

**随机选取教师的中间层与学生的中间层指导**
**RAndom Intermediate Layer Knowledge Distillation**<br>
* Md Akmal Haidar, Nithin Anchuri, Mehdi Rezagholizadeh, Abbas Ghaddar, Philippe Langlais, Pascal Poupart · *<br>
arXiv 2022  ·  [[PDF](https://arxiv.org/pdf/2109.10164)]

**通过注意力转移进行知识蒸馏**
**Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer**<br>
* Sergey Zagoruyko, Nikos Komodakis · *<br>
arXiv 2016  ·  [[PDF](https://arxiv.org/pdf/1612.03928v3)]  

**语义相似的输入往往会在训练好的网络中引发相似的激活模式**
**Similarity-Preserving Knowledge Distillation**<br>
* Frederick Tung, Greg Mori *<br>
ICCV 2019  ·  [[PDF](https://arxiv.org/pdf/1907.09682v2)]

**DisWOT在SP蒸馏方法的这个工作的基础之上，还增加了关系相似性**
**DisWOT: Student Architecture Search for Distillation WithOut Training**<br>
* Peijie Dong, Lujun Li, Zimian Wei · *<br>  
arXiv 2023  ·  [[PDF](https://arxiv.org/pdf/2303.15678v1)]  

**CCKD分析了实例与实例之间的相关性，提出实例之间的相关性对于蒸馏也是很有效果的。**
**Correlation Congruence for Knowledge Distillation**<br>
*  Baoyun Peng, Xiao Jin, Jiaheng Liu, Shunfeng Zhou, Yi-Chao Wu, Yu Liu, Dongsheng Li, Zhaoning Zhang ·*<br>
ICCV 2019  ·  [[PDF](https://arxiv.org/pdf/1904.01802v1)]

**VID提出了基于信息论的知识蒸馏框架。该框架最大化教师和学生网络之间的互信息**
**Variational Information Distillation for Knowledge Transfer**<br>
*  Sungsoo Ahn, Shell Xu Hu, Andreas Damianou, Neil D. Lawrence, Zhenwen Dai · *<br>
CVPR 2019  ·  [[PDF](https://arxiv.org/pdf/1904.05835v1)]

**transformer压缩成CNN或者CNN压缩为transformer**
**CMKD: CNN/Transformer-Based Cross-Model Knowledge Distillation for Audio Classification**<br>
*  Yuan Gong, Sameer Khurana, Andrew Rouditchenko, James Glass · * <br>
arXiv 2022  · [[PDF](https://arxiv.org/pdf/2203.06760v1)]

**惩罚关系中结构差异的距离方向和角度方向的蒸馏损失**
**Relational Knowledge Distillation**<br>
* Wonpyo Park, Dongju Kim, Yan Lu, Minsu Cho ·CVPR 2019  ·  Wonpyo Park, Dongju Kim, Yan Lu, Minsu Cho ·* <br>
CVPR 2019  ·  [[PDF](https://arxiv.org/pdf/1904.05068v2)]

**CRD首次将对比学习应用在知识蒸馏**
**Contrastive Representation Distillation**<br>
* Yonglong Tian, Dilip Krishnan, Phillip Isola * <br>
ICLR 2020.    [[PDF](https://arxiv.org/pdf/1910.10699v2)]

**现有的蒸馏方法主要是基于从中间层提取深层特征，而忽略了Logit蒸馏的重要性。为了给logit蒸馏研究提供一个新的视角，我们将经典的KD损失重新表述为两部分，即目标类知识蒸馏（TCKD）和非目标类知识蒸馏（NCKD）。**
**Decoupled Knowledge Distillation**<br>
*  Borui Zhao, Quan Cui, RenJie Song, Yiyu Qiu, Jiajun Liang · *<br>
CVPR 2022  ·  [[PDF](https://arxiv.org/pdf/2203.08679v2)]

**在传统Lotits的基础上对温度T进行了改良，不需要固定的温度，而是在训练的时候网络自动学习温度（动态温度）。**
**Curriculum Temperature for Knowledge Distillation**<br>
* Zheng Li, Xiang Li, Lingfeng Yang, Borui Zhao, Renjie Song, Lei Luo, Jun Li, Jian Yang *<br>
arXiv 2022  ·  [[PDF](https://arxiv.org/pdf/2211.16231)] 

**在传统Lotits的基础上对温度T进行了改良，不需要固定的温度，而是在训练的时候网络自动学习温度（动态温度）。**
**Knowledge Distillation with the Reused Teacher Classifier**<br>
* Defang Chen, Jian-Ping Mei, Hailin Zhang, Can Wang, Yan Feng, Chun Chen · *<br>
CVPR 2022  ·  [[PDF](https://arxiv.org/pdf/2203.14001v1)]

**MGD先随机生成Masked，然后通过2个3x3的卷积来生成特征，最后与教师的特征进行损失计算**
**Masked Generative Distillation**<br>
* Zhendong Yang, Zhe Li, Mingqi Shao, Dachuan Shi, Zehuan Yuan, Chun Yuan *<br>
ECCV 2022  · [[PDF](https://arxiv.org/pdf/2205.01529)]

**首次提出跨阶段连接路径来形成知识回顾，融合多阶段的知识**
**Distilling Knowledge via Knowledge Review**<br>
*  Pengguang Chen, Shu Liu, Hengshuang Zhao, Jiaya Jia · *<br>
CVPR 2021  ·  [[PDF](https://arxiv.org/pdf/2104.09044v1)]

**初始化3个线性回归器，学生网络的Logits输出经过3线性投影，然后取平均。最后与教师的Logits进行余弦损失计算。**
**Improved Feature Distillation via Projector Ensemble**<br>
* Yudong Chen, Sen Wang, Jiajun Liu, Xuwei Xu, Frank de Hoog, Zi Huang *<br>
NeurIPS 2022  ·  [[PDF](https://arxiv.org/pdf/2210.15274)]

**通过正则化特征的范数和方向来提升知识蒸馏的效果**
**Improving Knowledge Distillation via Regularizing Feature Norm and Direction**<br>
* Yuzhu Wang, Lechao Cheng, Manni Duan, Yongheng Wang, Zunlei Feng, Shu Kong *<br>
arXiv 2023   ·  [[PDF](https://arxiv.org/pdf/2305.17007)]

**在于通过独立处理特征尺度和其他因素，提高了学生模型对教师模型知识的学习效果。尺度正则化的引入不仅提升了蒸馏过程的稳定性，还优化了学生模型的泛化能力**
**Scale Decoupled Distillation**<br>
* Shicai Wei Chunbo Luo Yang Luo *<br>
arXiv 2024   ·  [[PDF]](https://arxiv.org/pdf/2403.13512)]

**设置迫使学生模仿教师的 logit 的具体值，而非其关系，论文方法提出 Logit 标准化**
**Logit Standardization in Knowledge Distillation**
* Shangquan Sun, Wenqi Ren, Jingzhi Li, Rui Wang, Xiaochun Cao *<br>
arXiv 2024   ·  [[PDF]](https://arxiv.org/pdf/2403.01427)]
### 神经架构设计与搜索（NAS）
### 模型量化
### 矩阵分解
## 大模型轻量化的挑战
## 轻量化技术的应用场景
## 未来发展方向
