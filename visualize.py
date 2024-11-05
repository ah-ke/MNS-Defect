import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

def visualize_score_maps(score_maps, B):
    """
    可视化评分图。

    :param score_maps: 张量，形状为 [(1296*B)×1]。
    :param B: 批量大小。
    """
    # 计算边长 H 和 W
    H = W = int(math.sqrt(score_maps.shape[0] // B))
    # 重新排列张量为 (B, 1, H, W)
    score_maps = score_maps.reshape(B, H, W, 1).permute(0, 3, 1, 2)
    # 获取第一个样本并转换为 numpy 数组
    score_map = score_maps[0].detach().cpu().numpy()
    # 去除单通道维度
    feature_map_numpy = score_map.squeeze(0)
    # 可视化
    plt.imshow(feature_map_numpy, cmap='viridis')  # 选择合适的颜色映射
    plt.colorbar()  # 添加颜色条
    plt.title('Visualized Score Map')
    plt.show()

def visualize_combined_feature_map_3d(feature_map): # [512×36×36]
    # 将所有通道的特征图相加
    combined_feature_map = torch.sum(feature_map, dim=0)  # 在通道维度上求和

    # 将张量转换为NumPy数组
    combined_feature_map_numpy = combined_feature_map.cpu().numpy()

    combined_feature_map_numpy = combined_feature_map_numpy.squeeze()
    plt.imshow(combined_feature_map_numpy, cmap='viridis')  # 选择合适的颜色映射
    plt.colorbar()  # 添加颜色条
    plt.show()


def visualize_combined_feature_map_2d(feature_map, cmap='viridis'): # [1296×1536]
    shape = feature_map.shape
    side_length = int(np.sqrt(shape[0]))
    feature_map = feature_map.reshape(side_length, side_length, shape[1])

    """Visualize a feature map."""
    if feature_map.requires_grad:
        combined_feature_map_tensor = feature_map.detach().cpu()
    else:
        combined_feature_map_tensor = feature_map.cpu()

    combined_feature_map_tensor = combined_feature_map_tensor.permute(2, 0, 1)  # 转换维度顺序
    combined_feature_map_numpy = combined_feature_map_tensor.numpy()  # 使用 detach() 获取不需要梯度的张量，并将其转换为 NumPy 数组
    combined_feature_map_numpy = np.sum(combined_feature_map_numpy, axis=0)  # 对特征图进行求和
    plt.imshow(combined_feature_map_numpy, cmap=cmap)
    plt.axis('off')
    plt.show()


def plot_feature_std_histogram(feature_map, name=None):

    # 计算每个维度的标准差
    std = torch.std(feature_map, dim=0)

    # 将标准差转换为 NumPy 数组
    std_np = std.cpu().detach().numpy()

    # 如果提供了特征名字，就设置标签
    if name is not None:
        label = str(name)
    else:
        label = None

    # 绘制直方图
    plt.hist(std_np, bins=50, alpha=0.7, label=label, density=True)
    plt.xlabel('Feature Std')
    plt.ylabel('Frequency')
    plt.title('Histogram of Standard Deviation of Features')
    plt.show()


def plot_multi_feature_std_histogram(features):
    # 设置中文字体路径，根据实际字体文件路径进行修改
    font_path = r'C:\Windows\Fonts\simhei.ttf'  # 以宋体（SimHei）为例

    # 设置中文字体，设置字体大小为14
    zhfont = matplotlib.font_manager.FontProperties(fname=font_path, size=14)

    # 从输入列表中提取特征数据和特征名称
    feature_tensors = []
    feature_names = []
    for i, item in enumerate(features):
        if isinstance(item, torch.Tensor):
            feature_tensors.append(item)
        else:
            feature_names.append(item)

    # 计算每个特征的标准差
    stds = [torch.std(feature, dim=0) for feature in feature_tensors]

    # 将标准差转换为 NumPy 数组
    stds_np = [std.cpu().detach().numpy() for std in stds]

    # 绘制直方图
    for i, std_np in enumerate(stds_np):
        plt.hist(std_np, bins=50, alpha=0.7, label='{}'.format(feature_names[i]), density=True)

    # 设置中文字体
    plt.xlabel('瓶子的特征标准差', fontproperties=zhfont)
    plt.ylabel('频率', fontproperties=zhfont)
    # plt.title('特征标准差直方图', fontproperties=zhfont)
    plt.legend(prop=zhfont)
    plt.savefig(r"G:\Anomaly\Dataset\MVTec_Analyse\bottle_std.png")
    plt.show()

def visualize_ndarray(features, cmap='viridis'):
    if features.shape[0] == 1:
        # 如果通道数为 1，则不需要叠加通道，直接可视化单个通道
        image = features[0]
    else:
        # 否则，将所有通道的值相加以获取单个图像
        image = np.sum(features, axis=0)

    # 可视化图像
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.title('Feature Image')
    plt.show()