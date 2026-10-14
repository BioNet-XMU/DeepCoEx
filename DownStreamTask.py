import numpy as np
import torch
import matplotlib.pyplot as plt

# 加载数据
oridata = np.load("/home/lsy/home/result/pg10-3.npy")

# 图像尺寸
image_size = (230, 93)

# 归一化处理，避免除零错误
max_val = np.max(oridata, axis=0)
min_val = np.min(oridata, axis=0)
diff = max_val - min_val
diff[diff == 0] = 1  # 避免除零
nor_data = (oridata - min_val) / diff

# 生成背景 mask
index = np.where(np.sum(oridata, axis=-1) == 0, 0, 1)  # 确保 `axis` 适用于 oridata 维度
print(f"背景像素数量: {len(np.where(index == 0)[0])}")

index = np.array(index).reshape(image_size)  # 确保 `index` 为二维数组
np.save('背景.npy', index)

# 显示背景
plt.subplot(2, 3, 1)
plt.imshow(index)
plt.show()

# 转换为 PyTorch 张量并保存
tensor = torch.from_numpy(index).float()
torch.save(tensor, 'tensor-pg10_3.pt')

# 扩展到目标形状
desired_shape = (100, 3, 230, 93)  # (batch, channel, height, width)
tensor1 = tensor.unsqueeze(0).unsqueeze(0)  # 变成 (1, 1, 230, 93)
expanded_tensor = tensor1.expand(desired_shape)

# 解决数据类型问题并保存
tensor_without_dtype = expanded_tensor.detach().clone().to(torch.float32)
torch.save(tensor_without_dtype, 'tensor-pg10-3-train.pth')

print("张量保存成功")

