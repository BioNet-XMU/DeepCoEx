from typing import Any, Dict, Optional
import numpy as np
import torch
from torch import Tensor
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D

class neg_transform(IntensityAugmentationBase2D):

    def __init__(
            self,
            tensor_path: str,
            target_height: int,  # 新增目标高度参数
            same_on_batch: bool = False,
            p: float = 1,
            keepdim: bool = False,
            return_transform: Optional[bool] = None,
    ) -> None:
        super().__init__(
            p=p,  same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim
        )
        self.tensor_path = tensor_path  # Store the tensor path
        self.target_height = target_height  # Store the目标高度

    def generate_parameters(self, shape: torch.Size) -> Dict[str, Tensor]:
        noise = torch.randn(shape)
        return dict(noise=noise)

    def apply_transform(
            self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:

        batch_size, _, height, width = input.shape

        # 判断输入的高度是否等于目标高度
        if height == self.target_height:
            # Load the tensor from the specified path
            data_pic = torch.load(self.tensor_path, map_location=input.device)
            input = data_pic[:1].to(dtype=input.dtype) - input

        return input

