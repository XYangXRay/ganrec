import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from torch import Tensor
from torch.nn.functional import conv2d, pad as torch_pad
from ganrec_dataloader import torch_reshape

from torchvision.transforms.functional_tensor import  *
def _get_gaussian_kernel1d(kernel_size: int, sigma: float) -> Tensor:
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()

    return kernel1d

def _get_gaussian_kernel2d(
    kernel_size: List[int], sigma: List[float], dtype: torch.dtype, device: torch.device
) -> Tensor:
    kernel1d_x = _get_gaussian_kernel1d(kernel_size[0], sigma[0]).to(device, dtype=dtype)
    kernel1d_y = _get_gaussian_kernel1d(kernel_size[1], sigma[1]).to(device, dtype=dtype)
    kernel2d = torch.mm(kernel1d_y[:, None], kernel1d_x[None, :])
    return kernel2d

def _cast_squeeze_in(img: Tensor, req_dtypes: List[torch.dtype]) -> Tuple[Tensor, bool, bool, torch.dtype]:
    need_squeeze = False
    # make image NCHW
    if img.ndim < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if out_dtype not in req_dtypes:
        need_cast = True
        req_dtype = req_dtypes[0]
        img = img.to(req_dtype)
    return img, need_cast, need_squeeze, out_dtype

def _cast_squeeze_out(img: Tensor, need_cast: bool, need_squeeze: bool, out_dtype: torch.dtype) -> Tensor:
    if need_squeeze:
        img = img.squeeze(dim=0)

    if need_cast:
        if out_dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            # it is better to round before cast
            img = torch.round(img)
        img = img.to(out_dtype)

    return img

def _is_tensor_a_torch_image(x: Tensor) -> bool:
    return x.ndim >= 2

def _assert_image_tensor(img: Tensor) -> None:
    if not _is_tensor_a_torch_image(img):
        raise TypeError("Tensor is not a torch image.")


def gaussian_blur(img: Tensor, kernel_size: List[int], sigma: List[float]) -> Tensor:
    if not (isinstance(img, torch.Tensor)):
        raise TypeError(f"img should be Tensor. Got {type(img)}")

    _assert_image_tensor(img)

    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    kernel = _get_gaussian_kernel2d(kernel_size, sigma, dtype=dtype, device=img.device)
    kernel = kernel.expand(img.shape[-3], 1, kernel.shape[0], kernel.shape[1])
    img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(img, [kernel.dtype])

    # padding = (left, right, top, bottom)
    padding = [kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[1] // 2]
    img = torch_pad(img, padding, mode="reflect")
    img = conv2d(img, kernel, groups=img.shape[-3])

    img = _cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
    return img, kernel

class Gaussian_challenge(nn.Module):
    def __init__(self, img: torch.Tensor, kernel_size: List[int], sigma: List[float], device: Optional[torch.device] = None) -> None:
        super().__init__()
        if not (isinstance(img, torch.Tensor)):
            img = torch_reshape(img)
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.device = device if device is not None else img.device
        self.dtype = img.dtype if torch.is_floating_point(img) else torch.float32
        self.shape = list(img.shape)

        self.kernel = _get_gaussian_kernel2d(kernel_size, sigma, dtype=self.dtype, device=device)
        self.kernel = self.kernel.expand(self.shape[-3], 1, self.kernel.shape[0], self.kernel.shape[1])
        img, self.need_cast, self.need_squeeze, out_dtype = _cast_squeeze_in(img, [self.kernel.dtype])
        self.kernel = _cast_squeeze_out(self.kernel, self.need_cast, self.need_squeeze, out_dtype)
        
        self.kernel = self.kernel.to(self.device)


    def forward(self, img: Tensor) -> Tensor:
        if not (isinstance(img, torch.Tensor)):
            img = torch_reshape(img)
        if self.shape != list(img.shape) or self.dtype != img.dtype or self.device != img.device:
            Warning('img shape, dtype or device is not the same as the initialized one')
            new_self = Gaussian_challenge(img, self.kernel_size, self.sigma, device = img.device)
            padding = [new_self.kernel_size[0] // 2, new_self.kernel_size[0] // 2, new_self.kernel_size[1] // 2, new_self.kernel_size[1] // 2]
            img = torch_pad(img, padding, mode="reflect")
            blurred = conv2d(img, new_self.kernel, groups=img.shape[-3])
            return _cast_squeeze_out(blurred, new_self.need_cast, new_self.need_squeeze, new_self.dtype)
        else:
            padding = [self.kernel_size[0] // 2, self.kernel_size[0] // 2, self.kernel_size[1] // 2, self.kernel_size[1] // 2]
            img = torch_pad(img, padding, mode="reflect")
            blurred = conv2d(img, self.kernel, groups=img.shape[-3])
            return _cast_squeeze_out(blurred, self.need_cast, self.need_squeeze, self.dtype)
    