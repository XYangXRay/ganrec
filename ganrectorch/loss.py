import torch
import torch.nn.functional as F


def discriminator_loss(real_output, fake_output, smoothing=0.1):
    bce = torch.nn.BCEWithLogitsLoss()
    real_loss = bce(real_output, torch.ones_like(real_output) * (1.0 - smoothing))
    fake_loss = bce(fake_output, torch.zeros_like(fake_output))
    total_loss = real_loss + fake_loss
    return total_loss


def l1_loss(img1, img2):
    return torch.mean(torch.abs(img1 - img2))


def l2_loss(img1, img2):
    return torch.pow(torch.mean(torch.abs(img1 - img2)), 2)


def _gaussian_kernel(size=11, sigma=1.5, device=None, dtype=torch.float32):
    coords = torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    return torch.outer(g, g).view(1, 1, size, size)


def ssim_loss(img_pred, img_target, max_val, size=11, sigma=1.5, k1=0.01, k2=0.03):
    """1 - mean SSIM, matching tf.image.ssim (Gaussian window, VALID padding)."""
    img_pred = img_pred.float()
    img_target = img_target.float()
    kernel = _gaussian_kernel(size, sigma, img_pred.device, img_pred.dtype)
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    mu_x = F.conv2d(img_pred, kernel)
    mu_y = F.conv2d(img_target, kernel)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = F.conv2d(img_pred * img_pred, kernel) - mu_x2
    sigma_y2 = F.conv2d(img_target * img_target, kernel) - mu_y2
    sigma_xy = F.conv2d(img_pred * img_target, kernel) - mu_xy
    ssim_map = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return 1.0 - ssim_map.mean()


def total_variation_loss(x):
    x = x.float()
    dh = x[:, :, 1:, :] - x[:, :, :-1, :]
    dw = x[:, :, :, 1:] - x[:, :, :, :-1]
    return torch.sum(torch.abs(dh)) + torch.sum(torch.abs(dw))


def mean_match(y_true, y_pred):
    m_t = y_true.float().mean(dim=[1, 2, 3])
    m_p = y_pred.float().mean(dim=[1, 2, 3])
    return torch.mean((m_t - m_p) ** 2)


def variance_floor(y, tau=0.02):
    v = torch.var(y.float(), dim=[1, 2, 3], unbiased=False)
    return torch.mean(F.relu(tau * tau - v))


def generator_loss(fake_output, img_target, img_pred, recon, l1_ratio):
    fake_output = fake_output.float()
    img_target = img_target.float()
    img_pred = img_pred.float()
    adv_loss = torch.mean(
        torch.nn.BCEWithLogitsLoss()(fake_output, torch.ones_like(fake_output))
    )
    abs_loss = l1_loss(img_pred, img_target)
    ssim_component = ssim_loss(img_pred, img_target, torch.max(img_target))
    tv_component = total_variation_loss(recon)
    gen_loss = (
        adv_loss
        + abs_loss * 20.0
        + ssim_component * 50.0
        + 1e-5 * tv_component
        + 0.05 * mean_match(img_target, img_pred)
        + 0.05 * variance_floor(img_pred)
    )
    return gen_loss
