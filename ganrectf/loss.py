import tensorflow as tf

@tf.function
def discriminator_loss(real_output, fake_output, smoothing = 0.1):
    real_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=real_output, labels=tf.ones_like(real_output) * (1 - smoothing))
    )
    fake_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output, labels=tf.zeros_like(fake_output))
    )
    total_loss = real_loss + fake_loss
    return total_loss

@tf.function
def generator_loss(fake_output, img_target, img_pred, recon, l1_ratio):
    # Cast all inputs to float32 for consistency.
    fake_output = tf.cast(fake_output, tf.float32)
    img_target = tf.cast(img_target, tf.float32)
    img_pred = tf.cast(img_pred, tf.float32)

    # Adversarial loss: compare fake_output with ones (targeting fooling the discriminator).
    adv_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake_output, 
            labels=tf.ones_like(fake_output)
        )
    )
    # Reconstruction loss: L1 loss between the ground truth image and the predicted image.
    abs_loss = l1_loss(img_pred, img_target)
    
    # SSIM loss: 1 - ssim score.
    ssim_component = ssim_loss(img_pred, img_target, max_val=tf.reduce_max(img_target)) 
 
    # Total Variation loss: encourages smoothness in the prediction.
    tv_component = total_variation_loss(recon)

    # Total generator loss: combine adversarial, reconstruction, SSIM, and TV losses.

    gen_loss = adv_loss + \
        abs_loss*40 + \
        ssim_component*50 + 1e-5 * tv_component + \
        0.05 * mean_match(img_target, img_pred) + \
        0.05 * variance_floor(img_pred)
    # gen_loss = adv_loss  + ssim_component*1e2
  
    # gen_loss  = adv_loss + svmbir_loss(img_target, img_pred, recon) * l1_ratio

    return gen_loss


def l1_loss(img1, img2):
    # Cast both images to float32 to ensure consistent types.
    img1 = tf.cast(img1, tf.float32)
    img2 = tf.cast(img2, tf.float32)
    return tf.reduce_mean(tf.abs(img1 - img2))

def l2_loss(img1, img2):
    return tf.square(tf.reduce_mean(tf.abs(img1 - img2)))


def _gaussian_kernel_3d(size: int, sigma: float) -> tf.Tensor:
    """
    Builds a 3D Gaussian kernel of shape [size, size, size, 1, 1].
    """
    ax = tf.range(-size//2 + 1, size//2 + 1, dtype=tf.float32)
    xx, yy, zz = tf.meshgrid(ax, ax, ax, indexing='ij')
    kernel = tf.exp(-(xx**2 + yy**2 + zz**2) / (2.0 * sigma**2))
    kernel = kernel / tf.reduce_sum(kernel)
    return kernel[:, :, :, tf.newaxis, tf.newaxis]


def _volumetric_ssim_3d(
    x: tf.Tensor,
    y: tf.Tensor,
    max_val: float,
    filter_size: int,
    filter_sigma: float,
    k1: float,
    k2: float,
    eps: float
) -> tf.Tensor:
    """
    Computes true 3D SSIM over volumes using a 3D Gaussian window.
    """
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    kernel = _gaussian_kernel_3d(filter_size, filter_sigma)
    strides = [1, 1, 1, 1, 1]
    padding = 'SAME'

    mu_x = tf.nn.conv3d(x, kernel, strides, padding)
    mu_y = tf.nn.conv3d(y, kernel, strides, padding)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = tf.nn.conv3d(x * x, kernel, strides, padding) - mu_x2
    sigma_y2 = tf.nn.conv3d(y * y, kernel, strides, padding) - mu_y2
    sigma_xy = tf.nn.conv3d(x * y, kernel, strides, padding) - mu_xy

    num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2 + eps)
    ssim_map = num / den
    return tf.reduce_mean(ssim_map)



def variance_floor(y, tau=0.02):  # tau ~ 1–3% of dynamic range in [-1,1]
    v = tf.math.reduce_variance(y, axis=[1,2,3])
    # penalize only if variance falls under tau^2
    return tf.reduce_mean(tf.nn.relu(tau*tau - v))

def mean_match(y_true, y_pred):
    m_t = tf.reduce_mean(y_true, axis=[1,2,3])
    m_p = tf.reduce_mean(y_pred, axis=[1,2,3])
    return tf.reduce_mean(tf.square(m_t - m_p))

def ssim_loss(
    x: tf.Tensor,
    y: tf.Tensor,
    max_val: float = 1.0,
    filter_size: int = 11,
    filter_sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    eps: float = 1e-8,
    volumetric: bool = False
) -> tf.Tensor:
    """
    Computes SSIM loss (1 - mean SSIM) for 2D images or 3D volumes.

    For 3D inputs, uses 2D slice-wise SSIM by default (with tf.image.ssim).
    Set volumetric=True to use true volumetric SSIM via 3D conv.

    Args:
        x: 4D ([B,H,W,C]) or 5D ([B,D,H,W,C]) Tensor.
        y: Ground-truth tensor of same shape as x.
        max_val: Dynamic range of input values.
        filter_size: Gaussian window size.
        filter_sigma: Gaussian window sigma.
        k1, k2: SSIM constants.
        eps: Small constant to stabilize division.
        volumetric: If True and input is 5D, use true 3D SSIM.

    Returns:
        Scalar Tensor: SSIM loss = 1 - mean SSIM.
    """
    ndim = x.shape.ndims
    if ndim == 4:
        # 2D SSIM via TensorFlow builtin
        ssim_map = tf.image.ssim(
            x, y, max_val,
            filter_size=filter_size,
            filter_sigma=filter_sigma,
            k1=k1, k2=k2
        )
        return 1.0 - tf.reduce_mean(ssim_map)
    elif ndim == 5:
        if volumetric:
            # True volumetric SSIM
            return 1.0 - _volumetric_ssim_3d(
                x, y, max_val, filter_size, filter_sigma, k1, k2, eps
            )
        else:
            # Slice-wise 2D SSIM over depth
            shape = tf.shape(x)
            batch, depth = shape[0], shape[1]
            # reshape to [batch*depth, H, W, C]
            x_slices = tf.reshape(x, [batch * depth, x.shape[2], x.shape[3], x.shape[4]])
            y_slices = tf.reshape(y, [batch * depth, y.shape[2], y.shape[3], y.shape[4]])
            ssim_map = tf.image.ssim(
                x_slices, y_slices, max_val,
                filter_size=filter_size,
                filter_sigma=filter_sigma,
                k1=k1, k2=k2
            )
            return 1.0 - tf.reduce_mean(ssim_map)
    else:
        raise ValueError(f"Input must be 4D or 5D, got {ndim}D.")


def total_variation_2d(x: tf.Tensor) -> tf.Tensor:
    """
    Computes total variation for 2D images.

    Args:
        x: 4D Tensor of shape [batch, height, width, channels].

    Returns:
        Scalar Tensor representing the total variation.
    """
    
    dh = x[:, 1:, :, :] - x[:, :-1, :, :]
    dw = x[:, :, 1:, :] - x[:, :, :-1, :]
    return tf.reduce_sum(tf.abs(dh)) + tf.reduce_sum(tf.abs(dw))


def total_variation_3d(x: tf.Tensor) -> tf.Tensor:
    """
    Computes total variation for 3D volumes.

    Args:
        x: 5D Tensor of shape [batch, depth, height, width, channels].

    Returns:
        Scalar Tensor representing the total variation.
    """
    dd = x[:, 1:, :, :, :] - x[:, :-1, :, :, :]
    dh = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
    dw = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
    return (
        tf.reduce_sum(tf.abs(dd))
        + tf.reduce_sum(tf.abs(dh))
        + tf.reduce_sum(tf.abs(dw))
    )


def total_variation_loss(
    x: tf.Tensor,
    reduction: str = 'sum'
) -> tf.Tensor:
    """
    Computes total variation loss for both 2D and 3D inputs.

    Args:
        x: 4D or 5D Tensor. Shape:
           - 4D: [batch, height, width, channels]
           - 5D: [batch, depth, height, width, channels]
        reduction: 'sum' or 'mean'. If 'mean', divides by the number of elements.

    Returns:
        Scalar Tensor: total variation loss.
    """
    ndim = x.shape.ndims
    if ndim == 4:
        tv = total_variation_2d(x)
    elif ndim == 5:
        tv = total_variation_3d(x)
    else:
        raise ValueError(f"Input tensor must be 4D or 5D, got {ndim}D.")

    if reduction == 'mean':
        return tv / tf.cast(tf.size(x), x.dtype)
    return tv

# -----------------------------
# Forward Model Loss: ||input_proj - pred_proj||^2_W
# -----------------------------

def forward_model_loss(pred_proj, input_proj, weight_type="transmission", sigma_y=1.0, eps=1e-8):
    """
    Computes the forward model loss.
    Args:
        pred_proj: Predicted projection tensor with shape (batch, n_angles, n_detectors).
        input_proj: Measured projection tensor with shape (batch, n_angles, n_detectors).
        weight_type: One of "unweighted", "transmission", "transmission_root", or "emission".
        sigma_y: Parameter controlling the assumed noise standard deviation.
        eps: Small constant for numerical stability.
    Returns:
        A scalar Tensor representing the forward model loss.
    """
    # Ensure inputs are float32.
    pred_proj = tf.cast(pred_proj, tf.float32)
    input_proj = tf.cast(input_proj, tf.float32)

    # Compute weighting Lambda based on weight_type.
    if weight_type == "unweighted":
        Lambda = tf.ones_like(input_proj)
    elif weight_type == "transmission":
        Lambda = tf.exp(-input_proj)
    elif weight_type == "transmission_root":
        Lambda = tf.exp(-input_proj / 2.0)
    elif weight_type == "emission":
        Lambda = 1.0 / (input_proj + 0.1)
    else:
        raise ValueError(f"Unknown weight_type: {weight_type}")
    
    diff = input_proj - pred_proj
    weighted_sq_error = Lambda * tf.square(diff)
    return tf.reduce_mean(weighted_sq_error) / (2 * sigma_y**2 + eps)


def qggmrf_3d_loss(pred_volume, sigma_x, p, q, T,
                   neighborhood_weight=1.0, eps=1e-8):
    """
    Computes the qGGMRF prior loss for a 3D volume, with extra safeguards
    to prevent NaNs by ensuring no division-by-zero or zero^negative_power.

    Args:
        pred_volume: 3D volume tensor with shape (batch, D, H, W, 1).
        sigma_x: Primary regularization parameter (scalar or tensor).
        p: Shape parameter controlling the potential function.
        q: Shape parameter controlling the potential function.
        T: Threshold parameter (scalar or tensor).
        neighborhood_weight: Scalar to weight the overall prior term.
        eps: Small constant for numerical stability.

    Returns:
        A scalar Tensor representing the qGGMRF prior loss.
    """
    pred = tf.cast(pred_volume, tf.float32)

    # Compute finite differences
    dz = pred[:, 1:, :, :, :] - pred[:, :-1, :, :, :]
    dy = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
    dx = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]

    # Flatten and concatenate
    batch = tf.shape(pred)[0]
    diffs = tf.concat([
        tf.reshape(dz, [batch, -1]),
        tf.reshape(dy, [batch, -1]),
        tf.reshape(dx, [batch, -1])
    ], axis=1)

    abs_diffs = tf.abs(diffs)

    # safe denominators
    safe_sigma = sigma_x + eps
    safe_T     = T + eps

    # scaled differences (clipped to [eps, ∞) to avoid zero^negative_power)
    scaled  = tf.clip_by_value(abs_diffs / safe_sigma, eps, tf.float32.max)
    scaledT = tf.clip_by_value(abs_diffs / safe_T,     eps, tf.float32.max)

    # numerator and denominator with eps offsets
    num = tf.pow(scaled + eps, q)
    den = 1.0 + tf.pow(scaledT + eps, q - p)

    # final loss terms, with divide_no_nan and eps in denominator
    loss_terms = tf.math.divide_no_nan(num,
                                       den + eps)

    # replace any remaining NaNs (just in case) with zeros
    loss_terms = tf.where(
        tf.math.is_nan(loss_terms),
        tf.zeros_like(loss_terms),
        loss_terms
    )

    return neighborhood_weight * tf.reduce_mean(loss_terms)


def proximal_map_prior_loss(pred_volume, prox_image, sigma_p, eps=1e-8):
    """
    Computes the proximal map prior loss.
    Args:
        pred_volume: Predicted 3D volume tensor with shape (batch, D, H, W, 1).
        prox_image: Denoised/processed image from an external prior.
        sigma_p: Parameter controlling the assumed noise level in the prior.
        eps: Small constant for numerical stability.
    Returns:
        A scalar Tensor representing the proximal map prior loss.
    """
    pred_volume = tf.cast(pred_volume, tf.float32)
    prox_image = tf.cast(prox_image, tf.float32)
    diff = pred_volume - prox_image
    return tf.reduce_mean(tf.square(diff)) / (2 * sigma_p**2 + eps)


def svmbir_loss(input_proj, pred_proj, pred_volume,
                sigma_y=1.0,
                weight_type="transmission",
                use_qggmrf=True,
                sigma_x=0.02, p=1.2, q=2.0, T=1.0,
                neighborhood_weight=1.0,
                use_prox=False,
                prox_image=None,
                sigma_p=0.01):
    """
    Computes the combined SVMBIR loss consisting of:
      - A forward model data fidelity term.
      - A prior model term (either qGGMRF or proximal map prior).
    Args:
        input_proj: Measured projection tensor (batch, n_angles, n_detectors).
        pred_proj: Predicted projection tensor (batch, n_angles, n_detectors).
        pred_volume: Predicted 3D volume tensor (batch, D, H, W, 1).
        sigma_y: Noise standard deviation for the forward model.
        weight_type: Weighting type for the projection.
        use_qggmrf: If True, use the qGGMRF prior; otherwise, use the proximal map prior.
        sigma_x, p, q, T: Hyperparameters for the qGGMRF prior.
        neighborhood_weight: Weighting for the prior term.
        use_prox: If True, use the proximal map prior.
        prox_image: Tensor for the proximal map prior (required if use_prox is True).
        sigma_p: Noise parameter for the proximal map prior.
    Returns:
        A scalar Tensor representing the total loss.
    """
    data_loss = forward_model_loss(pred_proj, input_proj, weight_type, sigma_y)
    
    if use_prox and prox_image is not None:
        prior_loss = proximal_map_prior_loss(pred_volume, prox_image, sigma_p)
    else:
        prior_loss = qggmrf_3d_loss(pred_volume, sigma_x, p, q, T, neighborhood_weight)
    
    total_loss = data_loss + prior_loss*1e-5
    # total_loss = data_loss

    # Optionally, add debugging checks:
    tf.debugging.check_numerics(total_loss, message="Total loss has NaNs")
    
    return total_loss