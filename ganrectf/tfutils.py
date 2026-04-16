from typing import Union, Callable, Optional, List
import importlib
import numpy as np
import tensorflow as tf


@tf.function
def normalize_to_target_range(
    generated: tf.Tensor,
    target: tf.Tensor,
    axis=None,
    epsilon: float = 1e-8
) -> tf.Tensor:
    """
    Linearly rescales `generated` so that its [min, max] equals
    the [min, max] of `target`, even if those values are negative.
    If `generated` is constant (min == max), it will be mapped
    to the midpoint of the `target` range.

    Args:
        generated: tf.Tensor of any shape.
        target:    tf.Tensor broadcastable to generated.
        axis:      Dimensions to reduce over when computing min/max
                   (e.g. [1,2,3] to keep batch dim). None = all elements.
        epsilon:   Small constant to avoid divide-by-zero.

    Returns:
        A tf.Tensor same shape as `generated`, but with
        min→min(target) and max→max(target).
    """
    # 1) compute per-sample minima and maxima
    gen_min = tf.reduce_min(generated, axis=axis, keepdims=True)
    gen_max = tf.reduce_max(generated, axis=axis, keepdims=True)
    tar_min = tf.reduce_min(target,    axis=axis, keepdims=True)
    tar_max = tf.reduce_max(target,    axis=axis, keepdims=True)

    # 2) compute ranges
    gen_range = gen_max - gen_min
    tar_range = tar_max - tar_min

    # 3) avoid zero division: if gen_range is too small, replace it with 1
    safe_gen_range = tf.where(
        gen_range < epsilon,
        tf.ones_like(gen_range),
        gen_range
    )

    # 4) normalize to [0,1], then scale to [tar_min, tar_max]
    normalized = (generated - gen_min) / safe_gen_range
    scaled = normalized * tar_range + tar_min

    # 5) for constant inputs, map to target midpoint
    target_mid = (tar_min + tar_max) * 0.5
    scaled = tf.where(gen_range < epsilon, target_mid, scaled)
    return scaled

@tf.function
def tfnor_tomo(img: tf.Tensor, eps: float = 1e-8) -> tf.Tensor:
    """
    Standardizes input (zero-mean, unit-variance) and shifts to non-negative range.
    Supports both 2D images and 3D volumes.

    Args:
        img: 4D Tensor [B, H, W, C] for images or
             5D Tensor [B, D, H, W, C] for volumes.
        eps: Small constant to avoid divide-by-zero.

    Returns:
        Tensor of same shape as img, with per-sample standardized values shifted
        so that the minimum is 0.
    """
    # Determine reduction axes based on tensor rank
    ndim = img.shape.ndims
    if ndim == 4:
        # [batch, height, width, channels]
        axes = [1, 2, 3]
    elif ndim == 5:
        # [batch, depth, height, width, channels]
        axes = [1, 2, 3, 4]
    else:
        # fallback: normalize over all elements
        axes = None

    # Compute per-sample mean and std
    mean = tf.reduce_mean(img, axis=axes, keepdims=True)
    std = tf.math.reduce_std(img, axis=axes, keepdims=True)
    # Standardize
    img_norm = (img - mean) / (std + eps)

    # Shift so minimum becomes 0
    min_val = tf.reduce_min(img_norm, axis=axes, keepdims=True)
    img_pos = (img_norm - min_val)/ (tf.reduce_max(img_norm - min_val) + eps)

    return img_pos


def tfnor_phase(img):
    img = tf.image.per_image_standardization(img)
    img = img / tf.reduce_max(img)
    return img


def flat_output(y, thr=1e-4):
    # y in [-1,1]; thr is very small variance
    v = tf.math.reduce_variance(y)
    return bool((v < thr).numpy())

def set_lr(opt, new_lr: float):
    try:
        opt.learning_rate.assign(new_lr)
    except Exception:
        opt.learning_rate = new_lr

def _batch_assign(dst_list, src_list):
    """Batch-assign src -> dst for a list of variable pairs (pure Python loop).
    Batching into a single compiled function avoids per-variable kernel launches."""
    for d, s in zip(dst_list, src_list):
        d.assign(s)


class DeviceEMA:
    """EMA on device: shadow Variables match model.trainable_variables."""
    def __init__(self, model, decay=0.999):
        self.model  = model
        self.decay  = tf.constant(decay, tf.float32)
        self.one_m  = tf.constant(1.0 - decay, tf.float32)
        self.src    = list(model.trainable_variables)
        self.shadow = [tf.Variable(v, trainable=False) for v in self.src]
        self.backup = [tf.Variable(v, trainable=False) for v in self.src]

    @tf.function(reduce_retracing=True)
    def update(self):
        for s, v in zip(self.shadow, self.src):
            s.assign_add(self.one_m * (v - s))  # numerically identical, one less multiply

    @tf.function(reduce_retracing=True)
    def swap_in(self):
        for b, v in zip(self.backup, self.src):
            b.assign(v)
        for v, s in zip(self.src, self.shadow):
            v.assign(s)

    @tf.function(reduce_retracing=True)
    def swap_out(self):
        for v, b in zip(self.src, self.backup):
            v.assign(b)


class DeviceSnapshot:
    """On-device snapshot/restore of variables (no host copies).

    Compiled snapshot/restore to batch all assigns into a single device launch.
    """
    def __init__(self, var_lists):
        self.targets = [list(vs) for vs in var_lists]
        self.copies  = [[tf.Variable(v, trainable=False) for v in vs]
                        for vs in self.targets]
        # Pre-compile snapshot/restore as tf.functions for this specific set of variables
        self._do_snapshot = tf.function(self._raw_snapshot, reduce_retracing=True)
        self._do_restore  = tf.function(self._raw_restore,  reduce_retracing=True)

    def _raw_snapshot(self):
        for copies, targets in zip(self.copies, self.targets):
            for c, v in zip(copies, targets):
                c.assign(v)

    def _raw_restore(self):
        for copies, targets in zip(self.copies, self.targets):
            for v, c in zip(targets, copies):
                v.assign(c)

    def snapshot(self):
        self._do_snapshot()

    def restore(self):
        self._do_restore()


class ScalarEMA:
    """Scalar EMA without host sync."""
    def __init__(self, decay=0.95):
        self.decay = tf.constant(decay, tf.float32)
        self.value = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.has   = tf.Variable(False, trainable=False, dtype=tf.bool)

    @tf.function(reduce_retracing=True)
    def update(self, x):
        x = tf.cast(x, tf.float32)
        def _first():
            self.value.assign(x); self.has.assign(True)
            return self.value
        def _upd():
            self.value.assign(self.decay * self.value + (1.0 - self.decay) * x)
            return self.value
        return tf.cond(self.has, _upd, _first)

major, minor, _ = tf.__version__.split(".")
if int(major) > 2 or (int(major) == 2 and int(minor) > 9):
    from tensorflow.keras import KerasTensor
else:
    from keras.engine.keras_tensor import KerasTensor


_IMAGE_DTYPES = {
    tf.dtypes.uint8,
    tf.dtypes.int32,
    tf.dtypes.int64,
    tf.dtypes.float16,
    tf.dtypes.float32,
    tf.dtypes.float64,
}

Number = Union[
    float,
    int,
    np.float16,
    np.float32,
    np.float64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]

Initializer = Union[None, dict, str, Callable, tf.keras.initializers.Initializer]
Regularizer = Union[None, dict, str, Callable, tf.keras.regularizers.Regularizer]
Constraint = Union[None, dict, str, Callable, tf.keras.constraints.Constraint]
Activation = Union[None, str, Callable]
if importlib.util.find_spec("tensorflow.keras.optimizers.legacy") is not None:
    Optimizer = Union[tf.keras.optimizers.Optimizer, tf.keras.optimizers.legacy.Optimizer, str]
else:
    Optimizer = Union[tf.keras.optimizers.Optimizer, str]

TensorLike = Union[
    List[Union[Number, list]],
    tuple,
    Number,
    np.ndarray,
    tf.Tensor,
    tf.SparseTensor,
    tf.Variable,
    KerasTensor,
]
FloatTensorLike = Union[tf.Tensor, float, np.float16, np.float32, np.float64]
AcceptableDTypes = Union[tf.DType, np.dtype, type, int, str, None]


def to_4D_image(image):
    """Convert 2/3/4D image to 4D image.

    Args:
      image: 2/3/4D `Tensor`.

    Returns:
      4D `Tensor` with the same type.
    """
    with tf.control_dependencies(
        [tf.debugging.assert_rank_in(image, [2, 3, 4], message="`image` must be 2/3/4D tensor")]
    ):
        ndims = image.get_shape().ndims
        if ndims is None:
            return _dynamic_to_4D_image(image)
        elif ndims == 2:
            return image[None, :, :, None]
        elif ndims == 3:
            return image[None, :, :, :]
        else:
            return image


def _dynamic_to_4D_image(image):
    shape = tf.shape(image)
    original_rank = tf.rank(image)
    # 4D image => [N, H, W, C] or [N, C, H, W]
    # 3D image => [1, H, W, C] or [1, C, H, W]
    # 2D image => [1, H, W, 1]
    left_pad = tf.cast(tf.less_equal(original_rank, 3), dtype=tf.int32)
    right_pad = tf.cast(tf.equal(original_rank, 2), dtype=tf.int32)
    new_shape = tf.concat(
        [
            tf.ones(shape=left_pad, dtype=tf.int32),
            shape,
            tf.ones(shape=right_pad, dtype=tf.int32),
        ],
        axis=0,
    )
    return tf.reshape(image, new_shape)


def from_4D_image(image, ndims):
    """Convert back to an image with `ndims` rank.

    Args:
      image: 4D `Tensor`.
      ndims: The original rank of the image.

    Returns:
      `ndims`-D `Tensor` with the same type.
    """
    with tf.control_dependencies([tf.debugging.assert_rank(image, 4, message="`image` must be 4D tensor")]):
        if isinstance(ndims, tf.Tensor):
            return _dynamic_from_4D_image(image, ndims)
        elif ndims == 2:
            return tf.squeeze(image, [0, 3])
        elif ndims == 3:
            return tf.squeeze(image, [0])
        else:
            return image


def _dynamic_from_4D_image(image, original_rank):
    shape = tf.shape(image)
    # 4D image <= [N, H, W, C] or [N, C, H, W]
    # 3D image <= [1, H, W, C] or [1, C, H, W]
    # 2D image <= [1, H, W, 1]
    begin = tf.cast(tf.less_equal(original_rank, 3), dtype=tf.int32)
    end = 4 - tf.cast(tf.equal(original_rank, 2), dtype=tf.int32)
    new_shape = shape[begin:end]
    return tf.reshape(image, new_shape)


def get_ndims(image):
    return image.get_shape().ndims or tf.rank(image)


def tfrotate(
    images: TensorLike,
    angles: TensorLike,
    interpolation: str = "nearest",
    fill_mode: str = "constant",
    name: Optional[str] = None,
    fill_value: TensorLike = 0.0,
) -> tf.Tensor:
    """Rotate image(s) counterclockwise by the passed angle(s) in radians.

    Args:
      images: A tensor of shape
        `(num_images, num_rows, num_columns, num_channels)`
        (NHWC), `(num_rows, num_columns, num_channels)` (HWC), or
        `(num_rows, num_columns)` (HW).
      angles: A scalar angle to rotate all images by, or (if `images` has rank 4)
        a vector of length num_images, with an angle for each image in the
        batch.
      interpolation: Interpolation mode. Supported values: "nearest",
        "bilinear".
      fill_mode: Points outside the boundaries of the input are filled according
        to the given mode (one of `{'constant', 'reflect', 'wrap', 'nearest'}`).
        - *reflect*: `(d c b a | a b c d | d c b a)`
          The input is extended by reflecting about the edge of the last pixel.
        - *constant*: `(k k k k | a b c d | k k k k)`
          The input is extended by filling all values beyond the edge with the
          same constant value k = 0.
        - *wrap*: `(a b c d | a b c d | a b c d)`
          The input is extended by wrapping around to the opposite edge.
        - *nearest*: `(a a a a | a b c d | d d d d)`
          The input is extended by the nearest pixel.
      fill_value: a float represents the value to be filled outside the
        boundaries when `fill_mode` is "constant".
      name: The name of the op.

    Returns:
      Image(s) with the same type and shape as `images`, rotated by the given
      angle(s). Empty space due to the rotation will be filled with zeros.

    Raises:
      TypeError: If `images` is an invalid type.
    """
    with tf.name_scope(name or "rotate"):
        image_or_images = tf.convert_to_tensor(images)
        if image_or_images.dtype.base_dtype not in _IMAGE_DTYPES:
            raise TypeError("Invalid dtype %s." % image_or_images.dtype)
        images = to_4D_image(image_or_images)
        original_ndims = get_ndims(image_or_images)

        image_height = tf.cast(tf.shape(images)[1], tf.dtypes.float32)[None]
        image_width = tf.cast(tf.shape(images)[2], tf.dtypes.float32)[None]
        output = transform(
            images,
            angles_to_projective_transforms(angles, image_height, image_width),
            interpolation=interpolation,
            fill_mode=fill_mode,
            fill_value=fill_value,
        )
        return from_4D_image(output, original_ndims)


def transform(
    images: TensorLike,
    transforms: TensorLike,
    interpolation: str = "nearest",
    fill_mode: str = "constant",
    output_shape: Optional[list] = None,
    name: Optional[str] = None,
    fill_value: TensorLike = 0.0,
) -> tf.Tensor:
    """Applies the given transform(s) to the image(s).

    Args:
      images: A tensor of shape (num_images, num_rows, num_columns,
        num_channels) (NHWC), (num_rows, num_columns, num_channels) (HWC), or
        (num_rows, num_columns) (HW).
      transforms: Projective transform matrix/matrices. A vector of length 8 or
        tensor of size N x 8. If one row of transforms is
        [a0, a1, a2, b0, b1, b2, c0, c1], then it maps the *output* point
        `(x, y)` to a transformed *input* point
        `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`,
        where `k = c0 x + c1 y + 1`. The transforms are *inverted* compared to
        the transform mapping input points to output points. Note that
        gradients are not backpropagated into transformation parameters.
      interpolation: Interpolation mode.
        Supported values: "nearest", "bilinear".
      fill_mode: Points outside the boundaries of the input are filled according
        to the given mode (one of `{'constant', 'reflect', 'wrap', 'nearest'}`).
        - *reflect*: `(d c b a | a b c d | d c b a)`
          The input is extended by reflecting about the edge of the last pixel.
        - *constant*: `(k k k k | a b c d | k k k k)`
          The input is extended by filling all values beyond the edge with the
          same constant value k = 0.
        - *wrap*: `(a b c d | a b c d | a b c d)`
          The input is extended by wrapping around to the opposite edge.
        - *nearest*: `(a a a a | a b c d | d d d d)`
          The input is extended by the nearest pixel.
      fill_value: a float represents the value to be filled outside the
        boundaries when `fill_mode` is "constant".
      output_shape: Output dimesion after the transform, [height, width].
        If None, output is the same size as input image.

      name: The name of the op.

    Returns:
      Image(s) with the same type and shape as `images`, with the given
      transform(s) applied. Transformed coordinates outside of the input image
      will be filled with zeros.

    Raises:
      TypeError: If `image` is an invalid type.
      ValueError: If output shape is not 1-D int32 Tensor.
    """
    with tf.name_scope(name or "transform"):
        image_or_images = tf.convert_to_tensor(images, name="images")
        transform_or_transforms = tf.convert_to_tensor(transforms, name="transforms", dtype=tf.dtypes.float32)
        if image_or_images.dtype.base_dtype not in _IMAGE_DTYPES:
            raise TypeError("Invalid dtype %s." % image_or_images.dtype)
        images = to_4D_image(image_or_images)
        original_ndims = get_ndims(image_or_images)

        if output_shape is None:
            output_shape = tf.shape(images)[1:3]

        output_shape = tf.convert_to_tensor(output_shape, tf.dtypes.int32, name="output_shape")

        if not output_shape.get_shape().is_compatible_with([2]):
            raise ValueError("output_shape must be a 1-D Tensor of 2 elements: " "new_height, new_width")

        if len(transform_or_transforms.get_shape()) == 1:
            transforms = transform_or_transforms[None]
        elif transform_or_transforms.get_shape().ndims is None:
            raise ValueError("transforms rank must be statically known")
        elif len(transform_or_transforms.get_shape()) == 2:
            transforms = transform_or_transforms
        else:
            transforms = transform_or_transforms
            raise ValueError("transforms should have rank 1 or 2, but got rank %d" % len(transforms.get_shape()))

        fill_value = tf.convert_to_tensor(fill_value, dtype=tf.float32, name="fill_value")
        output = tf.raw_ops.ImageProjectiveTransformV3(
            images=images,
            transforms=transforms,
            output_shape=output_shape,
            interpolation=interpolation.upper(),
            fill_mode=fill_mode.upper(),
            fill_value=fill_value,
        )
        return from_4D_image(output, original_ndims)


def angles_to_projective_transforms(
    angles: TensorLike,
    image_height: TensorLike,
    image_width: TensorLike,
    name: Optional[str] = None,
) -> tf.Tensor:
    """Returns projective transform(s) for the given angle(s).

    Args:
      angles: A scalar angle to rotate all images by, or (for batches of
        images) a vector with an angle to rotate each image in the batch. The
        rank must be statically known (the shape is not `TensorShape(None)`.
      image_height: Height of the image(s) to be transformed.
      image_width: Width of the image(s) to be transformed.

    Returns:
      A tensor of shape (num_images, 8). Projective transforms which can be
      given to `transform` op.
    """
    with tf.name_scope(name or "angles_to_projective_transforms"):
        angle_or_angles = tf.convert_to_tensor(angles, name="angles", dtype=tf.dtypes.float32)
        if len(angle_or_angles.get_shape()) == 0:
            angles = angle_or_angles[None]
        elif len(angle_or_angles.get_shape()) == 1:
            angles = angle_or_angles
        else:
            raise ValueError("angles should have rank 0 or 1.")
        cos_angles = tf.math.cos(angles)
        sin_angles = tf.math.sin(angles)
        x_offset = ((image_width - 1) - (cos_angles * (image_width - 1) - sin_angles * (image_height - 1))) / 2.0
        y_offset = ((image_height - 1) - (sin_angles * (image_width - 1) + cos_angles * (image_height - 1))) / 2.0
        num_angles = tf.shape(angles)[0]
        return tf.concat(
            values=[
                cos_angles[:, None],
                -sin_angles[:, None],
                x_offset[:, None],
                sin_angles[:, None],
                cos_angles[:, None],
                y_offset[:, None],
                tf.zeros((num_angles, 2), tf.dtypes.float32),
            ],
            axis=1,
        )
