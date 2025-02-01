from typing import Union, Callable, Optional, List
import importlib
import numpy as np
import tensorflow as tf

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

def gaussian_kernel(size: int, sigma: float):
    """
    Creates a 2D Gaussian kernel for convolution in TensorFlow.

    Parameters:
    size (int): Size of the kernel (e.g., 3, 5, 7).
    sigma (float): Standard deviation of the Gaussian.

    Returns:
    tf.Tensor: A 2D Gaussian kernel as a TensorFlow tensor.
    """
    # Create a 1D Gaussian distribution
    x = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
    gauss_1d = tf.exp(-0.5 * (x / sigma) ** 2)
    gauss_1d /= tf.reduce_sum(gauss_1d)  # Normalize

    # Create a 2D Gaussian kernel by outer product
    gauss_2d = tf.tensordot(gauss_1d, gauss_1d, axes=0)
    gauss_2d /= tf.reduce_sum(gauss_2d)  # Normalize

    return gauss_2d

def apply_gaussian_blur_4d(tensor_4d, kernel_size: int, sigma: float):
    """
    Applies Gaussian blur to a 4D tensor using a manually created Gaussian kernel.

    Parameters:
    tensor_4d (tf.Tensor): 4D tensor with shape [batch, height, width, channels].
    kernel_size (int): Size of the Gaussian kernel (e.g., 3, 5, 7).
    sigma (float): Standard deviation of the Gaussian.

    Returns:
    tf.Tensor: Blurred 4D tensor with the same shape as input.
    """
    # Create Gaussian kernel
    gauss_kernel = gaussian_kernel(kernel_size, sigma)
    gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]  # Add dimensions for channels

    # Repeat the kernel for each channel
    gauss_kernel = tf.repeat(gauss_kernel, tensor_4d.shape[-1], axis=-1)  # Shape [k, k, 1, channels]

    # Apply convolution to each channel independently
    blurred_tensor = tf.nn.depthwise_conv2d(
        tensor_4d,
        filter=gauss_kernel,
        strides=[1, 1, 1, 1],
        padding='SAME'
    )

    return blurred_tensor
  

