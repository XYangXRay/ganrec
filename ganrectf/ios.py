import tensorflow as tf
from dataclasses import dataclass

@dataclass
class InputData:
    projections: tf.Tensor
    angles: tf.Tensor

def organize_input(projections, angles):
    """
    Organize the input data "projections" and "angles" into a data class.

    Args:
    projections (tf.Tensor): A tensor of shape [batch_size, nang, px, 1] representing the projections.
    angles (tf.Tensor): A tensor of shape [batch_size, nang, 1] representing the angles.

    Returns:
    InputData: A data class containing the projections and angles tensors.
    """
    # Ensure angles have the right shape
    angles = tf.expand_dims(angles, axis=-1)  # Add a channel dimension

    return InputData(projections=projections, angles=angles)