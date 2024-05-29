import tensorflow as tf
from dataclasses import dataclass
from ganrectf.ganrec import GANrec

def tomography(projections, angles):
    @dataclass
    class InputData:
        projections: tf.Tensor
        angles: tf.Tensor
    
    if projections.dim == 2:
        InputData(projections=projections, angles=angles)
        recon = GANrec(InputData).recon
    elif projections.dim == 3:
        for slice in range(projections.shape[1]):
            InputData(projections=projections, angles=angles)
            recon = GANrec(InputData).recon
            
        
        
    
    
    
    