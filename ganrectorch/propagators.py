import torch
import torch.fft
import torchvision.transforms as transforms
import kornia

class TomoRadon:

    def __init__(self, rec, ang):
        self.rec = rec
        self.ang = ang

    def compute(self):
        nang = self.ang.shape[0]
        img = self.rec.permute(3, 1, 2, 0)
        img = img.repeat(nang, 1, 1, 1)
        img = kornia.geometry.rotate(img, -self.ang)
        sino = torch.mean(img, 1)
        sino = sino.permute(2, 0, 1)
        sino = sino.view(sino.shape[0], sino.shape[1], sino.shape[2], 1)
        return sino


class TensorRadon:

    def __init__(self, rec, ang, psi):
        self.strain_tensor = rec
        self.ang = ang
        self.psi = psi

    def tfnor_data(self, img):
        img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))
        return img

    def compute(self):
        detector_rows = self.strain_tensor.shape[0]
        detector_columns = self.strain_tensor.shape[1]
        strain_tensor = self.strain_tensor.float()
        vol_mask = torch.zeros((detector_rows, detector_columns, detector_columns))
        vol_mask = torch.sum(torch.abs(strain_tensor), axis=3) > 0.0
        vol_mask = vol_mask.view(-1, detector_columns, detector_columns, 1)
        vol_mask = vol_mask.float()
        angles = self.ang.float()
        thickness = TomoRadon(vol_mask, angles).compute()
        thickness = thickness.squeeze()
        strain_tensor = strain_tensor.permute(3, 1, 2, 0)
        proj_strain_comp = TomoRadon(strain_tensor, angles).compute()
        proj_strain_comp = proj_strain_comp.squeeze()
        cos_squared = torch.pow(torch.cos(angles), 2).unsqueeze(1)
        sin_squared = torch.pow(torch.sin(angles), 2).unsqueeze(1)
        cos_psi_squared = torch.pow(torch.cos(self.psi), 2)
        sin_psi_squared = torch.pow(torch.sin(self.psi), 2)
        sin_2angles = torch.sin(2 * angles).unsqueeze(1)
        sin_angles_sin_2psi = torch.sin(angles) * torch.sin(2 * self.psi).unsqueeze(1)
        cos_angles_sin_2psi = torch.cos(angles) * torch.sin(2 * self.psi).unsqueeze(1)
        proj_strain_ws = (proj_strain_comp[0] * cos_squared * sin_psi_squared + 
                          proj_strain_comp[1] * sin_squared * sin_psi_squared +
                          proj_strain_comp[2] * cos_psi_squared +
                          proj_strain_comp[3] * sin_2angles * sin_psi_squared +
                          proj_strain_comp[4] * sin_angles_sin_2psi +
                          proj_strain_comp[5] * cos_angles_sin_2psi)
        tensor_sino = proj_strain_ws
        tensor_sino = tensor_sino.view(1, tensor_sino.shape[0], tensor_sino.shape[1], 1)
        return tensor_sino


class PhaseFresnel:

    def __init__(self, phase, absorption, ff, px):
        self.phase = phase
        self.absorption = absorption
        self.ff = ff
        self.px = px

    def compute(self):
        paddings = torch.tensor([[self.px // 2, self.px // 2], [self.px // 2, self.px // 2]])
        pvalue = torch.mean(self.phase[:100, :])
        self.phase = torch.nn.functional.pad(self.phase, paddings, 'reflect')
        self.absorption = torch.nn.functional.pad(self.absorption, paddings, 'reflect')
        abfs = torch.complex(-self.absorption, self.phase)
        abfs = torch.exp(abfs)
        ifp = torch.abs(torch.fft.ifft2(self.ff * torch.fft.fft2(abfs))) ** 2
        ifp = ifp.view(ifp.shape[0], ifp.shape[1], 1)
        ifp = transforms.CenterCrop(ifp.shape[0]//2)(ifp)
        ifp = transforms.Normalize(0, 1)(ifp)
        ifp = ifp.view(1, ifp.shape[0], ifp.shape[1], 1)
        return ifp


class PhaseFraunhofer:

    def __init__(self, phase, absorption, shift_factor=100000):
        self.phase = phase
        self.absorption = absorption
        self.shift_factor = shift_factor

    def compute(self):
        wf = torch.complex(self.absorption, self.phase)
        ifp = torch.square(torch.abs(torch.fft.fft2(wf)))
        ifp = torch.log(ifp + self.shift_factor)
        ifp = torch.fft.fftshift(ifp)
        ifp = ifp.view(1, ifp.shape[0], ifp.shape[1], 1)
        ifp = transforms.Normalize(0, 1)(ifp)
        return ifp