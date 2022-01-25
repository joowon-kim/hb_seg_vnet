# VNet keras
#
# Joo-won Kim

from tensorflow import keras
import numpy as np
import nibabel as nib
import random


# define transformations
def normalization(dat_input, dat_target, hdr=None, aff=None):
    """Normalize intensity values of dat_input between 0 and 1"""
    dat = dat_input.copy()
    dat -= dat.min()
    dat /= dat.max()
    dat[dat>1.0] = 1.0
    return dat, dat_target


def flip_x(dat_input, dat_target, hdr=None, aff=None):
    """Flip x dimension (left-right)"""
    return dat_input[::-1,...], dat_target[::-1,...]

def random_flip_x(dat_input, dat_target, hdr=None, aff=None, threshold=0.5):
    """Flip x dimension (left-right)"""
    if random.random() >= threshold:
        return dat_input[::-1,...], dat_target[::-1,...]
    else:
        return dat_input, dat_target

class Add_noise(object):
    """Add random noise"""

    def __init__(self, noise_level=0.01):
        """normal noise std := dat.std()*noise_level"""
        self.noise_level = noise_level

    def __call__(self, dat_input, dat_target=None, hdr=None, aff=None):
        noise = np.random.normal(0, dat_input.mean()*self.noise_level, dat_input.shape)
        return dat_input + noise, dat_target


def normalization_single(dat_input, dat_target=None, hdr=None, aff=None):
    """Normalize intensity values of dat_input between 0 and 1"""
    dat = dat_input.copy()
    dat -= dat.min()
    dat /= dat.max()
    dat[dat>1.0] = 1.0
    return dat


class Crop_singel_image(object):
    """Crop single 3D image. Default numbers are for MNI 152 0.7 mm template"""
    def __init__(self, img_size=(64,64,64), center=(128,147,105)):
        self.img_size = img_size
        self.center = center

    def __call__(self, dat_input, dat_target=None, hdr=None, aff=None):
        min_xyz = [self.center[i] - self.img_size[i]//2 for i in range(len(self.img_size))]
        return dat_input[
                min_xyz[0]:min_xyz[0]+self.img_size[0],
                min_xyz[1]:min_xyz[1]+self.img_size[1],
                min_xyz[2]:min_xyz[2]+self.img_size[2],
                ]


def pad_single_image(dat_input, dat_target=None, hdr=None, aff=None, img_size=(260,311,260), center=(128,147,105)):
    """Pad single 3D image. Default numbers are for MNI 152 0.7 mm template"""
    dat_out = np.zeros(img_size, dtype=dat_input.dtype)
    min_xyz = [center[i] - dat_input.shape[i]//2 for i in range(len(img_size))]
    dat_out[
            min_xyz[0]:min_xyz[0]+dat_input.shape[0],
            min_xyz[1]:min_xyz[1]+dat_input.shape[1],
            min_xyz[2]:min_xyz[2]+dat_input.shape[2],
            ] = dat_input
    return dat_out


class Crop_pad(object):
    """Crop or pad image"""

    def __init__(self, target_size, padding_value=(0,0), is_training=False):
        self.target_size = target_size
        self.padding_value = padding_value
        self.is_training = is_training

    def __get_start_idx__(self, di):
        if self.is_training:
            return random.randint(0, di)
        else:
            return di//2

    def __call__(self, dat_input, dat_target=None, hdr=None, aff=None):
        lst_ori_start_idx, lst_ori_end_idx = self.crop(dat_input, dat_target)
        lst_mod_start_idx, lst_mod_end_idx = self.pad(dat_input, dat_target)

        dat_input_mod = np.full(
                self.target_size,
                fill_value=self.padding_value[0],
                dtype=dat_input.dtype
                )
        dat_input_mod[
                lst_mod_start_idx[0]:lst_mod_end_idx[0],
                lst_mod_start_idx[1]:lst_mod_end_idx[1],
                lst_mod_start_idx[2]:lst_mod_end_idx[2],
                ...] = dat_input[
                        lst_ori_start_idx[0]:lst_ori_end_idx[0],
                        lst_ori_start_idx[1]:lst_ori_end_idx[1],
                        lst_ori_start_idx[2]:lst_ori_end_idx[2],
                        ...]

        if dat_target is None:
            return dat_input_mod

        dat_target_mod = np.full(
                self.target_size,
                fill_value=self.padding_value[1],
                dtype=dat_target.dtype
                )
        dat_target_mod[
                lst_mod_start_idx[0]:lst_mod_end_idx[0],
                lst_mod_start_idx[1]:lst_mod_end_idx[1],
                lst_mod_start_idx[2]:lst_mod_end_idx[2],
                ...] = dat_target[
                        lst_ori_start_idx[0]:lst_ori_end_idx[0],
                        lst_ori_start_idx[1]:lst_ori_end_idx[1],
                        lst_ori_start_idx[2]:lst_ori_end_idx[2],
                        ...]
        return dat_input_mod, dat_target_mod

    def pad(self, dat_input, dat_target=None):
        shape = dat_input.shape[:3]
        lst_start_idx = [0] * 3
        lst_end_idx = [0] * 3
        for i in range(3):
            di = self.target_size[i] - shape[i]
            start_idx = 0 if di < 0 else self.__get_start_idx__(di)
            end_idx = self.target_size[i] if di < 0 else start_idx + shape[i]
            lst_start_idx[i] = start_idx
            lst_end_idx[i] = end_idx
        return lst_start_idx, lst_end_idx

    def crop(self, dat_input, dat_target=None):
        shape = dat_input.shape[:3]
        lst_start_idx = [0] * 3
        lst_end_idx = [0] * 3
        for i in range(3):
            di = shape[i] - self.target_size[i]
            start_idx = 0 if di < 0 else self.__get_start_idx__(di)
            end_idx = shape[i] if di < 0 else start_idx + self.target_size[i]
            lst_start_idx[i] = start_idx
            lst_end_idx[i] = end_idx
        return lst_start_idx, lst_end_idx


class LoadData(keras.utils.Sequence):
    """Helper to iterate over the data."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths, lst_transformations=[]):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.lst_transformations = lst_transformations

    def __len__(self):
        return int(np.ceil(len(self.input_img_paths) / self.batch_size))
        #return len(self.input_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        this_batch_size = self.batch_size if idx < self.__len__() - 1 else len(self.input_img_paths) % self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + this_batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + this_batch_size]
        x = np.zeros((this_batch_size,) + self.img_size + (1,), dtype="float32")
        y = np.zeros((this_batch_size,) + self.img_size + (1,), dtype="uint8")

        for j, (path_input, path_target) in enumerate(zip(batch_input_img_paths, batch_target_img_paths)):
            img_in = nib.load(path_input)
            img_tg = nib.load(path_target)
            dat_in = img_in.get_fdata().astype("float32")
            dat_tg = img_tg.get_fdata().astype("uint8")
            for transformation in self.lst_transformations:
                dat_in, dat_tg = transformation(dat_in, dat_tg, img_in.header, img_in.affine)
            x[j,:,:,:,0] = dat_in
            y[j,:,:,:,0] = dat_tg
        return x, y


class LoadPredictData(keras.utils.Sequence):
    """Helper to iterate over the data."""

    def __init__(self, batch_size, img_size, input_img_paths, lst_transformations=[]):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        #self.target_img_paths = target_img_paths
        self.lst_transformations = lst_transformations

    def __len__(self):
        return int(np.ceil(len(self.input_img_paths) / self.batch_size))
        #return len(self.input_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        #batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        #y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        y = None

        #for j, (path_input, path_target) in enumerate(zip(batch_input_img_paths, batch_target_img_paths)):
        for j, path_input in enumerate(batch_input_img_paths):
            img_in = nib.load(path_input)
            #img_tg = nib.load(path_target)
            dat_in = img_in.get_fdata().astype("float32")
            #dat_tg = img_tg.get_fdata().astype("uint8")
            for transformation in self.lst_transformations:
                dat_in = transformation(dat_in, hdr=img_in.header, aff=img_in.affine)
            x[j,:,:,:,0] = dat_in
            #y[j,:,:,:,0] = dat_tg
        return x, y

