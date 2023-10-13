import numpy as np
from dataclasses import replace
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.compiler import Compiler
from ffcv.transforms.utils import fast_crop
import numbers
import numba as nb

class ColorJitter(Operation):
    """Add ColorJitter with probability.
    Operates on raw arrays (not tensors).

    Parameters
    ----------
    brightness (float or tuple of float (min, max)): How much to jitter brightness.
        brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
        or the given [min, max]. Should be non negative numbers.
    contrast (float or tuple of float (min, max)): How much to jitter contrast.
        contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
        or the given [min, max]. Should be non negative numbers.
    saturation (float or tuple of float (min, max)): How much to jitter saturation.
        saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
        or the given [min, max]. Should be non negative numbers.
    hue (float or tuple of float (min, max)): How much to jitter hue.
        hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
        Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be non negative.")
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(f"{name} values should be between {bound}")
        else:
            raise TypeError(f"{name} should be a single number or a list/tuple with length 2.")

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def generate_code(self):
        my_range = Compiler.get_iterator()
        apply_brightness = (self.brightness is not None)
        if apply_brightness:
            brightness_min, brightness_max = self.brightness
        apply_contrast = (self.contrast is not None)
        if apply_contrast:
            contrast_min, contrast_max = self.contrast
        apply_saturation = (self.saturation is not None)
        if apply_saturation:
            saturation_min, saturation_max = self.saturation
        apply_hue = (self.hue is not None)
        if apply_hue:
            hue_min, hue_max = self.hue
            
        def color_jitter(images, *_):
            def blend(img1, img2, ratio): return (ratio*img1 + (1-ratio)*img2).clip(0, 255).astype(img1.dtype)

            for i in my_range(images.shape[0]):
                
                fn_idx = np.random.permutation(4)

                for fn_id in nb.prange(len(fn_idx)):
                    # Brightness
                    if fn_idx[fn_id] == 0 and apply_brightness:
                        ratio_brightness = np.random.uniform(brightness_min, brightness_max)
                        images[i] = blend(images[i], 0, ratio_brightness)
                    
                    # Contrast
                    elif fn_idx[fn_id] == 1 and apply_contrast:
                        ratio_contrast = np.random.uniform(contrast_min, contrast_max)
                        gray = 0.2989 * images[i,:,:,0:1] + 0.5870 * images[i,:,:,1:2] + 0.1140 * images[i,:,:,2:3]
                        images[i] = blend(images[i], gray.mean(), ratio_contrast)
                    
                    # Saturation
                    elif fn_idx[fn_id] == 2 and apply_saturation:
                        ratio_saturation = np.random.uniform(saturation_min, saturation_max)
                        r, g, b = images[i,:,:,0], images[i,:,:,1], images[i,:,:,2]
                        l_img = (0.2989 * r + 0.587 * g + 0.114 * b).astype(images[i].dtype)
                        l_img3 = np.zeros_like(images[i])
                        for j in my_range(images[i].shape[-1]):
                            l_img3[:,:,j] = l_img
                        images[i] = blend(images[i], l_img3, ratio_saturation)

                    # Hue
                    elif fn_idx[fn_id] == 3 and apply_hue:
                        img = images[i] / 255.0
                        hue_factor = np.random.uniform(hue_min, hue_max)
                        hue_factor_radians = hue_factor * 2.0 * np.pi
                        cosA = np.cos(hue_factor_radians)
                        sinA = np.sin(hue_factor_radians)
                        hue_rotation_matrix =\
                        [[cosA + (1.0 - cosA) / 3.0, 1./3. * (1.0 - cosA) - np.sqrt(1./3.) * sinA, 1./3. * (1.0 - cosA) + np.sqrt(1./3.) * sinA],
                        [1./3. * (1.0 - cosA) + np.sqrt(1./3.) * sinA, cosA + 1./3.*(1.0 - cosA), 1./3. * (1.0 - cosA) - np.sqrt(1./3.) * sinA],
                        [1./3. * (1.0 - cosA) - np.sqrt(1./3.) * sinA, 1./3. * (1.0 - cosA) + np.sqrt(1./3.) * sinA, cosA + 1./3. * (1.0 - cosA)]]
                        hue_rotation_matrix = np.array(hue_rotation_matrix, dtype=img.dtype)
                        for row in nb.prange(img.shape[0]):
                            for col in nb.prange(img.shape[1]):
                                r, g, b = img[row, col, :]
                                img[row, col, 0] = r * hue_rotation_matrix[0, 0] + g * hue_rotation_matrix[0, 1] + b * hue_rotation_matrix[0, 2]
                                img[row, col, 1] = r * hue_rotation_matrix[1, 0] + g * hue_rotation_matrix[1, 1] + b * hue_rotation_matrix[1, 2]
                                img[row, col, 2] = r * hue_rotation_matrix[2, 0] + g * hue_rotation_matrix[2, 1] + b * hue_rotation_matrix[2, 2]
                        images[i] = np.asarray(np.clip(img * 255., 0, 255), dtype=np.uint8)

            return images

        color_jitter.is_parallel = True
        return color_jitter

    def declare_state_and_memory(self, previous_state):
        return (replace(previous_state, jit_mode=True), AllocationQuery(shape=previous_state.shape, dtype=previous_state.dtype))


class RandomGrayscale(Operation):
    '''
    Randomly convert image to grayscale with a probability of p (not tensors).

    Parameters
    ----------
    p : float
        probability to apply contrast
    '''
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def generate_code(self):
        my_range = Compiler.get_iterator()
        p = self.p

        def rgb_to_grayscale(images, *_):

            apply_grayscale = np.random.rand(images.shape[0]) < p
            for i in my_range(images.shape[0]):
                if apply_grayscale[i]:
                    r, g, b = images[i,:,:,0], images[i,:,:,1], images[i,:,:,2]
                    l_img = (0.2989 * r + 0.587 * g + 0.114 * b).astype(images[i].dtype)
                    l_img = np.expand_dims(l_img, axis=-1)
                    l_img = np.broadcast_to(l_img, images[0].shape)
                    images[i] = l_img

            return images

        rgb_to_grayscale.is_parallel = True
        return rgb_to_grayscale

    def declare_state_and_memory(self, previous_state):
        return (replace(previous_state, jit_mode=True), AllocationQuery(previous_state.shape, previous_state.dtype))

class ResizedCrop(Operation):
    def __init__(self, size, ratio=1.0):
        super().__init__()
        self.ratio = ratio
        self.size = size

    def generate_code(self):
        ratio = self.ratio
        def resized_crop(im, dst):

            i, j, h, w = fast_crop.get_center_crop(im.shape[0], im.shape[1], ratio)
            fast_crop.resize_crop(im, i, i + h, j, j + w, dst)
            return dst

        return resized_crop

    def declare_state_and_memory(self, previous_state):
        assert previous_state.jit_mode
        return replace(previous_state, shape=(self.size, self.size, 3)), AllocationQuery((self.size, self.size, 3), dtype=np.dtype('uint8'))