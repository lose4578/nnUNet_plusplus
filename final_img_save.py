import matplotlib
# matplotlib.use('TkAgg')

from matplotlib import pylab as plt
import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D
import os
import gc
import numpy as np


def hu_to_grayscale(volume, hu_min=-512, hu_max=512):
    # Clip at max and min values if specified
    if hu_min is not None or hu_max is not None:
        volume = np.clip(volume, hu_min, hu_max)

    # Scale to values between 0 and 1
    mxval = np.max(volume)
    mnval = np.min(volume)
    im_volume = (volume - mnval) / max(mxval - mnval, 1e-3)

    # Return values scaled to 0-255 range, but *not cast to uint8*
    # Repeat three times to make compatible with color overlay
    #     im_volume = 255 * im_volume
    return np.stack((im_volume, im_volume, im_volume), axis=-1)


def overlay(volume_ims, segmentation_ims, segmentation, alpha):
    # Get binary array for places where an ROI lives
    segbin = np.greater(segmentation, 0)
    # repeated_segbin = np.stack((segbin, segbin, segbin), axis=-1)
    # Weighted sum where there's a value to overlay

    overlayed = np.where(
        segbin,
        (alpha * segmentation_ims + (1 - alpha) * volume_ims),
        (volume_ims)
    )
    return overlayed


def mask_to_color(mask, k_color, t_color):
    shp_mask = mask.shape
    seg_color = np.zeros((shp_mask[0], shp_mask[1], 3), dtype=np.float32)
    # seg_color[mask[70] == 0] = [255, 255, 255]
    seg_color[mask == 1] = k_color
    seg_color[mask == 2] = t_color
    return seg_color


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        pass


k_color = [255, 0, 0]
t_color = [0, 255, 0]

k_t_color = [255, 255, 0]
t_t_color = [0, 0, 255]

vol_filename = './final_test/img/00068_0000.nii.gz'
seg_filename = './final_test/lable_true/00068.nii.gz'
pre_seg_filename1 = './final_test/lable_test/final_1/00068.nii.gz'
pre_seg_filename2 = './final_test/lable_test/final_2/00068.nii.gz'
pre_seg_filename3 = './final_test/lable_test/final_3/00068.nii.gz'
pre_seg_filename4 = './final_test/lable_test/final_4/00068.nii.gz'

_vol = nib.load(vol_filename).get_data()
_mask = nib.load(seg_filename).get_data()
_pre_mask_1 = nib.load(pre_seg_filename1).get_data()
_pre_mask_2 = nib.load(pre_seg_filename2).get_data()
_pre_mask_3 = nib.load(pre_seg_filename3).get_data()
_pre_mask_4 = nib.load(pre_seg_filename4).get_data()

_pre_mask_list = [_pre_mask_1, _pre_mask_2, _pre_mask_3, _pre_mask_4]

mkdir('./final_test/save_img/img_00068')
for index, j in enumerate([k for k in range(_mask.shape[0])]):
    vol = _vol[j]

    # if mask.sum() == 0 and pre_mask.sum() == 0:
    #     continue

    mask = _mask[j]
    mask = mask_to_color(mask, k_color, t_color)

    vol_img = hu_to_grayscale(vol)
    true_img = overlay(vol_img, mask, mask, 0.3)

    pre_img_list = []
    for final_index in range(4):
        pre_mask = _pre_mask_list[final_index][j]
        pre_mask = mask_to_color(pre_mask, k_t_color, t_t_color)
        pre_img = overlay(vol_img, pre_mask, pre_mask, 0.3)
        pre_img_list.append(pre_img)

    fig = plt.figure()
    height, width, channels = true_img.shape
    width = width * 3
    height = height * 2
    fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)

    ax0 = fig.add_subplot(231)
    ax1 = fig.add_subplot(232)
    ax2 = fig.add_subplot(233)
    ax3 = fig.add_subplot(235)
    ax4 = fig.add_subplot(236)
    ax0.imshow(true_img)
    ax0.axis('off')
    ax1.imshow(pre_img_list[0])
    ax1.axis('off')
    ax2.imshow(pre_img_list[1])
    ax2.axis('off')
    ax3.imshow(pre_img_list[2])
    ax3.axis('off')
    ax4.imshow(pre_img_list[3])
    ax4.axis('off')
    plt.savefig('./final_test/save_img/img_00068/' + str(index) + '.png', dpi=600)
