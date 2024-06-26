from rs_to_volume import *

from scipy import ndimage
from scipy.spatial.transform import Rotation
from skimage.transform import resize

import numpy as np
import SimpleITK as sitk
import time


# TODO
# 1. Check slice integrity before Func. get_z_spacing()
# 2. Func. rotate_volume() need to be modified by only rotation_matrix
# 3. add Padding() or Crop() to image volume


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")
        return result
    return wrapper


def get_volume_histogram(image_volume):
    max = np.max(image_volume)
    min = np.min(image_volume)
    histogram = np.histogram(image_volume, bins=int(max*0.9), range=(max*0.01, max*0.9))[0]
    return histogram


def get_transformation_matrix_for_pixel2patient(series_data):
    first_slice = series_data[0]
    offset = np.array(first_slice.ImagePositionPatient)
    row_spacing, column_spacing = first_slice.PixelSpacing
    slice_spacing = get_spacing_between_slices(series_data)
    row_direction, column_direction, slice_direction = get_slice_directions(first_slice)
    
    mat = np.identity(4, dtype=np.float32)
    mat[:3, 0] = row_direction * row_spacing
    mat[:3, 1] = column_direction * column_spacing
    mat[:3, 2] = slice_direction * slice_spacing
    mat[:3, 3] = offset

    return mat


def decompose_rotation_matrix(mat):
    r = Rotation.from_matrix(mat[:3, :3])
    angles = r.as_euler('zyx', degrees=True)
    return angles


def rotate_volume(volume, angles):
    # reshape = False -> keep in/out shape
    rotated_volume = ndimage.rotate(volume, angles[0], axes=(1, 2), reshape=True, order=2, mode='grid-constant')
    rotated_volume = ndimage.rotate(rotated_volume, angles[1], axes=(0, 2), reshape=True, order=2, mode='grid-constant')
    rotated_volume = ndimage.rotate(rotated_volume, angles[2], axes=(0, 1), reshape=True, order=2, mode='grid-constant')
    return rotated_volume


def mri_spacing_normalize(image_volume, origin_spacing=[3, 0.5, 0.5], target_spacing=[1, 1, 1]):
    """
    standard_scale = z_scale, y_scale, x_scale
    """
    z_space = origin_spacing[0] * image_volume.shape[0] / target_spacing[0]
    y_space = origin_spacing[1] * image_volume.shape[1] / target_spacing[1]
    x_space = origin_spacing[2] * image_volume.shape[2] / target_spacing[2]
    # print(first_slice.ImagePositionPatient)
    # print(spacing)                  # (x, y)
    # print(thickness)                # z
    # print(image_volume.shape)         # (z, y, x)
    # z = image_volume.shape[0] * thickness
    # y = image_volume.shape[1] * spacing[1]
    # x = image_volume.shape[2] * spacing[0]
    # print(z, y, x)
    normalized_img_volume = resize(image_volume, (z_space, y_space, x_space))
    # print(f"resize image volume: {image_volume.shape}")
    return normalized_img_volume



@timeit
def mri_n4_bias_field_correction(image_volume):
    input_sitk_img = sitk.GetImageFromArray(image_volume)
    input_sitk_img = sitk.Cast(input_sitk_img, sitk.sitkFloat32)
    output_sitk_img = sitk.N4BiasFieldCorrection(input_sitk_img, input_sitk_img > 0)
    n4_img_volume = sitk.GetArrayFromImage(output_sitk_img)
    return n4_img_volume

def calc_distance_3d(point1, point2):
    """point = [x, y, z]"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)

def get_z_spacing(sorted_ds_list):
    return calc_distance_3d(sorted_ds_list[0].ImagePositionPatient, sorted_ds_list[-1].ImagePositionPatient) / (len(sorted_ds_list) - 1)
     


import matplotlib.pyplot as plt
def test_plot_value_histogram(image_volume):
    values = image_volume.flatten()
    plt.figure(figsize=(15, 6))
    n, bins, patches = plt.hist(values, bins=1000)  # Adjust the number of bins as needed
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Voxel Value')
    plt.ylim(0, max(n) * 0.5)
    plt.show()



if __name__ == "__main__":
    
    img_dir = "./Bseries_0"    # MR series資料夾路徑
    
    img_ds_list = load_sorted_image_series(img_dir)
    img_volume = get_img_volume(img_ds_list)    # shape = [z, y, x]

    first_slice = img_ds_list[0]
    for index in range(len(img_ds_list) - 1):
        dist = calc_distance_3d(img_ds_list[index].ImagePositionPatient, img_ds_list[index+1].ImagePositionPatient)
        print("distance:", dist, "|| thickness:", img_ds_list[index].SliceThickness)


    # ################ n4 bias field correction ###########################
    # print(first_slice.StudyDescription, first_slice.SeriesDescription)
    # img_volume = img_volume
    # values = img_volume.flatten()
    # plt.figure(figsize=(15, 6))
    # n, bins, patches = plt.hist(values, bins=1000)  # Adjust the number of bins as needed
    # plt.xlabel('Pixel Value')
    # plt.ylabel('Frequency')
    # plt.title(f'Histogram of Voxel Value {first_slice.StudyDescription, first_slice.SeriesDescription}')
    # plt.ylim(0, max(n) * 0.5)
    # plt.savefig(f'{img_dir}_histogram.png')
    # plt.show()
    
    # for i in range(0, 11):
    #     img_dir = f"./Pseries_{str(i)}"    # MR series資料夾路徑
        
    #     img_ds_list = load_sorted_image_series(img_dir)
    #     img_volume = get_img_volume(img_ds_list)    # shape = [z, y, x]

    #     first_slice = img_ds_list[0]
    #     for index in range(len(img_ds_list) - 1):
    #         dist = calc_distance_3d(img_ds_list[index].ImagePositionPatient, img_ds_list[index+1].ImagePositionPatient)
    #         print("distance:", dist, "|| thickness:", img_ds_list[index].SliceThickness)

    #     print(first_slice.StudyDescription, first_slice.SeriesDescription)
    #     img_volume = img_volume
    #     values = img_volume.flatten()
    #     plt.figure(figsize=(15, 6))
    #     n, bins, patches = plt.hist(values, bins=1000)  # Adjust the number of bins as needed
    #     plt.xlabel('Pixel Value')
    #     plt.ylabel('Frequency')
    #     plt.title(f'Histogram of Voxel Value {first_slice.StudyDescription, first_slice.SeriesDescription, first_slice.PatientID, first_slice.ScanningSequence}')
    #     plt.ylim(0, max(n) * 0.5)
    #     plt.savefig(f'{img_dir}_histogram.png')
    #     # plt.show()



    # ################ n4 bias field correction ###########################
    test_plot_value_histogram(img_volume)    
    n4b_cor_volume = mri_n4_bias_field_correction(img_volume)
    print(f"n4 bias field correction volume: {n4b_cor_volume.shape}")
    test_plot_value_histogram(n4b_cor_volume)
    # ######################################################################


    # ###################### resize image volume #########################

    # z, y, x
    z_spacing = get_z_spacing(img_ds_list)
    print(z_spacing)
    ori_spacing = [z_spacing, first_slice.PixelSpacing[1], first_slice.PixelSpacing[0]]

    # volume shape = [z, y, x], ori_spacing = [z, y, x], target_spacing=[z, y, x]
    nor_img_volume = mri_spacing_normalize(img_volume,
                                           origin_spacing=ori_spacing,
                                           target_spacing=[5, 0.5, 0.5])
    print(f"resize image volume: {nor_img_volume.shape}")
    # ####################################################################


    # ################# rotate volume ################
    # pixel_patient_mat = get_transformation_matrix_for_pixel2patient(img_ds_list)     # 把影像坐標系旋轉至與病人坐標系同方向 須注意這裡僅作旋轉不做縮放
    # print(pixel_patient_mat)

    # angles = decompose_rotation_matrix(pixel_patient_mat)   # (z, y, x)
    # print(angles)

    # rotated_img_volume = rotate_volume(img_volume, angles)
    # print(rotated_img_volume.shape)
    # ################################################

    # ################ derotate volume ###############
    # derotated_img_volume = rotate_volume(rotated_img_volume, [-angles[0], -angles[1], -angles[2]])
    # print(derotated_img_volume.shape)
    # ################################################



    # 以下測試用的code

    # import matplotlib.pyplot as plt
    # from skimage.transform import resize

    ############### test img volume ########################
    # fig, axs = plt.subplots(5, 5, figsize=(15, 15))
    # for i, z in enumerate(range(0, img_volume.shape[0], img_volume.shape[0] // 24)):
    #     axial_slice = img_volume[z, :, :]
    #     axial_slice = np.clip(axial_slice, np.max(axial_slice)*0.1, np.max(axial_slice)*0.9) 
    #     axs[i // 5, i % 5].imshow(axial_slice, cmap='gray')
    # plt.show()

    # for z in range(0, img_volume.shape[0], 10):
    #     axial_slice = img_volume[z, :, :]
    #     axial_slice = np.clip(axial_slice, np.max(axial_slice)*0.1, np.max(axial_slice)*0.9) 
    #     plt.imshow(axial_slice, cmap='gray')
    #     plt.show()

    # for z in range(0, n4_img_volume.shape[0], 10):
    #     axial_slice = n4_img_volume[z, :, :]
    #     axial_slice = np.clip(axial_slice, np.max(axial_slice)*0.1, np.max(axial_slice)*0.9) 
    #     plt.imshow(axial_slice, cmap='gray')
    #     plt.show()

    # 比較兩組影像
    # for z in range(0, img_volume.shape[0], 10):
    #     fig, axs = plt.subplots(1, 2, figsize=(15, 15))
    #     img_axial_slice = img_volume[z, :, :]
    #     img_axial_slice = np.clip(img_axial_slice, np.max(img_axial_slice)*0.1, np.max(img_axial_slice)*0.9) 
    #     dust_axial_slice = n4_img_volume[z, :, :]
    #     dust_axial_slice = np.clip(img_axial_slice, np.max(img_axial_slice)*0.1, np.max(img_axial_slice)*0.9)

    #     axs[0].imshow(img_axial_slice, cmap='gray')
    #     axs[1].imshow(dust_axial_slice, cmap='gray')
    #     plt.show()



    # for z in range(0, rotated_img_volume.shape[0], 10):
    #     axial_slice = rotated_img_volume[z, :, :]
    #     axial_slice = np.clip(axial_slice, np.max(axial_slice)*0.1, np.max(axial_slice)*0.9) 
    #     plt.imshow(axial_slice, cmap='gray')
    #     plt.show()

    # for z in range(0, rotated_img_volume.shape[0], 10):
    # axial_slice = rotated_img_volume[z, :, :]
    # axial_slice = np.clip(axial_slice, np.max(axial_slice)*0.1, np.max(axial_slice)*0.9) 
    # plt.imshow(axial_slice, cmap='gray')
    # plt.show()