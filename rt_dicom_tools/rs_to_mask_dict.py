import cv2
import pydicom
import numpy as np

from .coordinate_transform import get_patient_to_pixel_transformation_matrix, apply_transformation_to_3d_points
from .rs_to_volume import load_sorted_image_series


def add_to_dict(dictionary, key, value):
    if key in dictionary:
        dictionary[key].append(value)
    else:
        dictionary[key] = [value]
    return None


def get_mask_dict(rs_ds, img_ds_list):  # dict{key: roi_name, value: mask}
    x_shape, y_shape = img_ds_list[0].pixel_array.shape
    print(x_shape, y_shape)

    transfer_matrix = get_patient_to_pixel_transformation_matrix(img_ds_list)
    print(transfer_matrix)

    roi_num_name_dict = {}
    for ssroi_seq in rs_ds.StructureSetROISequence:
        roi_name = ssroi_seq.ROIName
        roi_num = ssroi_seq.ROINumber
        roi_num_name_dict[roi_num] = roi_name

    organ_ctr_list_dict = {}
    for ctr_seq in rs_ds.ROIContourSequence:    # loop by organ
        roi_num = ctr_seq.ReferencedROINumber
        ctr_list_dict = {}
        
        for ctr in ctr_seq.ContourSequence:
            ctr_data = ctr.ContourData
            reshape_ctr_data = np.reshape(ctr_data, [len(ctr_data) // 3, 3])    # 1 * n --> 3 * (n/3)   (x, y, z)
            transfer_ctr_data = apply_transformation_to_3d_points(reshape_ctr_data, transfer_matrix)
            transfer_ctr_data = np.round(transfer_ctr_data)
            z_index = int(transfer_ctr_data[0, 2])
            cv_ctr = np.array(transfer_ctr_data[:, 0:2], np.int32)
            add_to_dict(ctr_list_dict, z_index, cv_ctr)
        organ_ctr_list_dict[roi_num] = ctr_list_dict

    organ_mask_dict = {}
    for roi_num, ctr_list_dict in organ_ctr_list_dict.items():
        roi_name = roi_num_name_dict[roi_num]
        organ_mask_dict[roi_name] = {}
        organ_mask_volume = np.zeros((len(img_ds_list), y_shape, x_shape), np.uint8)
        for z_index, cv_ctr in ctr_list_dict.items():
            organ_mask_volume[z_index, :, :] = cv2.fillPoly(organ_mask_volume[z_index, :, :], cv_ctr, (1,))
        organ_mask_dict[roi_name] = organ_mask_volume

    return organ_mask_dict


def main():
    ct_dir_path = "./data/CT"
    rs_dcm_path = "./data/RS_A/RTSTRUCT.dcm"

    img_ds_list = load_sorted_image_series(ct_dir_path)
    rs_ds = pydicom.dcmread(rs_dcm_path)

    ##################### Start Proc ##############################################
    organ_mask_dict = get_mask_dict(rs_ds, img_ds_list)
    ###############################################################################

    print(organ_mask_dict.keys())
    print(organ_mask_dict['A_Aorta'].shape, organ_mask_dict['A_Aorta'].sum())
    mask = organ_mask_dict['A_Aorta']

    import matplotlib.pyplot as plt

    for i in range(0, mask.shape[0], 10):
        print(i)
        plt.figure(figsize=(15, 6))
        plt.imshow(mask[i, :, :])
        plt.show()