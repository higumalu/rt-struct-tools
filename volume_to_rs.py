from rs_to_volume import get_mask_volume
from rs_builder import create_rtstruct_dataset
from rs_builder import edit_required_elements
from rs_builder import add_mask3d_into_rsds

from pydicom import dcmread
from pydicom.dataset import Dataset
from glob import glob
from typing import List

import os
import numpy as np


def load_ds_list_from_dir(dir_path):
    ds_list = []
    dcm_list = glob(os.path.join(dir_path, "*.dcm"))
    for dcm in dcm_list:
        ds_list.append(dcmread(dcm))
    return ds_list


def sort_image_series(ds_series_list: List[Dataset]) -> List[Dataset]:
    if len(ds_series_list) <= 1:
        raise Exception("DICOM Images <= 1 in input path")
    ds_series_list.sort(key=get_slice_position, reverse=False)
    return ds_series_list


def get_slice_position(series_slice: Dataset):
    _, _, slice_direction = get_slice_directions(series_slice)
    return np.dot(slice_direction, series_slice.ImagePositionPatient)


def get_slice_directions(series_slice: Dataset):
    orientation = series_slice.ImageOrientationPatient
    row_direction = np.array(orientation[:3])
    column_direction = np.array(orientation[3:])
    slice_direction = np.cross(row_direction, column_direction)
    if not np.allclose(
        np.dot(row_direction, column_direction), 0.0, atol=1e-3
    ) or not np.allclose(np.linalg.norm(slice_direction), 1.0, atol=1e-3):
        raise Exception("Invalid Image Orientation (Patient) attribute")
    return row_direction, column_direction, slice_direction

##############################################################################
############################ Build RS ########################################

def create_rs_from_img_series_ds(img_ds_list):
    img_ds_list = sort_image_series(img_ds_list)
    rs_ds = create_rtstruct_dataset(img_ds_list)
    return rs_ds



if __name__ == "__main__":

    img_dir = "data/CT"
    rs_dcm = "data/RTSTRUCT.dcm"

    img_ds_list = load_ds_list_from_dir(img_dir)
    rs_ds = dcmread(rs_dcm)

    img_ds_list = sort_image_series(img_ds_list)

    mask_volume_4d = get_mask_volume(rs_ds, img_ds_list)


    ######################################## START CREATE RS #########################################################



    mask_volume = mask_volume_4d[9]

    print(mask_volume.shape, mask_volume.dtype, mask_volume.sum())

    
    # create blank rs
    blank_rs_ds = create_rs_from_img_series_ds(img_ds_list)
    blank_rs_ds = edit_required_elements(blank_rs_ds, structure_set_label="RSlabel")

    # define contour information
    mask_volume = mask_volume.astype(int)
    roi_color = [0, 255, 0]
    roi_number = 1
    roi_name = "test_roi_name"
    roi_description = "test_desc_for_rs_roi"
    
    
    writen_rs_ds = add_mask3d_into_rsds(
                            rs_ds=blank_rs_ds,
                            mask_volume=mask_volume,
                            image_series=img_ds_list,
                            roi_color=roi_color,
                            roi_number=roi_number,
                            roi_name=roi_name,
                            roi_description=roi_description,
                    )



    # define contour information
    mask_volume = mask_volume_4d[11].astype(int)
    roi_color = [255, 0, 0]
    roi_number = 2
    roi_name = "test_roi_02"
    roi_description = "test_desc_for_rs_roi"
    
    
    writen_rs_ds = add_mask3d_into_rsds(
                            rs_ds=writen_rs_ds,
                            mask_volume=mask_volume,
                            image_series=img_ds_list,
                            roi_color=roi_color,
                            roi_number=roi_number,
                            roi_name=roi_name,
                            roi_description=roi_description,
                    )
    
    writen_rs_ds.save_as("data/RTSTRUCT_test.dcm", write_like_original=False)