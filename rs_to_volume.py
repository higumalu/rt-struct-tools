
from pydicom import dcmread
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from typing import List

import os
import numpy as np
import cv2 as cv

# *** func ref. by rt-utils ***

def load_sorted_image_series(dicom_series_path: str):
    """
    File contains helper methods for loading / formatting DICOM images and contours
    """

    series_data = load_dcm_images_from_path(dicom_series_path)

    if len(series_data) == 0:
        raise Exception("No DICOM Images found in input path")

    # Sort slices in ascending order
    series_data.sort(key=get_slice_position, reverse=False)

    return series_data



def load_dcm_images_from_path(dicom_series_path: str) -> List[Dataset]:
    series_data = []
    for root, _, files in os.walk(dicom_series_path):
        for file in files:
            try:
                ds = dcmread(os.path.join(root, file))
                if hasattr(ds, "pixel_array"):
                    series_data.append(ds)

            except Exception:
                # Not a valid DICOM file
                continue

    return series_data

def create_empty_series_mask(series_data):
    ref_dicom_image = series_data[0]
    mask_dims = (
        len(series_data),
        int(ref_dicom_image.Columns),
        int(ref_dicom_image.Rows),
    )
    mask = np.zeros(mask_dims).astype(bool)
    return mask


def get_slice_position(series_slice: Dataset):
    _, _, slice_direction = get_slice_directions(series_slice)
    return np.dot(slice_direction, series_slice.ImagePositionPatient)

def get_distance_betweem_slices(sorted_image_series):
    if False in [hasattr(ds, "ImagePositionPatient") for ds in sorted_image_series]:
        raise Exception("No ImagePositionPatient attribute")
    image_position_list = np.asanyarray([ds.ImagePositionPatient for ds in sorted_image_series]).astype(np.float32)
    distance_list = np.linalg.norm(image_position_list[1:] - image_position_list[:-1], axis=1)
    distance_between_slices = np.mean(distance_list)
    distance_between_slices = self_round(distance_between_slices, 1)
    if np.std(distance_list) > 0.01:
        print("distance_between_slices_sd:{}".format(np.std(distance_list)))
    return distance_between_slices


def get_spacing_between_slices(series_data):
    if len(series_data) > 1:
        first = get_slice_position(series_data[0])
        last = get_slice_position(series_data[-1])
        return (last - first) / (len(series_data) - 1)

    # Return nonzero value for one slice just to make the transformation matrix invertible
    return 1.0


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


def get_transformation_matrix_for_patient2pixel(series_data):
    first_slice = series_data[0]

    offset = np.array(first_slice.ImagePositionPatient)
    row_spacing, column_spacing = first_slice.PixelSpacing
    slice_spacing = get_spacing_between_slices(series_data)
    row_direction, column_direction, slice_direction = get_slice_directions(first_slice)
    # row_direction, column_direction, slice_direction = [1, 0, 0], [0, 1, 0], [0, 0, 1]
    # M = [ rotation&scaling   translation ]
    #     [        0                1      ]
    #
    # inv(M) = [ inv(rotation&scaling)   -inv(rotation&scaling) * translation ]
    #          [          0                                1                  ]

    linear = np.identity(3, dtype=np.float32)
    linear[0, :3] = row_direction / row_spacing
    linear[1, :3] = column_direction / column_spacing
    linear[2, :3] = slice_direction / slice_spacing

    mat = np.identity(4, dtype=np.float32)
    mat[:3, :3] = linear
    mat[:3, 3] = offset.dot(-linear.T)

    return mat


def get_slice_contour_data(series_slice: Dataset, contour_sequence: Sequence):
    slice_contour_data = []

    # Traverse through sequence data and get all contour data pertaining to the given slice
    for contour in contour_sequence:
        for contour_image in contour.ContourImageSequence:
            if contour_image.ReferencedSOPInstanceUID == series_slice.SOPInstanceUID:
                slice_contour_data.append(contour.ContourData)
    return slice_contour_data


def create_empty_slice_mask(series_slice):
    mask_dims = (int(series_slice.Columns), int(series_slice.Rows))
    mask = np.zeros(mask_dims).astype(bool)
    return mask


def apply_transformation_to_3d_points(
    points: np.ndarray, transformation_matrix: np.ndarray
):
    """
    * Augment each point with a '1' as the fourth coordinate to allow translation
    * Multiply by a 4x4 transformation matrix
    * Throw away added '1's
    """
    vec = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    return vec.dot(transformation_matrix.T)[:, :3]


def get_slice_mask_from_slice_contour_data(
    series_slice: Dataset, slice_contour_data, transformation_matrix: np.ndarray
):
    # Go through all contours in a slice, create polygons in correct space and with a correct format 
    # and append to polygons array (appropriate for fillPoly) 
    polygons = []
    for contour_coords in slice_contour_data:
        reshaped_contour_data = np.reshape(contour_coords, [len(contour_coords) // 3, 3])
        translated_contour_data = apply_transformation_to_3d_points(reshaped_contour_data, transformation_matrix)
        polygon = [np.around([translated_contour_data[:, :2]]).astype(np.int32)]
        polygon = np.array(polygon).squeeze()
        polygons.append(polygon)
    slice_mask = create_empty_slice_mask(series_slice).astype(np.uint8)
    cv.fillPoly(img=slice_mask, pts = polygons, color = 1)
    return slice_mask


def get_contour_sequence_by_roi_number(rs_ds, roi_number):
    for roi_contour in rs_ds.ROIContourSequence:

        # Ensure same type
        if str(roi_contour.ReferencedROINumber) == str(roi_number):
            if hasattr(roi_contour, "ContourSequence"):
                return roi_contour.ContourSequence
            else:
                return Sequence()

    raise Exception(f"Referenced ROI number '{roi_number}' not found")


def create_series_mask_from_contour_sequence(img_ds_list, contour_sequence: Sequence):
    mask = create_empty_series_mask(img_ds_list)
    transformation_matrix = get_transformation_matrix_for_patient2pixel(img_ds_list)

    # Iterate through each slice of the series, If it is a part of the contour, add the contour mask
    for i, series_slice in enumerate(img_ds_list):
        slice_contour_data = get_slice_contour_data(series_slice, contour_sequence)
        if len(slice_contour_data):
            mask[i, :, :] = get_slice_mask_from_slice_contour_data(
                series_slice, slice_contour_data, transformation_matrix
            )
    print(f'mask shape: {mask.shape}')
    return mask


def get_roi_mask_list(rs_ds, img_ds_list) -> np.ndarray:
    """
    Returns the 3D binary mask of the ROI with the given input name
    """
    mask_list = []

    for structure_roi in rs_ds.StructureSetROISequence:

        contour_sequence = get_contour_sequence_by_roi_number(rs_ds, structure_roi.ROINumber)
        
        binary_mask = create_series_mask_from_contour_sequence(img_ds_list, contour_sequence)
        
        mask_list.append(binary_mask)

    return mask_list

def get_roi_mask_4d(rs_ds, img_ds_list) -> np.ndarray:
    """
    Returns the 3D binary mask of the ROI with the given input name
    """
    mask_list = []
    for structure_roi in rs_ds.StructureSetROISequence:
        print(structure_roi.ROIName)
        contour_sequence = get_contour_sequence_by_roi_number(rs_ds, structure_roi.ROINumber)
        
        binary_mask = create_series_mask_from_contour_sequence(img_ds_list, contour_sequence)
        mask_list.append(binary_mask)
        
    mask_4d = np.stack(mask_list, axis=0)
    # print(mask_4d.shape)
    return mask_4d

def get_roi_mask_dict(rs_ds, img_ds_list):
    mask_dict = {}
    for structure_roi in rs_ds.StructureSetROISequence:

        # Get the contour sequence for the current ROI
        contour_sequence = get_contour_sequence_by_roi_number(rs_ds, structure_roi.ROINumber)
        
        # Create a 4D binary mask from the contour sequence and update the mask
        binary_mask = create_series_mask_from_contour_sequence(img_ds_list, contour_sequence)

        # Update the mask dictionary
        mask_dict[structure_roi.ROINumber, structure_roi.ROIName] = binary_mask

    return mask_dict


def transform_img_data(img_ds):
    img = img_ds.pixel_array
    return img


def get_img_volume(img_ds_list):
    
    img_list = [transform_img_data(img) for img in img_ds_list]
    img_volume = np.array(img_list)     # z, y, x
    return img_volume


def flip_volume_by_position(volume, img_ds_list):
    first_slice = img_ds_list[0]
    _, _, slice_direction = get_slice_directions(img_ds_list[0])

    pos = first_slice.PatientPosition
    if pos[0] == "H":
        if 1 - slice_direction[2] >= 0.5:
            volume = np.flip(volume, axis=0)
            print('flip Head First')

    elif pos[0] == "F":
        if 1 - slice_direction[2] < 0.5:
            volume = np.flip(volume, axis=0)
            print('flip Feet First')
    else :
        print("weird position")

    if pos[2] == "P":
        volume = np.flip(volume, axis=1)
        print('flip Prone')

    return volume

def get_mask_volume(rs_ds, img_ds_list):
    mask_volume = get_roi_mask_4d(rs_ds, img_ds_list)
    return mask_volume


if __name__ == "__main__":

    img_dir = "./MR"                   # image series dir path
    rs_dcm = "./RS/RTSTRUCT.dcm"       # RS file

    img_ds_list = load_sorted_image_series(img_dir)
    rs_ds = dcmread(rs_dcm)

    img_volume = get_img_volume(img_ds_list)    # shape = [z, y, x]

    mask_volume = get_roi_mask_4d(rs_ds, img_ds_list)   # shape = [cate, z, y, x]
