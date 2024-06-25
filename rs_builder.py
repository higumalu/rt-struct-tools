

import datetime
import cv2
import numpy as np

from pydicom.uid import generate_uid
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ImplicitVRLittleEndian

from contour_process_method import contour_process
from coordinate_transform import get_pixel_to_patient_transformation_matrix, apply_transformation_to_3d_points


def edit_required_elements(ds: FileDataset,
                           structure_set_label="RSlabel",
                           manufacturer="higumalu",
                           manufacturer_model_name="modelv1.0",
                           institution_name="higumalu"):
    
    ds.StructureSetLabel = structure_set_label
    ds.SeriesDescription = structure_set_label+"__test"
    ds.Manufacturer = manufacturer
    ds.ManufacturerModelName = manufacturer_model_name
    ds.InstitutionName = institution_name
    return ds


def create_rtstruct_dataset(series_data) -> FileDataset:
    ds = generate_base_dataset()
    add_study_and_series_information(ds, series_data)
    add_patient_information(ds, series_data)
    add_refd_frame_of_ref_sequence(ds, series_data)
    return ds


def generate_base_dataset() -> FileDataset:
    file_name = "rs_test"
    file_meta = get_file_meta()
    ds = FileDataset(file_name, {}, file_meta=file_meta, preamble=b"\0" * 128)
    add_required_elements_to_ds(ds)
    add_sequence_lists_to_ds(ds)
    return ds

def get_file_meta() -> FileMetaDataset:
    file_meta = FileMetaDataset()
    file_meta.FileMetaInformationGroupLength = 202
    file_meta.FileMetaInformationVersion = b"\x00\x01"
    file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.481.3"
    file_meta.MediaStorageSOPInstanceUID = (
        generate_uid()
    )  # TODO find out random generation is fine
    file_meta.ImplementationClassUID = "1.2.826.0.1.3680043.8.498.1"  # pydicom's (0002,0012) *Implementation Class UID*  
    return file_meta


def add_required_elements_to_ds(ds: FileDataset):
    dt = datetime.datetime.now()
    # Append data elements required by the DICOM standarad
    ds.SpecificCharacterSet = "ISO_IR 100"
    ds.InstanceCreationDate = dt.strftime("%Y%m%d")
    ds.InstanceCreationTime = dt.strftime("%H%M%S.%f")
    ds.StructureSetLabel = "RSlabel"
    ds.StructureSetDate = dt.strftime("%Y%m%d")
    ds.StructureSetTime = dt.strftime("%H%M%S.%f")
    ds.Modality = "RTSTRUCT"
    ds.Manufacturer = "higumalu"
    ds.ManufacturerModelName = "modelv1"
    ds.InstitutionName = "higumalu"
    # Set the transfer syntax
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    # Set values already defined in the file meta
    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID

    ds.ApprovalStatus = "UNAPPROVED"


def add_sequence_lists_to_ds(ds: FileDataset):
    ds.StructureSetROISequence = Sequence()
    ds.ROIContourSequence = Sequence()
    ds.RTROIObservationsSequence = Sequence()


def add_study_and_series_information(ds: FileDataset, series_data):
    reference_ds = series_data[0]  # All elements in series should have the same data
    ds.StudyDate = reference_ds.StudyDate
    ds.SeriesDate = getattr(reference_ds, "SeriesDate", "")
    ds.StudyTime = reference_ds.StudyTime
    ds.SeriesTime = getattr(reference_ds, "SeriesTime", "")
    ds.StudyDescription = getattr(reference_ds, "StudyDescription", "")
    ds.SeriesDescription = getattr(reference_ds, "SeriesDescription", "")
    ds.StudyInstanceUID = reference_ds.StudyInstanceUID
    ds.SeriesInstanceUID = generate_uid()  # TODO: find out if random generation is ok
    ds.StudyID = reference_ds.StudyID
    ds.SeriesNumber = "1"  # TODO: find out if we can just use 1 (Should be fine since its a new series)


def add_patient_information(ds: FileDataset, series_data):
    reference_ds = series_data[0]  # All elements in series should have the same data
    ds.PatientName = getattr(reference_ds, "PatientName", "")
    ds.PatientID = getattr(reference_ds, "PatientID", "")
    ds.PatientBirthDate = getattr(reference_ds, "PatientBirthDate", "")
    ds.PatientSex = getattr(reference_ds, "PatientSex", "")
    ds.PatientAge = getattr(reference_ds, "PatientAge", "")
    ds.PatientSize = getattr(reference_ds, "PatientSize", "")
    ds.PatientWeight = getattr(reference_ds, "PatientWeight", "")


def add_refd_frame_of_ref_sequence(ds: FileDataset, series_data):
    refd_frame_of_ref = Dataset()
    refd_frame_of_ref.FrameOfReferenceUID = getattr(series_data[0], 'FrameOfReferenceUID', generate_uid())
    refd_frame_of_ref.RTReferencedStudySequence = create_frame_of_ref_study_sequence(series_data)

    ds.ReferencedFrameOfReferenceSequence = Sequence()
    ds.ReferencedFrameOfReferenceSequence.append(refd_frame_of_ref)



# General Study --> Referenced Study Sequence --> [ReferencedSOPClassUID, ReferencedSOPInstanceUID]
################################## Referenced Study Sequence Attribute ###################################
def create_frame_of_ref_study_sequence(series_data) -> Sequence:
    reference_ds = series_data[0]
    rt_refd_series = Dataset()
    rt_refd_series.SeriesInstanceUID = reference_ds.SeriesInstanceUID
    rt_refd_series.ContourImageSequence = create_contour_image_sequence(series_data)

    rt_refd_series_sequence = Sequence()
    rt_refd_series_sequence.append(rt_refd_series)

    rt_refd_study = Dataset()
    rt_refd_study.ReferencedSOPClassUID = "1.2.840.10008.3.1.2.3.1"
    rt_refd_study.ReferencedSOPInstanceUID = reference_ds.StudyInstanceUID
    rt_refd_study.RTReferencedSeriesSequence = rt_refd_series_sequence

    rt_refd_study_sequence = Sequence()
    rt_refd_study_sequence.append(rt_refd_study)

    return rt_refd_study_sequence


def create_contour_image_sequence(series_data) -> Sequence:
    contour_image_sequence = Sequence()
    for series in series_data:
        contour_image = Dataset()
        contour_image.ReferencedSOPClassUID = series.SOPClassUID
        contour_image.ReferencedSOPInstanceUID = series.SOPInstanceUID
        contour_image_sequence.append(contour_image)

    return contour_image_sequence




# StructureSetROISequence  --> [ROINumber, ReferencedFrameOfReferenceUID, ROIName, ROIGenerationAlgorithm=AUTOMATIC]
################################ StructureSetROISequence ####################################################################
def create_structure_set_roi(roi_number, refd_frame_of_ref_uid, roi_name, roi_description) -> Dataset:
    # Structure Set ROI Sequence: Structure Set ROI 1
    structure_set_roi = Dataset()
    structure_set_roi.ROINumber = roi_number
    structure_set_roi.ReferencedFrameOfReferenceUID = refd_frame_of_ref_uid
    structure_set_roi.ROIName = roi_name
    structure_set_roi.ROIDescription = roi_description
    structure_set_roi.ROIGenerationAlgorithm = "AUTOMATIC"
    return structure_set_roi



# create ROIContourSequence -> ContourSequence -> [ContourImageSequence / ContourData / ContourNumber / ...]
################################ ROIContourSequence ####################################################################
def create_roi_contour_sequence(roi_number, roi_color, mask_volume, image_series) -> Dataset:     # == create_roi_contour() in rt-utils
    roi_contour_sequence = Dataset()
    roi_contour_sequence.ReferencedROINumber = roi_number    # also in create_structure_set_roi
    roi_contour_sequence.ROIDisplayColor = roi_color
    roi_contour_sequence.ContourSequence = create_contour_sequence(mask_volume, image_series)
    return roi_contour_sequence

# 3Dmask -[for loop]-> 2Dmask -[cv2.findContours]-> poly_list -[for loop]-> 2Dcontour
def create_contour_sequence(mask_volume, image_series) -> Sequence:
    contour_sequence = Sequence()

    contour_data_seq = volume_to_contour_list(mask_volume, image_series)

    for index, [contour_data, image_slice] in enumerate(contour_data_seq):
        contour = create_contour_sequence_block(contour_data, image_slice)
        contour_sequence.append(contour)

    return contour_sequence


# ContourSequence -> [ContourImageSequence, ContourGeometricType, NumberOfContourPoints, ContourNumber, ContourData]    
# -> ContourImageSequence -> [ReferencedSOPClassUID, ReferencedSOPInstanceUID]
def create_contour_sequence_block(contour_data, image_slice) -> Dataset:
    contour_image = Dataset()
    contour_image.ReferencedSOPClassUID = image_slice.SOPClassUID
    contour_image.ReferencedSOPInstanceUID = image_slice.SOPInstanceUID
    contour_image_sequence = Sequence()
    contour_image_sequence.append(contour_image)
    
    contour = Dataset()
    contour.ContourImageSequence = contour_image_sequence
    contour.ContourGeometricType = "CLOSED_PLANAR"
    contour.NumberOfContourPoints = (len(contour_data) / 3)
    contour_data = [round(coord, 3) for coord in contour_data]
    contour.ContourData = contour_data

    return contour



# 3Dmask
# -[for loop with ds_list]-> 2Dmask with series slice 
# -[cv2.findContours]-> cv2_contours 
# -[for loop]-> contour(polygon) 
# -[smooth_ctr (maybe)]-> contour 
# -[coordination transform]-> dicom fromat contour (ContourData)
def volume_to_contour_list(mask_volume, image_series):
    formatted_contours = []
    for index, series_slice in enumerate(image_series):
        mask_slice = mask_volume[index, :, :]           # mask_volume.shape = (z, y, x)     ex: (144, 512, 512)

        if np.sum(mask_slice) == 0:
            continue

        cv2_contours, hierarchy = cv2.findContours(
                                                mask_slice.astype(np.uint8),
                                                cv2.RETR_CCOMP,
                                                cv2.CHAIN_APPROX_NONE
                                                )
        if cv2_contours is None or hierarchy is None: continue


        for hier_index, contour in enumerate(cv2_contours):
            poly_hierarchy = hierarchy[0][hier_index][3]
            
            x_points, y_points = contour.T[:, 0]
            

            # interface for contour smoothing and other process
            x_points, y_points = contour_process(x_points, y_points, poly_hierarchy,
                                                 external_noise_size=10,
                                                 internal_noise_size=10,
                                                 low_pass_ratio=10)
            
            # check if contour is not empty
            if len(x_points) == 0 or len(y_points) == 0: continue

            contour = np.stack([x_points, y_points]).T
            
            contour = np.concatenate((np.array(contour), np.full((len(contour), 1), index)), axis=1) #  add index into contour-z-axis
            
            transformation_matrix = get_pixel_to_patient_transformation_matrix(image_series)
            transformed_contour = apply_transformation_to_3d_points(contour, transformation_matrix)

            dicom_formatted_contour = np.ravel(transformed_contour).tolist()
            formatted_contours.append(dicom_formatted_contour)
            formatted_contours.append([dicom_formatted_contour, series_slice])

    return formatted_contours


################################ RTROIObservationsSequence #######################################################
def create_rtroi_observation(roi_number) -> Dataset:
    rtroi_observation = Dataset()
    rtroi_observation.ObservationNumber = roi_number
    rtroi_observation.ReferencedROINumber = roi_number

    rtroi_observation.ROIObservationDescription = "Type:Soft,Range:*/*,Fill:0,Opacity:0.0,Thickness:1,LineThickness:2,read-only:false"
    rtroi_observation.private_creators = "higumalu"
    rtroi_observation.RTROIInterpretedType = ""
    rtroi_observation.ROIInterpreter = ""
    return rtroi_observation




def add_mask3d_into_rsds(
            rs_ds,
            mask_volume,
            image_series,
            roi_color,
            roi_number,
            roi_name,
            roi_description
            ) -> Dataset:
    
    refd_frame_of_ref_uid = image_series[0].FrameOfReferenceUID
    rs_ds.StructureSetROISequence.append(create_structure_set_roi(roi_number, refd_frame_of_ref_uid, roi_name, roi_description))
    rs_ds.ROIContourSequence.append(create_roi_contour_sequence(roi_number, roi_color, mask_volume, image_series))
    rs_ds.RTROIObservationsSequence.append(create_rtroi_observation(roi_number))
    
    return rs_ds
