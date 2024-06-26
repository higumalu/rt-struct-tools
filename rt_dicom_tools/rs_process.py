from typing import List, Any
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence


from .rs_builder import create_rtstruct_dataset, edit_required_elements
from .rs_builder import create_structure_set_roi, merge_roi_contour_sequence, create_rtroi_observation


def merge_rs(reference_ds_series: Any, old_rs_list: List[FileMetaDataset], new_rs_dict: dict):
    reference_ds_series = reference_ds_series.copy()
    if isinstance(reference_ds_series, list):
        mod = getattr(reference_ds_series[0], 'Modality', None)
        if mod == 'RTSTRUCT':
            ref_rs_ds = reference_ds_series[0]
        if mod == 'CT':
            ref_rs_ds = create_rtstruct_dataset(reference_ds_series)
    
    if isinstance(reference_ds_series, FileDataset):
        mod = getattr(reference_ds_series, 'Modality', None)
        if mod == 'RTSTRUCT':
            ref_rs_ds = reference_ds_series
        if mod == 'CT':
            raise Exception("ref_ds CT must be a series list")
    # print(ref_rs_ds)
    
    ref_frame_of_ref_uid = getattr(ref_rs_ds, 'FrameOfReferenceUID', None)
    if ref_frame_of_ref_uid is None:
        ref_frame_of_ref_uid = getattr(ref_rs_ds.StructureSetROISequence[0], 'ReferencedFrameOfReferenceUID', None)
        if ref_frame_of_ref_uid is None:
            raise Exception("ref_ds must have FrameOfReferenceUID")

    old_rs_dict_list = [extract_rs_ds(rs_ds) for rs_ds in old_rs_list]
    
    ref_rs_ds.StructureSetROISequence = Sequence()
    ref_rs_ds.ROIContourSequence = Sequence()
    ref_rs_ds.RTROIObservationsSequence = Sequence()

    roi_num = 0
    for new_roi_name, value in new_rs_dict.items():
        rs_index = value["rs_list_index"]
        old_roi_name = value["ROIName"]
        roi_color = value.get("ROIDisplayColor", None)
        roi_desc = value.get("ROIDescription", None)
        roi_algo = value.get("ROIGenerationAlgorithm", None)
        roi_algo_desc = value.get("ROIGenerationAlgorithmDescription", None)

        roi_num = roi_num + 1

        roi_color = old_rs_dict_list[rs_index][old_roi_name]["ROIDisplayColor"] if roi_color is None else roi_color
        roi_desc = old_rs_dict_list[rs_index][old_roi_name]["ROIDescription"] if roi_desc is None else roi_desc
        roi_algo = old_rs_dict_list[rs_index][old_roi_name]["ROIGenerationAlgorithm"] if roi_algo is None else roi_algo
        roi_algo_desc = old_rs_dict_list[rs_index][old_roi_name]["ROIGenerationAlgorithmDescription"] if roi_algo_desc is None else roi_algo_desc
        roi_ctr_seq = old_rs_dict_list[rs_index][old_roi_name]["ContourSequence"]

        ref_rs_ds.StructureSetROISequence.append(create_structure_set_roi(roi_num, ref_frame_of_ref_uid, new_roi_name, roi_desc, roi_algo, roi_algo_desc))
        ref_rs_ds.ROIContourSequence.append(merge_roi_contour_sequence(roi_num, roi_color, roi_ctr_seq))
        ref_rs_ds.RTROIObservationsSequence.append(create_rtroi_observation(roi_num))

    return ref_rs_ds


def extract_rs_ds(rs_ds):
    rs_info_dict = {}

    for ssroi in rs_ds.StructureSetROISequence:
        roi_ref_uid = getattr(ssroi, "ReferencedFrameOfReferenceUID", None)
        roi_num = getattr(ssroi, "ROINumber", None)
        roi_name = getattr(ssroi, "ROIName", None)
        roi_desc = getattr(ssroi, "ROIDescription", None)
        roi_algo = getattr(ssroi, "ROIGenerationAlgorithm", None)
        roi_algo_desc = getattr(ssroi, "ROIGenerationAlgorithmDescription", None)

        for roi_ctr in rs_ds.ROIContourSequence:
            if roi_ctr.ReferencedROINumber != roi_num:
                continue
            roi_color = getattr(roi_ctr, "ROIDisplayColor", None)
            roi_ctr_seq = getattr(roi_ctr, "ContourSequence", None)
            if roi_ctr_seq is None: continue
            
            rs_info_dict[roi_name] = {}
            rs_info_dict[roi_name]["ReferencedFrameOfReferenceUID"] = roi_ref_uid
            rs_info_dict[roi_name]["ROINumber"] = roi_num
            rs_info_dict[roi_name]["ROIName"] = roi_name
            rs_info_dict[roi_name]["ROIDisplayColor"] = roi_color
            rs_info_dict[roi_name]["ROIDescription"] = roi_desc
            rs_info_dict[roi_name]["ROIGenerationAlgorithm"] = roi_algo
            rs_info_dict[roi_name]["ROIGenerationAlgorithmDescription"] = roi_algo_desc
            rs_info_dict[roi_name]["ContourSequence"] = roi_ctr_seq

    return rs_info_dict


def get_rs_organ_info(rs_ds):

    rs_dict = extract_rs_ds(rs_ds)
    rs_info_dict = {}
    for roi_name, value in rs_dict.items():
        rs_info_dict[roi_name] = {}
        rs_info_dict[roi_name]["ReferencedFrameOfReferenceUID"] = str(value["ReferencedFrameOfReferenceUID"])
        rs_info_dict[roi_name]["ROIName"] = str(value["ROIName"])
        rs_info_dict[roi_name]["ROIDisplayColor"] = [int(i) for i in value["ROIDisplayColor"]]
        rs_info_dict[roi_name]["ROIDescription"] = str(value["ROIDescription"])
        rs_info_dict[roi_name]["ROIGenerationAlgorithm"] = str(value["ROIGenerationAlgorithm"])
        rs_info_dict[roi_name]["ROIGenerationAlgorithmDescription"] = str(value["ROIGenerationAlgorithmDescription"])
        
    return rs_info_dict


def pop_rs_organ(rs_ds: FileDataset, roi_name_list: list[str]):
    organ_name_list = [ssroi.ROIName for ssroi in rs_ds.StructureSetROISequence]

    for roi_name in roi_name_list:  # check roi_name_list is leagal
        if roi_name not in organ_name_list:
            raise ValueError(f"{roi_name} is not in RS dataset")
        
        for index, ssroi in enumerate(rs_ds.StructureSetROISequence):
            if ssroi.ROIName == roi_name:
                roi_number = ssroi.ROINumber             # get roi number
                ssroi_pop_index = index

        for index, roi_ctr in enumerate(rs_ds.ROIContourSequence):
            if roi_ctr.ReferencedROINumber == roi_number:
                roi_ctr_pop_index = index

        for index, rtroi in enumerate(rs_ds.RTROIObservationsSequence):
            if rtroi.ReferencedROINumber == roi_number:
                rtroi_pop_index = index

        rs_ds.StructureSetROISequence.pop(ssroi_pop_index)
        rs_ds.ROIContourSequence.pop(roi_ctr_pop_index)
        rs_ds.RTROIObservationsSequence.pop(rtroi_pop_index)

    return rs_ds
    
