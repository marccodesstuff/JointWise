import datetime
import os
from pathlib import Path

import h5py
import numpy as np
import pydicom

from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import generate_uid

import xmltodict

def fastmri_to_dicom(
    filename: Path,
    reconstruction_name: str,
    outfolder: Path | None,
    flip_up_down: bool = False,
    flip_left_right: bool = False,
) -> None:

    fileparts = os.path.splitext(filename.name)
    patientName = fileparts[0]

    # Decide output folder
    if outfolder is None or str(outfolder) == "":
        outfolder = Path(patientName)
    outfolder.mkdir(parents=True, exist_ok=True)

    with h5py.File(filename, 'r') as f:
        if 'ismrmrd_header' not in f.keys():
            raise Exception('ISMRMRD header not found in file')

        if reconstruction_name not in f.keys():
            raise Exception('Reconstruction name not found in file')

        # Get some header information
        head = xmltodict.parse(f['ismrmrd_header'][()])
        reconSpace = head['ismrmrdHeader']['encoding']['reconSpace']  # ['matrixSize', 'fieldOfView_mm']
        measurementInformation = head['ismrmrdHeader']['measurementInformation']  # ['measurementID', 'patientPosition', 'protocolName', 'frameOfReferenceUID']
        acquisitionSystemInformation = head['ismrmrdHeader']['acquisitionSystemInformation']  # ['systemVendor', 'systemModel', 'systemFieldStrength_T', 'relativeReceiverNoiseBandwidth' 'receiverChannels', 'coilLabel', 'institutionName']
        H1resonanceFrequency_Hz = head['ismrmrdHeader']['experimentalConditions']['H1resonanceFrequency_Hz']
        sequenceParameters = head['ismrmrdHeader']['sequenceParameters']  # ['TR', 'TE', 'TI', 'flipAngle_deg', 'sequence_type', 'echo_spacing']

        # Some calculated values
        pixelSizeX = float(reconSpace['fieldOfView_mm']['x']) / float(reconSpace['matrixSize']['x'])
        pixelSizeY = float(reconSpace['fieldOfView_mm']['y']) / float(reconSpace['matrixSize']['y'])

        # Get and prep pixel data
        img_data = f[reconstruction_name][:]
        slices = img_data.shape[0]

        if flip_left_right:
            img_data = img_data[:, :, ::-1]

        if flip_up_down:
            img_data = img_data[:, ::-1, :]

        image_max = 1024
        scale = image_max / np.percentile(img_data, 99.9)
        pixels_scaled = np.clip((scale * img_data), 0, image_max).astype('int16')
        windowWidth = 2 * (np.percentile(pixels_scaled, 99.9) - np.percentile(pixels_scaled, 0.1))
        windowCenter = windowWidth / 2

        studyInstanceUid = generate_uid('999.')
        seriesInstanceUid = generate_uid('9999.')

        for s in range(0, slices):
            slice_filename = f"{patientName}_{s:03d}.dcm"
            slice_full_path = outfolder / slice_filename
            slice_pixels = pixels_scaled[s, :, :]

            # File meta info data elements
            file_meta = FileMetaDataset()
            file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.4'
            file_meta.MediaStorageSOPInstanceUID = "1.2.3"
            file_meta.ImplementationClassUID = "1.2.3.4"
            file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'

            # Main data elements
            ds = Dataset()

            dt = datetime.datetime.now()
            ds.ContentDate = dt.strftime('%Y%m%d')
            timeStr = dt.strftime('%H%M%S.%f')  # long format with micro seconds
            ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.4'
            ds.SOPInstanceUID = generate_uid('9999.')
            ds.ContentTime = timeStr
            ds.Modality = 'MR'
            ds.ModalitiesInStudy = ['', 'PR', 'MR', '']
            ds.StudyDescription = measurementInformation['protocolName']
            ds.PatientName = patientName
            ds.PatientID = patientName
            ds.PatientBirthDate = '19700101'
            ds.PatientSex = 'M'
            ds.PatientAge = '030Y'
            ds.PatientIdentityRemoved = 'YES'
            ds.MRAcquisitionType = '2D'
            ds.SequenceName = sequenceParameters['sequence_type']
            ds.SliceThickness = reconSpace['fieldOfView_mm']['z']
            ds.RepetitionTime = sequenceParameters['TR']
            ds.EchoTime = sequenceParameters['TE']
            ds.ImagingFrequency = H1resonanceFrequency_Hz
            ds.ImagedNucleus = '1H'
            ds.EchoNumbers = "1"
            ds.MagneticFieldStrength = acquisitionSystemInformation['systemFieldStrength_T']
            ds.SpacingBetweenSlices = reconSpace['fieldOfView_mm']['z']  # 2D, assume 0 slice spacing
            ds.FlipAngle = str(sequenceParameters['flipAngle_deg'])
            ds.PatientPosition = measurementInformation['patientPosition']
            ds.StudyInstanceUID = studyInstanceUid
            ds.SeriesInstanceUID = seriesInstanceUid
            ds.StudyID = measurementInformation['measurementID']
            ds.InstanceNumber = str(s + 1)
            ds.ImagesInAcquisition = str(slices)
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = 'MONOCHROME2'
            ds.NumberOfFrames = "1"
            ds.Rows = slice_pixels.shape[0]
            ds.Columns = slice_pixels.shape[1]
            ds.PixelSpacing = [pixelSizeX, pixelSizeY]
            ds.PixelAspectRatio = [1, 1]
            ds.BitsAllocated = 16
            ds.BitsStored = 12
            ds.HighBit = 11
            ds.PixelRepresentation = 1
            ds.SmallestImagePixelValue = 0
            ds.LargestImagePixelValue = 1024
            ds.BurnedInAnnotation = 'NO'
            ds.WindowCenter = str(windowCenter)
            ds.WindowWidth = str(windowWidth)
            ds.LossyImageCompression = '00'
            ds.StudyStatusID = 'COMPLETED'
            ds.ResultsID = ''

            ds.set_pixel_data(slice_pixels, 'MONOCHROME2', 12)

            ds.file_meta = file_meta
            ds.is_implicit_VR = False
            ds.is_little_endian = True
            ds.save_as(slice_full_path, write_like_original=False)

def collect_h5_files(input_path: Path, recursive: bool = False) -> list[Path]:
    """Return a list of .h5/.hdf5 files under input_path.

    If input_path is a file, returns [input_path]. If a directory, glob it.
    """
    if input_path.is_file():
        return [input_path]
    patterns = ["**/*.h5", "**/*.hdf5"] if recursive else ["*.h5", "*.hdf5"]
    files: list[Path] = []
    for pat in patterns:
        files.extend(sorted(input_path.glob(pat)))
    return files


def main():
    # Configure these paths/flags as needed:
    INPUT_PATH = Path("/path/to/your/h5_directory_or_file")  # set this to your directory (or single .h5 file)
    OUTPUT_BASE = Path("output-folder")  # base directory where per-file subfolders will be created; can set to None
    RECONSTRUCTION_NAME = "reconstruction_rss"  # e.g., reconstruction_rss
    FLIP_UP_DOWN = True
    FLIP_LEFT_RIGHT = False
    RECURSIVE = False  # set True to search subdirectories when INPUT_PATH is a directory

    if not INPUT_PATH.exists():
        print(f"Input path does not exist: {INPUT_PATH}. Please edit demos/fastmri_to_dicom.py and set INPUT_PATH.")
        return

    files = collect_h5_files(INPUT_PATH, recursive=RECURSIVE)
    if not files:
        print(f"No H5 files found under {INPUT_PATH}")
        return

    for fpath in files:
        # Create a per-file output folder to avoid collisions
        per_file_out = (OUTPUT_BASE / fpath.stem) if OUTPUT_BASE else Path(fpath.stem)
        per_file_out.mkdir(parents=True, exist_ok=True)
        print(f"Converting {fpath} -> {per_file_out} ...")
        fastmri_to_dicom(
            filename=fpath,
            reconstruction_name=RECONSTRUCTION_NAME,
            outfolder=per_file_out,
            flip_up_down=FLIP_UP_DOWN,
            flip_left_right=FLIP_LEFT_RIGHT,
        )
        print(f"Done: {fpath}")

if __name__ == '__main__':
    main()