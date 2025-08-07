# JointWise

A comprehensive medical imaging toolkit for processing and analyzing knee MRI data, with a focus on converting FastMRI datasets to standard medical imaging formats and visualizing anatomical annotations.

> ⚠️ **Note:** JointWise is still under active development. Some features may be incomplete or have known issues. Please report any bugs or unexpected behavior via GitHub issues.

## Overview

JointWise provides tools for:
- Converting FastMRI H5 files to DICOM format
- Converting DICOM files to PNG images for visualization
- Processing and analyzing knee MRI annotations
- Visualizing bounding box annotations on medical images
- Working with medical imaging datasets for research and analysis

## Features

### 🏥 Medical Image Format Conversion
- **FastMRI to DICOM**: Convert FastMRI H5 files to standard DICOM format with proper metadata
- **DICOM to PNG**: Convert DICOM files to PNG images with windowing and normalization options

### 📊 Annotation Processing
- Load and process knee pathology annotations from CSV files
- Visualize bounding box annotations on MRI slices
- Support for multiple pathology types:
  - Cartilage defects (partial/full thickness)
  - Bone abnormalities (subchondral edema)
  - Ligament injuries (MCL sprains)

### 🔬 Research Tools
- Jupyter notebook workflows for interactive analysis
- Batch processing capabilities
- Integration with popular ML/AI libraries (PyTorch, scikit-learn, etc.)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/marccodesstuff/JointWise.git
cd JointWise
```

2. Create a virtual environment (recommended):
```bash
python -m venv jointwise-env
source jointwise-env/bin/activate  # On Linux/Mac
# or
jointwise-env\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### FastMRI to DICOM Conversion

```python
from pathlib import Path
from fastmri_to_dicom import fastmri_to_dicom

# Convert a FastMRI file to DICOM
input_file = Path("path/to/fastmri_file.h5")
output_folder = Path("output/dicom/")
fastmri_to_dicom(input_file, "reconstruction_rss", output_folder)
```

### DICOM to PNG Conversion

```python
from dicom_to_png import dicom_to_png

# Convert DICOM to PNG with windowing
dicom_to_png(
    dicom_path="path/to/file.dcm",
    output_path="output/image.png",
    apply_windowing=True,
    normalize=True
)
```

### Annotation Visualization

```python
import pandas as pd
from annotation_demo import plot_bounding_box

# Load annotations
df = pd.read_csv("knee.csv")
labels_for_file = df.loc[df['file'] == 'file1000002']

# Visualize annotations on MRI slice
plot_bounding_box(mri_image, labels_for_file)
```

## Dataset Structure

The project works with knee MRI datasets containing:
- **FastMRI H5 files**: Raw MRI reconstruction data
- **Annotation CSV**: Bounding box annotations with pathology labels
- **Output formats**: DICOM and PNG files for further analysis

### Annotation Format
The annotation CSV contains the following columns:
- `file`: MRI file identifier
- `slice`: Slice number within the volume
- `study_level`: Study-level classification
- `x, y, width, height`: Bounding box coordinates
- `label`: Pathology type (e.g., "Cartilage - Partial Thickness loss/defect")

## Project Structure

```
jointwise-code/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── fastmri_to_dicom.py          # FastMRI to DICOM conversion
├── dicom_to_png.py              # DICOM to PNG conversion
├── demo.py                      # Basic demo script
├── knee.csv                     # Knee pathology annotations
├── jointwise-notebook.ipynb     # Main analysis notebook
├── annotation_demo.ipynb        # Annotation visualization demo
├── dicom_to_png.ipynb           # DICOM conversion notebook
├── fastmri_to_png.ipynb         # FastMRI processing notebook
├── output/                      # Generated DICOM files
└── png-output/                  # Generated PNG files
```

## Dependencies

Key libraries used in this project:
- **Medical Imaging**: `pydicom`, `h5py`, `fastmri`
- **Image Processing**: `PIL`, `opencv-python`, `scikit-image`
- **Data Analysis**: `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Machine Learning**: `torch`, `torchvision`, `pytorch-lightning`, `scikit-learn`
- **Development**: `jupyter`, `ipython`

## Getting Started

1. **Quick Demo**: Run `demo.py` to check your dataset paths
2. **Conversion Workflow**: Use the Jupyter notebooks for step-by-step processing
3. **Batch Processing**: Modify the scripts for your specific dataset structure

### Example Notebooks
- `jointwise-notebook.ipynb`: Complete workflow from FastMRI to analysis
- `annotation_demo.ipynb`: Visualize pathology annotations
- `dicom_to_png.ipynb`: Convert DICOM files to images
- `fastmri_to_png.ipynb`: Direct FastMRI to PNG conversion

## Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is intended for research and educational purposes. Please ensure compliance with relevant data usage agreements when working with medical imaging datasets.

## Acknowledgments

To be announced.

---

For questions or support, please open an issue on this repository.
