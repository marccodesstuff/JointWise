import pydicom
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
import os

def dicom_to_png(dicom_path, output_path=None, apply_windowing=True, normalize=True):
    """
    Convert a DICOM file to PNG format.
    
    Args:
        dicom_path (str or Path): Path to the DICOM file
        output_path (str or Path, optional): Output PNG file path. If None, uses same name as DICOM with .png extension
        apply_windowing (bool): Whether to apply DICOM windowing (window center/width)
        normalize (bool): Whether to normalize pixel values to 0-255 range
    
    Returns:
        str: Path to the saved PNG file
    """
    # Read DICOM file
    dicom_path = Path(dicom_path)
    if not dicom_path.exists():
        raise FileNotFoundError(f"DICOM file not found: {dicom_path}")
    
    ds = pydicom.dcmread(dicom_path)
    
    # Get pixel array
    pixel_array = ds.pixel_array
    
    # Handle different pixel representations
    if ds.PixelRepresentation == 1:  # Signed integers
        # Convert to signed if needed
        pixel_array = pixel_array.astype(np.int16)
    
    # Apply rescale slope and intercept if present
    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
    
    # Apply windowing if requested and window parameters are available
    if apply_windowing and hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
        window_center = float(ds.WindowCenter) if isinstance(ds.WindowCenter, (int, float, str)) else float(ds.WindowCenter[0])
        window_width = float(ds.WindowWidth) if isinstance(ds.WindowWidth, (int, float, str)) else float(ds.WindowWidth[0])
        
        # Apply windowing
        window_min = window_center - window_width / 2
        window_max = window_center + window_width / 2
        
        pixel_array = np.clip(pixel_array, window_min, window_max)
        pixel_array = (pixel_array - window_min) / (window_max - window_min) * 255
    elif normalize:
        # Normalize to 0-255 range
        pixel_min = np.min(pixel_array)
        pixel_max = np.max(pixel_array)
        if pixel_max > pixel_min:
            pixel_array = (pixel_array - pixel_min) / (pixel_max - pixel_min) * 255
        else:
            pixel_array = np.zeros_like(pixel_array)
    
    # Convert to uint8
    pixel_array = pixel_array.astype(np.uint8)
    
    # Handle multi-frame DICOM (3D data)
    if len(pixel_array.shape) == 3:
        print(f"Multi-frame DICOM detected with {pixel_array.shape[0]} frames")
        # Save each frame as a separate PNG
        output_dir = output_path.parent if output_path else dicom_path.parent
        base_name = output_path.stem if output_path else dicom_path.stem
        
        saved_files = []
        for i, frame in enumerate(pixel_array):
            frame_output = output_dir / f"{base_name}_frame_{i:03d}.png"
            
            # Create PIL Image and save
            if len(frame.shape) == 2:  # Grayscale
                img = Image.fromarray(frame, mode='L')
            else:  # RGB
                img = Image.fromarray(frame)
            
            img.save(frame_output)
            saved_files.append(str(frame_output))
            print(f"Saved frame {i}: {frame_output}")
        
        return saved_files
    
    else:
        # Single frame DICOM
        if output_path is None:
            output_path = dicom_path.with_suffix('.png')
        else:
            output_path = Path(output_path)
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create PIL Image and save
        if len(pixel_array.shape) == 2:  # Grayscale
            img = Image.fromarray(pixel_array, mode='L')
        else:  # RGB
            img = Image.fromarray(pixel_array)
        
        img.save(output_path)
        print(f"Saved: {output_path}")
        return str(output_path)

def process_directory(input_dir, output_dir=None, apply_windowing=True, normalize=True):
    """
    Process all DICOM files in a directory.
    
    Args:
        input_dir (str or Path): Directory containing DICOM files
        output_dir (str or Path, optional): Output directory for PNG files
        apply_windowing (bool): Whether to apply DICOM windowing
        normalize (bool): Whether to normalize pixel values
    """
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir / "png_output"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all DICOM files (common extensions)
    dicom_extensions = ['.dcm', '.dicom', '.ima', '']
    dicom_files = []
    
    for ext in dicom_extensions:
        if ext:
            dicom_files.extend(input_dir.glob(f"*{ext}"))
        else:
            # Check files without extension
            for file in input_dir.iterdir():
                if file.is_file() and not file.suffix:
                    try:
                        # Try to read as DICOM
                        pydicom.dcmread(file, stop_before_pixels=True)
                        dicom_files.append(file)
                    except:
                        continue
    
    if not dicom_files:
        print(f"No DICOM files found in {input_dir}")
        return
    
    print(f"Found {len(dicom_files)} DICOM files")
    
    for dicom_file in dicom_files:
        try:
            output_path = output_dir / f"{dicom_file.stem}.png"
            dicom_to_png(dicom_file, output_path, apply_windowing, normalize)
        except Exception as e:
            print(f"Error processing {dicom_file}: {e}")

def main():
    # Set your input and output directories here
    input_dir = Path("/home/bictor0301/Code/jointwise-code/output")
    output_dir = Path("/home/bictor0301/Code/jointwise-code/png-output")
    apply_windowing = True
    normalize = True

    if not input_dir.exists():
        print(f"Error: Input path does not exist: {input_dir}")
        return

    process_directory(input_dir, output_dir, apply_windowing, normalize)

if __name__ == "__main__":
    main()
