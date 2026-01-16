# Image Encoder-Decoder System

Python implementation demonstrating spatial resolution (sampling) and intensity resolution (quantization) for digital image processing.

## Project Files

### `encoder.py`
The encoder module that processes and compresses images:
- **Preprocessing**: Loads image, converts to grayscale, crops center square
- **Spatial Sampling**: Resizes image to selected resolution (100×100, 200×200, 400×400, or 800×800) using nearest-neighbor interpolation
- **Intensity Quantization**: Quantizes pixel values to selected bit depth (1, 2, 4, or 8 bits)
- **Bit Packing**: Packs header (4 bits) and quantized pixel data into binary format
- **File Output**: Writes encoded data to `.bin` file

**Main Function**: `encode_image(image_path, spatial_index, intensity_index, output_path)`

### `decoder.py`
The decoder module that reconstructs images from encoded binary files:
- **File Reading**: Reads the encoded `.bin` file
- **Header Extraction**: Extracts 4-bit header to determine spatial and intensity resolution
- **Bit Unpacking**: Unpacks pixel data from bit-packed format
- **De-quantization**: Converts quantized values back to 0-255 range for display
- **Image Reconstruction**: Reconstructs and returns the decoded image

**Main Function**: `decode_image(file_path)`

### `demo.py`
Interactive demonstration script that runs the complete encoder-decoder pipeline:
- Prompts user for input image path
- Interactive menu for selecting spatial and intensity resolutions
- Calls encoder to compress the image
- Calls decoder to reconstruct the image
- Displays side-by-side comparison of original, quantized, and reconstructed images
- Calculates and displays statistics (file size, compression ratio, PSNR)
- Saves comparison image as `results_comparison.png`

**Usage**: Run `python demo.py` for interactive demo

### `requirements.txt`
Lists all Python package dependencies:
- `numpy`: Numerical operations and array handling
- `opencv-python`: Image loading, processing, and saving
- `matplotlib`: Image visualization and plotting

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Interactive Demo (Recommended)

```bash
python demo.py
```

This will guide you through:
1. Selecting an input image
2. Choosing spatial resolution (sampling)
3. Choosing intensity resolution (quantization)
4. Viewing results and statistics

### Command Line Usage

**Encode an image:**
```bash
python encoder.py <image_path> <spatial_index> <intensity_index> <output_file>
```

Example:
```bash
python encoder.py image.jpg 01 01 encoded.bin
```

**Decode an image:**
```bash
python decoder.py <encoded_file> <output_image>
```

Example:
```bash
python decoder.py encoded.bin reconstructed.png
```

## Parameters

### Spatial Resolution (Sampling)
- `00`: 100 × 100 pixels
- `01`: 200 × 200 pixels
- `10`: 400 × 400 pixels
- `11`: 800 × 800 pixels

### Intensity Resolution (Quantization)
- `00`: 1 bit (2 levels)
- `01`: 2 bits (4 levels)
- `10`: 4 bits (16 levels)
- `11`: 8 bits (256 levels)

## File Format

The encoded binary file format:
- **First 4 bits**: Header
  - Bits 0-1: Spatial resolution index (00, 01, 10, 11)
  - Bits 2-3: Intensity resolution index (00, 01, 10, 11)
- **Remaining bits**: Pixel data packed tightly using exactly `b` bits per pixel
  - No padding except to complete the final byte

## Quantization Rule

For `b` bits:
- `levels = 2^b`
- `step = 256 / levels`
- `quantized_pixel = floor(pixel / step)`

## Workflow

1. **Encoding**: Image → Preprocess → Sample → Quantize → Pack → Binary File
2. **Decoding**: Binary File → Unpack → Extract Header → De-quantize → Reconstruct → Image
