# Affine Transformations on Digital Images

Python implementation of affine transformations (scaling, rotation, translation, and shearing) on digital images using manual mathematical equations.

## Project Files

### `affine_transform.py`
Main program that performs affine transformations on images:
- **Image Loading**: Reads input image in grayscale mode
- **User Input**: Prompts for transformation parameters (scaling, rotation, translation, shearing)
- **Manual Transformation**: Applies affine transformation using mathematical equations (no built-in functions)
- **Pixel Mapping**: Explicitly maps each pixel to new coordinates
- **Boundary Checking**: Ensures transformed coordinates are within image bounds
- **Output**: Saves and displays transformed image

**Main Function**: `apply_affine_transform(image_path, output_path)`

## Installation

```bash
pip install opencv-python numpy
```

## Usage

```bash
python affine_transform.py
```

The program will prompt you for:
1. Input image path (default: `input.jpg`)
2. Output image path (default: `output.jpg`)
3. Transformation parameters:
   - Horizontal scaling factor
   - Vertical scaling factor
   - Rotation angle (degrees)
   - Horizontal translation
   - Vertical translation
   - Horizontal shear factor
   - Vertical shear factor

## Affine Transformation Equations

The transformation is applied using the following equations:

**x' = sx × x × cos(θ) - sy × y × sin(θ) + shx × y + tx**

**y' = sx × x × sin(θ) + sy × y × cos(θ) + shy × x + ty**

Where:
- `sx`, `sy`: Scaling factors
- `θ`: Rotation angle (converted to radians)
- `tx`, `ty`: Translation values
- `shx`, `shy`: Shear factors

## Implementation Details

- **No Built-in Functions**: Transformation is implemented manually using mathematical equations
- **Forward Mapping**: Each pixel from the input image is mapped to new coordinates in the output image
- **Boundary Handling**: Only pixels within image bounds are copied
- **Nearest Neighbor**: Uses integer rounding for pixel coordinates

## Example

```python
# Example transformation parameters:
# Horizontal scaling: 1.5
# Vertical scaling: 1.5
# Rotation angle: 45 degrees
# Horizontal translation: 50
# Vertical translation: 50
# Horizontal shear: 0.2
# Vertical shear: 0.2
```

## Output

- Input image displayed in a window
- Transformed image displayed in a window
- Output image saved as specified file (default: `output.jpg`)
