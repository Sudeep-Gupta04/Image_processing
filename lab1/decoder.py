import numpy as np
import cv2
from typing import Tuple

SPATIAL_RESOLUTIONS = {
    '00': 100,
    '01': 200,
    '10': 400,
    '11': 800
}

INTENSITY_RESOLUTIONS = {
    '00': 1,
    '01': 2,
    '10': 4,
    '11': 8
}

def unpack_bits(data: bytes, bits_per_pixel: int, num_pixels: int, header_size: int = 4) -> Tuple[np.ndarray, str]:
    bit_string = ''
    for byte_val in data:
        bit_string += format(byte_val, '08b')
    
    header_bits = bit_string[:header_size]
    
    pixels = []
    pixel_start = header_size
    for i in range(num_pixels):
        start_idx = pixel_start + i * bits_per_pixel
        end_idx = start_idx + bits_per_pixel
        if end_idx > len(bit_string):
            break
        pixel_bits = bit_string[start_idx:end_idx]
        pixel_val = int(pixel_bits, 2)
        pixels.append(pixel_val)
    
    return np.array(pixels, dtype=np.uint8), header_bits

def dequantize_image(quantized: np.ndarray, bits: int) -> np.ndarray:
    levels = 2 ** bits
    step = 256.0 / levels
    dequantized = (quantized.astype(np.float32) * step + step / 2.0).astype(np.uint8)
    dequantized = np.clip(dequantized, 0, 255)
    return dequantized

def decode_image(file_path: str) -> Tuple[np.ndarray, int, int]:
    with open(file_path, 'rb') as f:
        file_data = f.read()
    
    if len(file_data) < 1:
        raise ValueError("File is empty")
    
    first_byte = file_data[0]
    header_bits_str = format(first_byte, '08b')[:4]
    
    spatial_index = header_bits_str[:2]
    intensity_index = header_bits_str[2:4]
    
    if spatial_index not in SPATIAL_RESOLUTIONS:
        raise ValueError(f"Invalid spatial_index in header: {spatial_index}")
    if intensity_index not in INTENSITY_RESOLUTIONS:
        raise ValueError(f"Invalid intensity_index in header: {intensity_index}")
    
    spatial_size = SPATIAL_RESOLUTIONS[spatial_index]
    bits = INTENSITY_RESOLUTIONS[intensity_index]
    num_pixels = spatial_size * spatial_size
    
    quantized_flat, extracted_header = unpack_bits(file_data, bits, num_pixels, header_size=4)
    
    if extracted_header != header_bits_str:
        raise ValueError(f"Header mismatch: expected {header_bits_str}, got {extracted_header}")
    
    quantized = quantized_flat[:num_pixels].reshape(spatial_size, spatial_size)
    reconstructed = dequantize_image(quantized, bits)
    
    print(f"Decoded image from {file_path}")
    print(f"  Spatial resolution: {spatial_size}x{spatial_size} (index: {spatial_index})")
    print(f"  Intensity resolution: {bits} bits ({2**bits} levels) (index: {intensity_index})")
    
    return reconstructed, spatial_size, bits

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python decoder.py <encoded_file> <output_image_path>")
        sys.exit(1)
    
    encoded_file = sys.argv[1]
    output_path = sys.argv[2]
    
    reconstructed, size, bits = decode_image(encoded_file)
    cv2.imwrite(output_path, reconstructed)
    print(f"Reconstructed image saved to {output_path}")
