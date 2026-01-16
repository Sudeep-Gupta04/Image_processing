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

def preprocess_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    size = min(h, w)
    start_h = (h - size) // 2
    start_w = (w - size) // 2
    cropped = gray[start_h:start_h + size, start_w:start_w + size]
    
    return cropped

def resize_image(image: np.ndarray, target_size: int) -> np.ndarray:
    return cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

def quantize_image(image: np.ndarray, bits: int) -> np.ndarray:
    levels = 2 ** bits
    step = 256.0 / levels
    quantized = np.floor(image.astype(np.float32) / step).astype(np.uint8)
    quantized = np.clip(quantized, 0, levels - 1)
    return quantized

def pack_bits(data: np.ndarray, bits_per_pixel: int, header_bits: str) -> bytes:
    pixels = data.flatten()
    bit_string = header_bits
    
    for pixel in pixels:
        bit_string += format(int(pixel), f'0{bits_per_pixel}b')
    
    padding = (8 - (len(bit_string) % 8)) % 8
    bit_string += '0' * padding
    
    byte_array = bytearray()
    for i in range(0, len(bit_string), 8):
        byte_val = int(bit_string[i:i+8], 2)
        byte_array.append(byte_val)
    
    return bytes(byte_array)

def encode_image(image_path: str, spatial_index: str, intensity_index: str, output_path: str) -> Tuple[np.ndarray, np.ndarray]:
    if spatial_index not in SPATIAL_RESOLUTIONS:
        raise ValueError(f"Invalid spatial_index: {spatial_index}")
    if intensity_index not in INTENSITY_RESOLUTIONS:
        raise ValueError(f"Invalid intensity_index: {intensity_index}")
    
    target_size = SPATIAL_RESOLUTIONS[spatial_index]
    bits = INTENSITY_RESOLUTIONS[intensity_index]
    
    cropped = preprocess_image(image_path)
    sampled = resize_image(cropped, target_size)
    quantized = quantize_image(sampled, bits)
    
    header_bits = spatial_index + intensity_index
    packed_data = pack_bits(quantized, bits, header_bits)
    
    with open(output_path, 'wb') as f:
        f.write(packed_data)
    
    print(f"Encoded image saved to {output_path}")
    print(f"  Spatial resolution: {target_size}x{target_size} (index: {spatial_index})")
    print(f"  Intensity resolution: {bits} bits ({2**bits} levels) (index: {intensity_index})")
    print(f"  File size: {len(packed_data)} bytes")
    
    return cropped, quantized

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 5:
        print("Usage: python encoder.py <image_path> <spatial_index> <intensity_index> <output_path>")
        print("  spatial_index: '00' (100x100), '01' (200x200), '10' (400x400), '11' (800x800)")
        print("  intensity_index: '00' (1 bit), '01' (2 bits), '10' (4 bits), '11' (8 bits)")
        sys.exit(1)
    
    image_path = sys.argv[1]
    spatial_index = sys.argv[2]
    intensity_index = sys.argv[3]
    output_path = sys.argv[4]
    
    encode_image(image_path, spatial_index, intensity_index, output_path)
