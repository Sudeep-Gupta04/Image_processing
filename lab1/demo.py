import numpy as np
import cv2
import matplotlib.pyplot as plt
from encoder import encode_image, SPATIAL_RESOLUTIONS, INTENSITY_RESOLUTIONS
from decoder import decode_image, dequantize_image

def get_user_choice(options: dict, prompt: str) -> str:
    print(f"\n{prompt}")
    print("Options:")
    for idx, value in options.items():
        if isinstance(value, int):
            if idx == '00' and value == 1:
                print(f"  {idx}: {value} bit ({2**value} levels)")
            elif idx == '00' and value == 100:
                print(f"  {idx}: {value} × {value}")
            else:
                if value < 10:
                    print(f"  {idx}: {value} bits ({2**value} levels)")
                else:
                    print(f"  {idx}: {value} × {value}")
        else:
            print(f"  {idx}: {value}")
    
    while True:
        choice = input("Enter choice: ").strip()
        if choice in options:
            return choice
        print("Invalid choice. Please try again.")

def display_results(original: np.ndarray, quantized: np.ndarray, 
                   reconstructed: np.ndarray, spatial_size: int, bits: int):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title(f'Original Cropped\n{original.shape[0]} × {original.shape[1]}')
    axes[0].axis('off')
    
    quantized_display = dequantize_image(quantized, bits)
    axes[1].imshow(quantized_display, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(f'Quantized & Sampled\n{spatial_size} × {spatial_size}, {bits} bits')
    axes[1].axis('off')
    
    axes[2].imshow(reconstructed, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title(f'Reconstructed Decoded\n{spatial_size} × {spatial_size}, {bits} bits')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('results_comparison.png', dpi=150, bbox_inches='tight')
    print("\nResults saved to 'results_comparison.png'")
    plt.show()

def main():
    print("=" * 60)
    print("Image Encoder-Decoder System for DSP Lab")
    print("=" * 60)
    
    image_path = input("\nEnter path to input image: ").strip()
    spatial_index = get_user_choice(SPATIAL_RESOLUTIONS, "\nSelect Spatial Resolution (Sampling):")
    intensity_index = get_user_choice(INTENSITY_RESOLUTIONS, "\nSelect Intensity Resolution (Quantization):")
    output_file = "encoded_image.bin"
    
    print("\n" + "=" * 60)
    print("ENCODING")
    print("=" * 60)
    
    original, quantized = encode_image(image_path, spatial_index, intensity_index, output_file)
    
    print("\n" + "=" * 60)
    print("DECODING")
    print("=" * 60)
    
    reconstructed, spatial_size, bits = decode_image(output_file)
    
    print("\n" + "=" * 60)
    print("DISPLAYING RESULTS")
    print("=" * 60)
    
    display_results(original, quantized, reconstructed, spatial_size, bits)
    
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)
    
    import os
    file_size = os.path.getsize(output_file)
    print(f"Encoded file size: {file_size} bytes")
    
    original_size = original.shape[0] * original.shape[1]
    original_bytes = original_size
    compression_ratio = original_bytes / file_size
    print(f"Compression ratio: {compression_ratio:.2f}:1")
    
    if original.shape == reconstructed.shape:
        mse = np.mean((original.astype(np.float32) - reconstructed.astype(np.float32)) ** 2)
        if mse > 0:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
            print(f"PSNR: {psnr:.2f} dB")
        else:
            print("PSNR: ∞ (perfect reconstruction)")
    else:
        print("Note: Images have different sizes, PSNR not calculated")
    
    print("\nDemo complete!")

if __name__ == "__main__":
    main()
