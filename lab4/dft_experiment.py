
import numpy as np
import matplotlib.pyplot as plt

# ============== 1. Generate 8×8 2-D DFT Basis ==============

def generate_dft_basis():
    """
    For each (u,v) in [0,7]×[0,7], compute basis:
    B_{u,v}(x,y) = exp(-j * 2*pi * (u*x/8 + v*y/8))
    Return magnitude of each basis, arranged as 64 blocks for composite.
    """
    N = 8
    basis_blocks = []
    for v in range(N):
        row_blocks = []
        for u in range(N):
            block = np.zeros((N, N), dtype=np.complex128)
            for y in range(N):
                for x in range(N):
                    exponent = -2j * np.pi * (u * x / N + v * y / N)
                    block[y, x] = np.exp(exponent)
            magnitude = np.abs(block)
            row_blocks.append(magnitude)
        basis_blocks.append(np.hstack(row_blocks))
    composite = np.vstack(basis_blocks)
    return composite


# ============== 2. Create Binary 64×64 Image with Rectangle ==============

def create_rectangle_image():
    """
    Create 64×64 binary image (zeros). User inputs top-left (x0,y0), width, height.
    Set rectangle pixels to 1. Return image.
    """
    img = np.zeros((64, 64), dtype=np.float64)
    print("Enter rectangle parameters for 64×64 image:")
    x0 = int(input("  Top-left x (0-63): "))
    y0 = int(input("  Top-left y (0-63): "))
    w = int(input("  Width (pixels): "))
    h = int(input("  Height (pixels): "))
    x1 = min(x0 + w, 64)
    y1 = min(y0 + h, 64)
    x0 = max(0, x0)
    y0 = max(0, y0)
    img[y0:y1, x0:x1] = 1.0
    return img


# ============== 3. 2-D DFT from Scratch ==============

def dft2d_manual(f):
    """
    Compute 2-D DFT using nested loops.
    F(u,v) = sum_{x=0}^{M-1} sum_{y=0}^{N-1} f(x,y) * exp(-j*2*pi*(u*x/M + v*y/N))
    f: 2D array, f[y,x] = image at row y, column x (height N, width M).
    Returns F[u,v] same shape as f.
    """
    N, M = f.shape
    F = np.zeros((M, N), dtype=np.complex128)
    for v in range(N):
        for u in range(M):
            s = 0.0 + 0.0j
            for y in range(N):
                for x in range(M):
                    exponent = -2j * np.pi * (u * x / M + v * y / N)
                    s += f[y, x] * np.exp(exponent)
            F[u, v] = s
    return F


# ============== Main: Run all experiments ==============

def main():
    # ----- 1. Display 8×8 DFT basis composite (64×64) -----
    print("1. Generating 8×8 2-D DFT basis composite image...")
    basis_composite = generate_dft_basis()

    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 8))
    im1 = ax1.imshow(basis_composite, cmap='gray')
    ax1.set_title("8×8 2-D DFT Basis (64 basis images, each 8×8)")
    plt.colorbar(im1, ax=ax1)
    plt.tight_layout()
    plt.savefig("lab4_dft_basis.png", dpi=150)
    plt.show()

    # ----- 2. Create rectangle image and display -----
    print("\n2. Create binary 64×64 image with rectangle.")
    rect_image = create_rectangle_image()

    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
    im2 = ax2.imshow(rect_image, cmap='gray', vmin=0, vmax=1)
    ax2.set_title("Binary 64×64 Image with Rectangle")
    plt.colorbar(im2, ax=ax2)
    plt.tight_layout()
    plt.savefig("lab4_rectangle_image.png", dpi=150)
    plt.show()

    # ----- 3. Compute and plot 2-D DFT of rectangle image (from scratch) -----
    print("\n3. Computing 2-D DFT of rectangle image (manual implementation)...")
    F_rect = dft2d_manual(rect_image)
    log_magnitude = np.log(1 + np.abs(F_rect))

    fig3, ax3 = plt.subplots(1, 1, figsize=(7, 6))
    im3 = ax3.imshow(log_magnitude, cmap='gray')
    ax3.set_title("Log Magnitude Spectrum: log(1 + |F(u,v)|)")
    plt.colorbar(im3, ax=ax3)
    plt.tight_layout()
    plt.savefig("lab4_dft_spectrum.png", dpi=150)
    plt.show()

    # ----- 4. Centered image and its DFT -----
    print("\n4. Centered image f_c(x,y) = f(x,y) * (-1)^(x+y) and its DFT...")
    N, M = rect_image.shape
    centered = rect_image.copy()
    for y in range(N):
        for x in range(M):
            centered[y, x] = rect_image[y, x] * ((-1) ** (x + y))

    F_centered = dft2d_manual(centered)
    log_magnitude_centered = np.log(1 + np.abs(F_centered))

    fig4, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(centered, cmap='gray', vmin=-1, vmax=1)
    axes[0].set_title("Centered Image f_c(x,y) = f(x,y)·(-1)^{x+y}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    im4 = axes[1].imshow(log_magnitude_centered, cmap='gray')
    axes[1].set_title("Log Magnitude Spectrum of Centered Image")
    plt.colorbar(im4, ax=axes[1])
    plt.tight_layout()
    plt.savefig("lab4_centered_and_spectrum.png", dpi=150)
    plt.show()

    print("\nDone. Figures saved as lab4_dft_basis.png, lab4_rectangle_image.png,")
    print("lab4_dft_spectrum.png, lab4_centered_and_spectrum.png")


if __name__ == "__main__":
    main()
