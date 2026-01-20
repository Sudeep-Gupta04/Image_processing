import cv2
import numpy as np
import math

def apply_affine_transform(image_path, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    height, width = image.shape
    
    print("Enter transformation parameters:")
    sx = float(input("Enter horizontal scaling factor: "))
    sy = float(input("Enter vertical scaling factor: "))
    angle = float(input("Enter rotation angle (degrees): "))
    tx = float(input("Enter horizontal translation: "))
    ty = float(input("Enter vertical translation: "))
    shx = float(input("Enter horizontal shear factor: "))
    shy = float(input("Enter vertical shear factor: "))
    
    rad = math.radians(angle)
    cos_theta = math.cos(rad)
    sin_theta = math.sin(rad)
    
    output = np.zeros((height, width), dtype=np.uint8)
    
    for y_out in range(height):
        for x_out in range(width):
            x_prime = x_out - tx
            y_prime = y_out - ty
            
            a11 = sx * cos_theta
            a12 = -sy * sin_theta + shx
            a21 = sx * sin_theta + shy
            a22 = sy * cos_theta
            
            det = a11 * a22 - a12 * a21
            
            if abs(det) < 1e-10:
                continue
            
            x_in = (a22 * x_prime - a12 * y_prime) / det
            y_in = (-a21 * x_prime + a11 * y_prime) / det
            
            x_in_int = int(round(x_in))
            y_in_int = int(round(y_in))
            
            if 0 <= x_in_int < width and 0 <= y_in_int < height:
                output[y_out, x_out] = image[y_in_int, x_in_int]
    
    cv2.imwrite(output_path, output)
    
    cv2.imshow("Input Image", image)
    cv2.imshow("Affine Transformed Image", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(f"Transformed image saved to {output_path}")

if __name__ == "__main__":
    input_image = input("Enter input image path (or press Enter for 'input.jpg'): ").strip().strip('"').strip("'")
    if not input_image:
        input_image = "input.jpg"
    
    output_image = input("Enter output image path (or press Enter for 'output.jpg'): ").strip().strip('"').strip("'")
    if not output_image:
        output_image = "output.jpg"
    
    apply_affine_transform(input_image, output_image)
