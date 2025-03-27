import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt


# Function to plot images
def plot_images(images, titles):
    plt.figure(figsize=(15, 5))
    for i, (img, title) in enumerate(zip(images, titles), start=1):
        plt.subplot(1, len(images), i)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    plt.show()


# The image path
image_path = './figs/farm1_265_19646119714_5d705f602f_b.jpg'

# Adjust the brightness
image = Image.open(image_path)
enhancer = ImageEnhance.Brightness(image)
brightness_factors = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8)
brightness_images = [enhancer.enhance(factor) for factor in brightness_factors]
brightness_paths = ['./edited_figs/brightness_{0}.jpg'.format(factor) for factor in brightness_factors]
for img, path in zip(brightness_images, brightness_paths):
    img.save(path)


# Adjust the contrast
image = Image.open(image_path)
contrast_enhancer = ImageEnhance.Contrast(image)
contrast_factors = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8)
contrast_images = [contrast_enhancer.enhance(factor) for factor in contrast_factors]
contrast_paths = ['./edited_figs/contrast_{0}.jpg'.format(factor) for factor in contrast_factors]
for img, path in zip(contrast_images, contrast_paths):
    img.save(path)


# # [Not Use] Adjust the shadows by applying an "unsharp mask" which can bring out details in shadows
# image = Image.open(image_path)
# shadow_factors = (20, 40, 60, 80, 100, 120, 140, 160, 180)
# shadow_images = [image.filter(ImageFilter.UnsharpMask(radius=2, percent=factor, threshold=3)) for factor in shadow_factors]
# shadow_paths = ['./edited_figs/shadow_{0}.jpg'.format(factor) for factor in shadow_factors]
# for img, path in zip(shadow_images, shadow_paths):
#     img.save(path)


# Enhance the dark parts of an image using gamma correction with OpenCV
def dark_enhancer(image, gamma):
    adjusted = np.power(image / 255.0, gamma)
    adjusted = (adjusted * 255).astype('uint8')
    return adjusted

image = cv2.imread(image_path)
shadow_factors = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
shadow_images = [dark_enhancer(image, factor) for factor in shadow_factors]
shadow_paths = ['./edited_figs/shadow_{0}.jpg'.format(factor) for factor in shadow_factors]
for img, path in zip(shadow_images, shadow_paths):
    cv2.imwrite(path, img)



# Using OpenCV, you can convert the image to HSV color space and modify the saturation channel
def saturation_adjuster(image, scale):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image[:, :, 1] = np.clip(image[:, :, 1] * scale, 0, 255)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image

image = cv2.imread(image_path)
saturation_factors = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8)
saturation_images = [saturation_adjuster(image, factor) for factor in saturation_factors]
saturation_paths = ['./edited_figs/saturation_{0}.jpg'.format(factor) for factor in saturation_factors]
for img, path in zip(saturation_images, saturation_paths):
    cv2.imwrite(path, img)


# Vibrance is similar to saturation but affects the less saturated colors more. 
# There isn't a standard algorithm, but a simple approach is to apply a non-linear 
# scale to the saturation channel.
def vibrance_adjuster(image, scale):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image[:, :, 1] = np.clip(image[:, :, 1] * scale, 0, 255)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image

image = cv2.imread(image_path)
vibrance_factors = (0.1, 0.2, 0.3, 0.8, 1.0, 1.2, 1.7, 1.8, 1.9)
vibrance_images = [vibrance_adjuster(image, factor) for factor in vibrance_factors]
vibrance_paths = ['./edited_figs/vibrance_{0}.jpg'.format(factor) for factor in vibrance_factors]
for img, path in zip(vibrance_images, vibrance_paths):
    cv2.imwrite(path, img)


# Adjust the hue by modifying the hue channel in HSV color space.
# Adjust hue (0-180 in OpenCV)
def hue_adjuster(image, scale):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image[:, :, 0] = (image[:, :, 0] + scale * 180) % 180
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image

image = cv2.imread(image_path)
hue_factors = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)
hue_images = [hue_adjuster(image, factor) for factor in hue_factors]
hue_paths = ['./edited_figs/hue_{0}.jpg'.format(factor) for factor in hue_factors]
for img, path in zip(hue_images, hue_paths):
    cv2.imwrite(path, img)


# Adjusting color temperature and tint typically involves changing 
# the balance of the blue/yellow and green/magenta channels. 
# You might want to convert the image to LAB color space for this.
def temperature_adjuster(image, scale):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image[:, :, 2] = np.clip(image[:, :, 2] + scale * 100, 0, 255)
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    return image

def tint_adjuster(image, scale):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image[:, :, 1] = np.clip(image[:, :, 1] + scale * 100, 0, 255)
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    return image

image = cv2.imread(image_path)
temperature_factors = (-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8)
temperature_images = [temperature_adjuster(image, factor) for factor in temperature_factors]
temperature_paths = ['./edited_figs/temperature_{0}.jpg'.format(factor) for factor in temperature_factors]
for img, path in zip(temperature_images, temperature_paths):
    cv2.imwrite(path, img)

image = cv2.imread(image_path)
tint_factors = (-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8)
tint_images = [tint_adjuster(image, factor) for factor in tint_factors]
tint_paths = ['./edited_figs/tint_{0}.jpg'.format(factor) for factor in tint_factors]
for img, path in zip(tint_images, tint_paths):
    cv2.imwrite(path, img)


# Color Fliters
image = cv2.imread(image_path)
color_maps = {
    'Autumn': cv2.COLORMAP_AUTUMN,
    'Winter': cv2.COLORMAP_WINTER,
    'Rainbow': cv2.COLORMAP_RAINBOW,
    'Ocean': cv2.COLORMAP_OCEAN,
    'Summer': cv2.COLORMAP_SUMMER,
    'Spring': cv2.COLORMAP_SPRING,
    'Cool': cv2.COLORMAP_COOL,
    'Pink': cv2.COLORMAP_PINK
}
for name, colormap in color_maps.items():
    filtered_image = cv2.applyColorMap(image, colormap)
    output_path = f'./edited_figs/filter_{name.lower()}.jpg'
    alpha = 0.4  # For a 50% filter effect
    reduced_filter_image = cv2.addWeighted(filtered_image, alpha, image, 1 - alpha, 0)
    cv2.imwrite(output_path, reduced_filter_image)


# Motion blur can be simulated by applying a linear convolution 
# with a specific kernel that represents the blur in one direction.
def motion_blur_adjuster(image, size):
    size = int(size * 30)
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    image = cv2.filter2D(image, -1, kernel_motion_blur)
    return image
    
image = cv2.imread(image_path)
motion_blur_factors = (0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
motion_blur_images = [motion_blur_adjuster(image, factor) for factor in motion_blur_factors]
motion_blur_paths = ['./edited_figs/motion_blur_{0}.jpg'.format(factor) for factor in motion_blur_factors]
for img, path in zip(motion_blur_images, motion_blur_paths):
    cv2.imwrite(path, img)


# JPEG Compression
image = Image.open(image_path)
JPEG_compression_factors = (20, 30, 40, 50, 60, 70, 80, 90, 100)
JPEG_compression_paths = ['./edited_figs/JPEG_compression_{0}.jpg'.format(factor) for factor in JPEG_compression_factors]
for factor, path in zip(JPEG_compression_factors, JPEG_compression_paths):
    image.save(path, 'JPEG', quality=factor)


# Pixelate the image
def pixelate_adjuster(image, pixelation_level):
    small = cv2.resize(image, None, fx=pixelation_level, fy=pixelation_level, interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, image.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    return pixelated

image = cv2.imread(image_path)
pixelate_factors = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)
pixelate_images = [pixelate_adjuster(image, factor) for factor in pixelate_factors]
pixelate_paths = ['./edited_figs/pixelate_{0}.jpg'.format(factor) for factor in pixelate_factors]
for img, path in zip(pixelate_images, pixelate_paths):
    cv2.imwrite(path, img)


# Gaussian blur
image = cv2.imread(image_path)
gaussian_blur_factors = (3, 5, 9, 11, 15, 19, 27, 41)
gaussian_blur_images = [cv2.GaussianBlur(image, (factor, factor), 0) for factor in gaussian_blur_factors]
gaussian_blur_paths = ['./edited_figs/gaussian_blur_{0}.jpg'.format(factor) for factor in gaussian_blur_factors]
for img, path in zip(gaussian_blur_images, gaussian_blur_paths):
    cv2.imwrite(path, img)


# Gaussian noise
def gaussian_noise_adjuster(image, sigma):
    mean = 0
    gaussian_noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = np.clip(image + gaussian_noise * 255, 0, 255).astype(np.uint8)
    return noisy_image

image = cv2.imread(image_path)
gaussian_noise_factors = (0.00, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45)
gaussian_noise_images = [gaussian_noise_adjuster(image, factor) for factor in gaussian_noise_factors]
gaussian_noise_paths = ['./edited_figs/gaussian_noise_{:.2f}.jpg'.format(factor) for factor in gaussian_noise_factors]
for img, path in zip(gaussian_noise_images, gaussian_noise_paths):
    cv2.imwrite(path, img)


# Impulse Noise (Salt and Pepper Noise)
def impulse_noise_adjuster(img, amount):
    output = np.copy(img)
    num_salt = np.ceil(amount * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    output[tuple(coords)] = 255
    return output

image = cv2.imread(image_path)
impulse_noise_factors = (0.00, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45)
impulse_noise_images = [impulse_noise_adjuster(image, factor) for factor in impulse_noise_factors]
impulse_noise_paths = ['./edited_figs/impulse_noise_{:.2f}.jpg'.format(factor) for factor in impulse_noise_factors]
for img, path in zip(impulse_noise_images, impulse_noise_paths):
    cv2.imwrite(path, img)


# # [Not Use] Modify Vignette
# def vignette_adjuster(img, vignette_scale):
#     vignette_scale = vignette_scale * 1
#     rows, cols = img.shape[:2]
#     kernel_x = cv2.getGaussianKernel(cols, vignette_scale * cols)
#     kernel_y = cv2.getGaussianKernel(rows, vignette_scale * rows)
#     kernel = kernel_y * kernel_x.T
#     mask = 255 * kernel / np.linalg.norm(kernel)
#     vignette = np.copy(img)
#     for i in range(3):
#         vignette[:, :, i] = vignette[:, :, i] * mask
#     return vignette

# image = cv2.imread(image_path)
# vignette_factors = (0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.30, 1.5)
# vignette_images = [vignette_adjuster(image, factor) for factor in vignette_factors]
# vignette_paths = ['./edited_figs/vignette_{:.2f}.jpg'.format(factor) for factor in vignette_factors]
# for img, path in zip(vignette_images, vignette_paths):
#     cv2.imwrite(path, img)


# Modify Vignette (halo effect)
def vignette_adjuster(img, alpha):
    blurred_image = image.filter(ImageFilter.GaussianBlur(radius=15))
    halo_image = Image.blend(image, blurred_image, alpha)
    return halo_image

image = Image.open(image_path)
vignette_factors = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
vignette_images = [vignette_adjuster(image, factor) for factor in vignette_factors]
vignette_paths = ['./edited_figs/vignette_{0}.jpg'.format(factor) for factor in vignette_factors]
for img, path in zip(vignette_images, vignette_paths):
    img.save(path)


# Fisheye Distortion
def distortion_adjuster(image, distortion=0.5, scale=0.5):
    scale = scale * 0.4 + 0.3
    height, width = image.shape[:2]
    # Maps for remapping
    map_x = np.zeros((height, width), dtype=np.float32)
    map_y = np.zeros((height, width), dtype=np.float32)
    # Intermediate calculations
    fx = width * scale
    fy = height * scale
    cx = width / 2
    cy = height / 2
    # Compute the map
    for y in range(height):
        for x in range(width):
            # Normalize coordinates to have the origin at the image center
            x_normalized = (x - cx) / fx
            y_normalized = (y - cy) / fy
            r = np.sqrt(x_normalized**2 + y_normalized**2)
            theta = np.arctan(r)
            # Fisheye model
            r_distorted = (theta + distortion * theta**3) / np.pi * 2
            # Back to unnormalized coordinates
            x_distorted = r_distorted * x_normalized
            y_distorted = r_distorted * y_normalized
            # Map back and ensure we don't go outside the image boundaries
            map_x.itemset((y, x), min(max(x_distorted * fx + cx, 0), width - 1))
            map_y.itemset((y, x), min(max(y_distorted * fy + cy, 0), height - 1))
    # Remap the original image to the fisheye
    fisheye_img = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    return fisheye_img

image = cv2.imread(image_path)
distortion_factors = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
distortion_images = [distortion_adjuster(image, scale=factor) for factor in distortion_factors]
distortion_paths = ['./edited_figs/distortion_{:.2f}.jpg'.format(factor) for factor in distortion_factors]
for img, path in zip(distortion_images, distortion_paths):
    cv2.imwrite(path, img)


# Rotate
def rotate_adjuster(img, angle):
    # angle = -angle * 360
    original_width, original_height = img.size
    rotated_img = img.rotate(angle, expand=True)
    # Calculate the new size to crop to, which would be the original image's size
    rotated_width, rotated_height = rotated_img.size
    left = (rotated_width - original_width) / 2
    top = (rotated_height - original_height) / 2
    right = (rotated_width + original_width) / 2
    bottom = (rotated_height + original_height) / 2
    # Crop the image to the original size
    cropped_img = rotated_img.crop((left, top, right, bottom))
    return cropped_img

image = Image.open(image_path)
rotate_factors = (40, 80, 120, 160, 200, 240, 280, 320, 360)
rotate_images = [rotate_adjuster(image, factor) for factor in rotate_factors]
rotate_paths = ['./edited_figs/rotate_{0}.jpg'.format(factor) for factor in rotate_factors]
for img, path in zip(rotate_images, rotate_paths):
    img.save(path)


# Filp
image = Image.open(image_path)
vignette_images = [image.transpose(Image.FLIP_TOP_BOTTOM), image.transpose(Image.FLIP_LEFT_RIGHT)]
vignette_paths = ['./edited_figs/horizonal.jpg', './edited_figs/vertical.jpg']
for img, path in zip(vignette_images, vignette_paths):
    img.save(path)