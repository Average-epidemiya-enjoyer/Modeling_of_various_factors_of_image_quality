import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from Modeling_of_various_factors_of_image_quality.main import unshielded_otf

grid_size = 1024
pupil_radius = 1.0
epsilons = np.arange(0.1, 0.9, 0.1)


def create_aperture(pupil_radius, epsilon, grid_size):
    x = np.linspace(-pupil_radius, pupil_radius, grid_size)
    y = np.linspace(-pupil_radius, pupil_radius, grid_size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X ** 2 + Y ** 2)

    aperture = np.zeros_like(R)
    aperture[R <= pupil_radius] = 1
    aperture[R <= epsilon * pupil_radius] = 0
    return aperture


def calculate_otf(aperture):
    fft_aperture = np.fft.fftshift(np.fft.fft2(aperture))
    otf = np.abs(fft_aperture) ** 2
    return otf / np.max(otf)


def calculate_psf(otf):
    psf = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(otf)))
    psf_normalized = np.abs(psf) ** 2
    return psf_normalized / np.max(psf_normalized)


def simulate_image(image, psf):
    image_fft = np.fft.fft2(image)
    blurred_image_fft = image_fft * psf
    blurred_image = np.fft.ifft2(blurred_image_fft)
    return np.abs(blurred_image)


cat_image = Image.open('cat.png').convert('L')

cat_image = cat_image.resize((grid_size, grid_size))
cat_image_array = np.array(cat_image)[:, :]

simulated_images = {}
for epsilon in epsilons:
    aperture = create_aperture(pupil_radius, epsilon, grid_size)
    otf = calculate_otf(aperture)
    psf = calculate_psf(otf)
    simulated_image = simulate_image(cat_image_array, psf)
    simulated_images[epsilon] = simulated_image

fig, axes = plt.subplots(3, len(epsilons), figsize=(20, 6))
for i, epsilon in enumerate(epsilons):
    aperture = create_aperture(pupil_radius, epsilon, grid_size)
    axes[0, i].imshow(aperture, cmap='gray')
    axes[0, i].set_title(f'Апертура ε={epsilon}')
    axes[0, i].axis('off')

    psf = calculate_psf(calculate_otf(aperture))
    axes[1, i].imshow(psf, cmap='hot')
    axes[1, i].set_title(f'ФРТ ε={epsilon}')
    axes[1, i].axis('off')

    simulated_image = simulated_images[epsilon]
    axes[2, i].imshow(simulated_image.astype(np.uint8))
    axes[2, i].set_title(f'Изображение ε={epsilon}')
    axes[2, i].axis('off')

plt.tight_layout()
plt.show()


def simulate_optical_system_bw(image_array, psf):
    image_fft = np.fft.fftshift(np.fft.fft2(image_array))

    blurred_image_fft = image_fft * psf

    blurred_image = np.fft.ifft2(np.fft.ifftshift(blurred_image_fft))

    return np.real(blurred_image).clip(0, 255).astype(np.uint8)


cat_image_bw = cat_image.convert('L')
cat_image_bw_array = np.array(cat_image_bw)

simulated_images_bw = {}

for epsilon in epsilons:
    aperture = create_aperture(pupil_radius, epsilon, grid_size)
    otf = calculate_otf(aperture)
    psf = calculate_psf(otf)
    simulated_images_bw[epsilon] = simulate_optical_system_bw(cat_image_bw_array, psf)

simulated_images_bw['unshielded'] = simulate_optical_system_bw(cat_image_bw_array, calculate_psf(unshielded_otf))

fig, axes = plt.subplots(2, 5, figsize=(20, 8))

axes[0, 0].imshow(simulated_images_bw['unshielded'], cmap='gray')
axes[0, 0].set_title('Без экранирования')
axes[0, 0].axis('off')

for i, epsilon in enumerate(epsilons):
    row = i // 4
    col = i % 4 + 1
    axes[row, col].imshow(simulated_images_bw[epsilon], cmap='gray')
    axes[row, col].set_title(f'ε = {epsilon:.1f}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()
