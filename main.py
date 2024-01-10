import numpy as np
import matplotlib.pyplot as plt

pupil_radius = 1.0
grid_size = 1024
wavelength = 550e-9
k = 2 * np.pi / wavelength

# Создаем сетку для моделирования апертуры
x = np.linspace(-pupil_radius, pupil_radius, grid_size)
y = np.linspace(-pupil_radius, pupil_radius, grid_size)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X ** 2 + Y ** 2)


def create_aperture(pupil_radius, screening_coefficient, R):
    aperture = np.zeros_like(R)
    aperture[R <= pupil_radius] = 1
    aperture[R <= pupil_radius * screening_coefficient] = 0
    return aperture


def calculate_otf(aperture):
    fft_aperture = np.fft.fftshift(np.fft.fft2(aperture))
    otf = np.abs(fft_aperture) ** 2
    otf /= otf.max()
    return otf


def calculate_psf(otf):
    psf = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(otf)))
    psf = np.abs(psf) ** 2
    psf /= psf.max()
    return psf


def calculate_mtf(otf):
    mtf = np.abs(otf)
    mtf /= mtf.max()
    return mtf


screening_coefficients = np.linspace(0.1, 0.8, 8)
results = {}

for epsilon in screening_coefficients:
    aperture = create_aperture(pupil_radius, epsilon, R)
    otf = calculate_otf(aperture)
    psf = calculate_psf(otf)
    mtf = calculate_mtf(otf)

    results[epsilon] = {
        'aperture': aperture,
        'otf': otf,
        'psf': psf,
        'mtf': mtf
    }

unshielded_aperture = create_aperture(pupil_radius, 0, R)
unshielded_otf = calculate_otf(unshielded_aperture)
unshielded_psf = calculate_psf(unshielded_otf)
unshielded_mtf = calculate_mtf(unshielded_otf)

results['unshielded'] = {
    'aperture': unshielded_aperture,
    'otf': unshielded_otf,
    'psf': unshielded_psf,
    'mtf': unshielded_mtf
}

for epsilon in screening_coefficients:
    example = results[epsilon]
    plt.figure(figsize=(8, 8))
    plt.imshow(example['aperture'], cmap='gray')
    plt.colorbar()
    plt.title(f'Апертура с экранированием (ε = {epsilon})')
    plt.show()

    # Отображение ФРТ
    plt.figure(figsize=(8, 8))
    plt.imshow(example['psf'], cmap='viridis')
    plt.colorbar()
    plt.title(f'ФРТ с экранированием (ε = {epsilon})')
    plt.show()

    # Отображение ФПМ
    plt.figure(figsize=(8, 8))
    plt.imshow(example['mtf'], cmap='viridis')
    plt.colorbar()
    plt.title(f'ФПМ с экранированием (ε = {epsilon})')
    plt.show()
