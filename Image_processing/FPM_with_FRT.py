import numpy as np
import matplotlib.pyplot as plt
from Modeling_of_various_factors_of_image_quality.main import calculate_psf

pupil_radius = 1.0
wavelength = 550e-9
grid_size = 1024
dx = (2 * pupil_radius) / grid_size
epsilons = np.arange(0.1, 0.9, 0.1)


def create_aperture(radius, epsilon, grid_size):
    y, x = np.ogrid[-radius:radius:grid_size * 1j, -radius:radius:grid_size * 1j]
    mask = x ** 2 + y ** 2 <= radius ** 2
    aperture = np.zeros((grid_size, grid_size))
    aperture[mask] = 1
    shield_radius = epsilon * radius
    inner_mask = x ** 2 + y ** 2 <= shield_radius ** 2
    aperture[inner_mask] = 0
    return aperture


def calculate_otf(aperture):
    fft_aperture = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(aperture)))
    otf = np.abs(fft_aperture) ** 2
    return otf / otf.max()


def calculate_mtf(otf):
    center = otf.shape[0] // 2
    mtf = otf[center, :]
    return mtf / mtf.max()


plt.figure(figsize=(15, 8))

for epsilon in epsilons:
    aperture = create_aperture(pupil_radius, epsilon, grid_size)
    otf = calculate_otf(aperture)
    mtf = calculate_mtf(otf)
    plt.plot(mtf, label=f'ε = {epsilon:.1f}')

unshielded_aperture = create_aperture(pupil_radius, 0, grid_size)
unshielded_otf = calculate_otf(unshielded_aperture)
unshielded_psf = calculate_psf(unshielded_otf)
unshielded_mtf = calculate_mtf(unshielded_otf)
plt.plot(unshielded_mtf, label='Без экранирования', color='purple', lw=1)

plt.title('Сравнение функции передачи модуляции (ФПМ)')
plt.xlabel('Пространственная частота')
plt.ylabel('Модуляция')
plt.legend()
plt.grid(True)
plt.show()

fig, axes = plt.subplots(2, len(epsilons) + 1, figsize=(20, 8))

axes[0, 0].imshow(unshielded_aperture, cmap='gray', extent=[-pupil_radius, pupil_radius, -pupil_radius, pupil_radius])
axes[0, 0].set_title('Апертура без экранирования')

for i, epsilon in enumerate(epsilons):
    aperture = create_aperture(pupil_radius, epsilon, grid_size)
    axes[0, i + 1].imshow(aperture, cmap='gray', extent=[-pupil_radius, pupil_radius, -pupil_radius, pupil_radius])
    axes[0, i + 1].set_title(f'Апертура с ε = {epsilon:.1f}')

axes[1, 0].imshow(unshielded_psf, extent=[-pupil_radius, pupil_radius, -pupil_radius, pupil_radius])
axes[1, 0].set_title('ФРТ без экранирования')

for i, epsilon in enumerate(epsilons):
    otf = calculate_otf(create_aperture(pupil_radius, epsilon, grid_size))
    psf = calculate_psf(otf)
    axes[1, i + 1].imshow(psf, extent=[-pupil_radius, pupil_radius, -pupil_radius, pupil_radius])
    axes[1, i + 1].set_title(f'ФРТ с ε = {epsilon:.1f}')

plt.tight_layout()
plt.show()
