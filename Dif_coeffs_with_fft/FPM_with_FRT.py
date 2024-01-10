import numpy as np
import matplotlib.pyplot as plt

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
unshielded_mtf = calculate_mtf(unshielded_otf)
plt.plot(unshielded_mtf, label='Без экранирования', color='purple', lw=1)

plt.title('Сравнение функции передачи модуляции (ФПМ)')
plt.xlabel('Пространственная частота')
plt.ylabel('Модуляция')
plt.legend()
plt.grid(True)
plt.show()


def calculate_psf(otf):
    psf = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(otf)))
    psf_normalized = np.abs(psf) ** 2
    return psf_normalized / psf_normalized.max()


fig, axes = plt.subplots(1, len(epsilons) + 1, figsize=(20, 4))

unshielded_psf = calculate_psf(unshielded_otf)
im = axes[0].imshow(unshielded_psf, extent=[-pupil_radius, pupil_radius, -pupil_radius, pupil_radius])
axes[0].set_title('ФРТ без экранирования')
plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

for i, epsilon in enumerate(epsilons):
    aperture = create_aperture(pupil_radius, epsilon, grid_size)
    otf = calculate_otf(aperture)
    psf = calculate_psf(otf)
    im = axes[i + 1].imshow(psf, extent=[-pupil_radius, pupil_radius, -pupil_radius, pupil_radius])
    axes[i + 1].set_title(f'ФРТ с ε = {epsilon:.1f}')
    plt.colorbar(im, ax=axes[i + 1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
