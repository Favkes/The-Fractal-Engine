import cv2
import numpy as np
import os



def get_new_id():
    # render_0.png
    if not os.path.exists('./saved-renders'):
        os.mkdir('./saved-renders')

    files = os.listdir('./saved-renders')
    id_s = [
        int(
            file.replace('render_', '').replace('.png', '')
        ) for file in files if file.startswith('render')
    ]
    id_s.sort()
    for i, id_ in enumerate(id_s):
        if i != id_:
            return i
    return len(id_s)


def render_pixel(value: complex, const: complex = 0, limit: int = 78, func = lambda z, c: z*z+c):
    iteration = 0
    while limit > 0:
        limit -= 1
        iteration += 1

        value = func(value, const)
        if abs(value) > 2:
            break
    return iteration


def render_submatrix(arr: np.ndarray, const: complex = 0, limit: int = 78, func = lambda z, c: z*z+c):
    values      = arr.astype(np.complex128)
    values_left = np.ones_like(arr, dtype=bool)
    output      = np.zeros_like(arr, dtype=np.uint64)

    iteration = 0
    while limit > 0:
        limit -= 1
        iteration += 1

        values[values_left] = func(values[values_left], const)
        escaped_mask = values_left & (np.abs(values) > 2)
        output[escaped_mask] = iteration
        values_left[escaped_mask] = False

    output[output == 0] = iteration
    return output


def complex_grid(start: tuple[int, int], shape: tuple[int, int]):
    height, width = shape

    im = np.arange(height)
    re = np.arange(width)

    imag = start[0] + im[:, np.newaxis]
    real = start[1] + re[np.newaxis, :]

    return real + imag * 1j



def render_julia_divided(img: np.ndarray,
                         const: complex = 0,
                         scale: float = 1,
                         func = lambda z, c: z*z+c,
                         max_iterations: int = 78,
                         chunk_size: int = 100,
                         center_shift: tuple[int, int] = (0, 0)):
    output = np.zeros_like(img, dtype=np.float64)
    y_centerer = round(output.shape[0] / 2) + center_shift[0]
    x_centerer = round(output.shape[1] / 2) + center_shift[1]

    y_div, y_mod = divmod(output.shape[0], chunk_size)
    x_div, x_mod = divmod(output.shape[1], chunk_size)

    chunk_square    = complex_grid((0, 0), (chunk_size, chunk_size))
    chunk_side      = complex_grid((0, 0), (chunk_size, x_mod))
    chunk_bottom    = complex_grid((0, 0), (y_mod, chunk_size))
    chunk_remainder = complex_grid((0, 0), (y_mod, x_mod))

    func_ = lambda arr: render_submatrix(arr, const, limit=max_iterations, func=func)

    for y in range(0, output.shape[0]-chunk_size, chunk_size):
        y_ = y - y_centerer
        for x in range(0, output.shape[1]-chunk_size, chunk_size):
            x_ = x - x_centerer

            # Normal tiles (chunk_size size)
            print(f'{y}:{y+chunk_size} {x}:{x+chunk_size}')
            chunk = (chunk_square + complex(x_, y_)) / scale
            output[y:y+chunk_size, x:x+chunk_size] = func_(chunk)

        # Rightside remainders
        x = output.shape[1] - x_mod
        x_ = x - x_centerer
        chunk = (chunk_side + complex(x_, y_)) / scale
        output[y:y+chunk_size, x:] = func_(chunk)

    # Bottomside remainders
    y = output.shape[0] - y_mod
    y_ = y - y_centerer
    for x in range(0, output.shape[1]-chunk_size, chunk_size):
        x_ = x - x_centerer
        chunk = (chunk_bottom + complex(x_, y_)) / scale
        output[y:, x:x+chunk_size] = func_(chunk)

    # Bottom-rightside remainder
    x = output.shape[1] - x_mod
    x_ = x - x_centerer
    chunk = (chunk_remainder + complex(x_, y_)) / scale
    output[y:, x:] = func_(chunk)

    y, x = output.shape[:2]
    y //= 2
    x //= 2
    output[y, x] = np.average(
        [output[y-1, x], output[y+1, x], output[y, x-1], output[y, x+1]]
    )
    M, m = np.max(output), np.min(output)
    M_ = 254 / M
    output *= M_

    return output.astype(np.uint8).copy()


def render_julia(img: np.ndarray,
                 const: complex = 0,
                 scale: float = 1,
                 func = lambda z, c: z*z+c,
                 max_iterations: int = 78):
    values = np.zeros_like(img, dtype=np.float64)
    y_centerer = round(values.shape[0] / 2)
    x_centerer = round(values.shape[1] / 2)

    for y in range(values.shape[0]):
        if y % 128 == 0:
            print(round(y/Y, 2), '%')
        y_ = y - y_centerer
        for x in range(values.shape[1]):
            x_ = x - x_centerer

            z = complex(x_, y_) / scale
            values[y, x] = render_pixel(z, const, limit=max_iterations, func=func)

    values[y_centerer, x_centerer] = 0
    M, m = np.max(values), np.min(values)
    M_ = 254 / M
    values *= M_

    return values.astype(np.uint8).copy()

