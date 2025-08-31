import cv2
import numpy as np
import time
import core



def main():
    resolutions = {
        '360p': (360, 480),
        'SD': (480, 640),
        'HD': (720, 1280),
        'FHD': (1080, 1920),
        'QHD': (1440, 2560),
        'UHD': (2160, 3840)
    }

    Y, X = resolutions['360p']

    SCALE = 2500
    C_REAL = -0.18
    C_IMAG = -0.65

    cv2.namedWindow('Fractal')
    DRAGGING, START_X, START_Y, OFFSET_X, OFFSET_Y = False, 0, 0, 0, 0

    img = np.zeros((Y, X), np.uint8)
    start = time.time()
    img = core.render_julia_divided(img, const=C_REAL + C_IMAG * 1j, scale=SCALE, max_iterations=200, chunk_size=100)
    # img = render_julia(img, const=0.36-0.06j, scale=700, max_iterations=100)
    end = time.time()
    print(f'Rendering time: {end - start} s \n'
          f'Rendering speed: {round(X * Y / (end - start) * 1e-6, 5)} Mpixel/s')

    rgb = np.zeros((img.shape[:2] + (3,)), dtype=np.uint8)
    dark = np.array((100, 60, 30)[::-1])
    medium = np.array((130, 150, 120)[::-1])
    light = np.array((250, 200, 160)[::-1])

    normalized = img / 255.0

    mask1 = normalized <= 0.5
    t1 = normalized[mask1] / 0.5
    rgb[mask1] = (dark + (medium - dark) * t1[..., None]).astype(np.uint8)

    mask2 = normalized > 0.5
    t2 = (normalized[mask2] - 0.5) / 0.5
    rgb[mask2] = (medium + (light - medium) * t2[..., None]).astype(np.uint8)

    while True:

        cv2.imshow('Fractal', rgb)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f'saved-renders/render_{core.get_new_id()}.png', rgb)


if __name__=="__main__":
    main()
