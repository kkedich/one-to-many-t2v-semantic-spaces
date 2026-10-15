import numpy as np
from PIL import Image
from skimage import img_as_ubyte
from skimage.transform import rotate


def load_img(path, grayscale=False, target_size=None, flip=False, rotate_img=False, rotate_angle=0):
    """Load an image into PIL format.
    # Arguments
        path: path to image file
        grayscale: boolean
        target_size: None (default to original size)
            or (img_height, img_width)
    """
    img = Image.open(path)
    if grayscale:
        img = img.convert('L')
    else:  # Ensure 3 channel even when loaded image is grayscale
        img = img.convert('RGB')

    if target_size:
        # Function resize takes img.resize((new_width, new_height))
        img = img.resize((target_size[1], target_size[0]))

    if flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif rotate_img:
        img = rotate_image(img, rotate_angle, grayscale)

    return img


def rotate_image(img, angle, grayscale=False):
    mode = 'reflect'
    if grayscale:
        mode = 'constant'

    new_img = img
    if not isinstance(new_img, np.ndarray):
        new_img = np.array(new_img)
    new_img = rotate(new_img, angle=angle, resize=False, mode=mode)

    if img.dtype == np.uint8:
        new_img = img_as_ubyte(new_img)
    else:
        raise NotImplementedError

    return new_img


def load_image(image_file, img_height, img_width):
    """Load ONE image and return the corresponding input for the model."""

    input_image = np.zeros((1, img_width, img_height, 3), dtype=np.float32)
    img = load_img(image_file, target_size=(img_height, img_width))
    input_image[0, :, :, :] = np.asarray(img, dtype='float32')

    return input_image


def extract_frames_from_gif(input_file: str):
    import imageio.v3 as iio

    # Read the GIF and extract all frames
    # Setting index=None tells imageio to read all images in the file
    # and stack them along the first axis.
    frames = iio.imread(input_file, index=None)

    # 'frames' will be a NumPy array with the shape (num_frames, height, width, channels)
    # For example, if you have a GIF with 36 frames, each 150x200 pixels with 3 color channels (RGB):
    # print(frames.shape) would output: (36, 150, 200, 3)

    # You can also iterate through the frames
    for i, frame in enumerate(frames):
        # Process each frame as needed
        # For example, save each frame as a separate image file
        iio.imwrite(f"frame_{i:03d}.png", frame)
