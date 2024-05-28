from PIL import Image
import numpy as np
from django.conf import settings
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from .forms import ImageUploadForm, StepsForm
import os
import cv2
import pywt
from scipy.linalg import svd
from skimage import img_as_float



def convert_to_binary(image_path, threshold=128):
    try:
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        image = image.point(lambda p: p > threshold and 255)  # Binarize the image
        return image
    except Exception as e:
        print(f"Error converting image to binary: {e}")
        return None

def create_shares(binary_image):
    if binary_image is None:
        return None, None

    width, height = binary_image.size
    share1 = Image.new('1', (width * 2, height * 2))
    share2 = Image.new('1', (width * 2, height * 2))

    pixels = np.array(binary_image)
    pixels1 = np.array(share1)
    pixels2 = np.array(share2)

    for y in range(height):
        for x in range(width):
            pixel = pixels[y, x]
            pattern = np.random.randint(0, 2)
            rn = np.random.randint(0, 2)

            if pixel == 0:  # Black
                if pattern == 0:
                    pixels1[2 * y + rn, 2 * x + rn] = pixels1[2 * y + (1 + rn) % 2, 2 * x + (1 + rn) % 2] = 0
                    pixels1[2 * y + rn, 2 * x + (1 + rn) % 2] = pixels1[2 * y + (1 + rn) % 2, 2 * x + rn] = 1

                    pixels2[2 * y + rn, 2 * x + rn] = pixels1[2 * y + (1 + rn) % 2, 2 * x + (1 + rn) % 2] = 0
                    pixels2[2 * y + rn, 2 * x + (1 + rn) % 2] = pixels1[2 * y + (1 + rn) % 2, 2 * x + rn] = 1
                else:
                    pixels1[2 * y + rn, 2 * x + rn] = pixels1[2 * y + (1 + rn) % 2, 2 * x + (1 + rn) % 2] = 1
                    pixels1[2 * y + rn, 2 * x + (1 + rn) % 2] = pixels1[2 * y + (1 + rn) % 2, 2 * x + rn] = 0

                    pixels2[2 * y + rn, 2 * x + rn] = pixels2[2 * y + (1 + rn) % 2, 2 * x + (1 + rn) % 2] = 1
                    pixels2[2 * y + rn, 2 * x + (1 + rn) % 2] = pixels2[2 * y + (1 + rn) % 2, 2 * x + rn] = 0
            else:  # White
                if pattern == 0:
                    pixels1[2 * y + rn, 2 * x + rn] = pixels1[2 * y + rn, 2 * x + (1 + rn) % 2] = 0
                    pixels1[2 * y + (1 + rn) % 2, 2 * x + rn] = pixels1[2 * y + (1 + rn) % 2, 2 * x + (1 + rn) % 2] = 1

                    pixels2[2 * y + rn, 2 * x + rn] = pixels2[2 * y + rn, 2 * x + (1 + rn) % 2] = 1
                    pixels2[2 * y + (1 + rn) % 2, 2 * x + rn] = pixels2[2 * y + (1 + rn) % 2, 2 * x + (1 + rn) % 2] = 0
                else:
                    pixels1[2 * y + rn, 2 * x + rn] = pixels1[2 * y + rn, 2 * x + (1 + rn) % 2] = 1
                    pixels1[2 * y + (1 + rn) % 2, 2 * x + rn] = pixels1[2 * y + (1 + rn) % 2, 2 * x + (1 + rn) % 2] = 0

                    pixels2[2 * y + rn, 2 * x + rn] = pixels2[2 * y + rn, 2 * x + (1 + rn) % 2] = 0
                    pixels2[2 * y + (1 + rn) % 2, 2 * x + rn] = pixels2[2 * y + (1 + rn) % 2, 2 * x + (1 + rn) % 2] = 1

    return Image.fromarray(pixels1), Image.fromarray(pixels2)

def save_image(share, share_path):
    try:
        share.save(share_path)
    except Exception as e:
        print(f"Error saving images: {e}")

def overlay_shares(shares):
    if len(shares) < 2:
        return None

    pixels1 = np.array(shares[0])
    pixels2 = np.array(shares[1])
    pixels_result = np.bitwise_and(pixels1, pixels2)

    for i in range(2, len(shares), 2):
        pixels1 = np.array(shares[i])
        pixels2 = np.array(shares[i + 1])
        pxtemp = np.bitwise_and(pixels1, pixels2)
        pixels_result = np.bitwise_and(pixels_result, pxtemp)

    return Image.fromarray(pixels_result)

def invert_image(image):
    inverted_image = Image.eval(image, lambda p: 255 - p)
    return inverted_image

def main(image_path, num):
    shares = []
    sharesPath = []
    output_dir = os.path.join(settings.MEDIA_ROOT, 'output')
    os.makedirs(output_dir, exist_ok=True)

    for i in range(2**num):
        sharesPath.append(os.path.join(output_dir, f'share{i + 1}.png'))
    overlay_path = os.path.join(output_dir, 'overlay.png')

    binary_image = convert_to_binary(image_path)
    if binary_image is None:
        return [], None, None

    share1, share2 = create_shares(binary_image)
    shares = [share1, share2]
    if num > 1:
        c = 1
        while c < num:
            sharest = shares.copy()
            for sh in sharest:
                temp = create_shares(sh)
                shares.append(temp[0])
                shares.append(temp[1])
                shares.pop(0)
            c += 1

    for i in range(len(shares)):
        save_image(shares[i], sharesPath[i])

    overlay = overlay_shares(shares)
    overlay = invert_image(overlay)
    if overlay:
        overlay.save(overlay_path)

    return sharesPath, overlay_path, image_path

def index(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        steps_form = StepsForm(request.POST)
        if form.is_valid() and steps_form.is_valid():
            image = form.cleaned_data['image']
            num_steps = steps_form.cleaned_data['num_steps']
            image_path = default_storage.save('uploads/' + image.name, ContentFile(image.read()))
            image_full_path = os.path.join(settings.MEDIA_ROOT, image_path)
            sharesPath, overlay_path, original_image_path = main(image_full_path, num_steps)
            if not sharesPath:
                return render(request, 'index.html', {
                    'form': form,
                    'steps_form': steps_form,
                    'error': 'Error processing the image.'
                })
            return render(request, 'result.html', {
                'share_paths': [os.path.join(settings.MEDIA_URL, os.path.relpath(path, settings.MEDIA_ROOT)) for path in sharesPath],
                'overlay_path': os.path.join(settings.MEDIA_URL, os.path.relpath(overlay_path, settings.MEDIA_ROOT)),
                'original_image_path': os.path.join(settings.MEDIA_URL, image_path),
            })
    else:
        form = ImageUploadForm()
        steps_form = StepsForm()
    return render(request, 'index.html', {'form': form, 'steps_form': steps_form})

def descripcion_app(request):
    return render(request, 'info.html')

def criptografia_visual(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        steps_form = StepsForm(request.POST)
        if form.is_valid() and steps_form.is_valid():
            image = form.cleaned_data['image']
            num_steps = steps_form.cleaned_data['num_steps']
            image_path = default_storage.save('uploads/' + image.name, ContentFile(image.read()))
            image_full_path = os.path.join(settings.MEDIA_ROOT, image_path)
            sharesPath, overlay_path, original_image_path = main(image_full_path, num_steps)
            if not sharesPath:
                return render(request, 'criptografia_visual.html', {
                    'form': form,
                    'steps_form': steps_form,
                    'error': 'Error processing the image.'
                })
            return render(request, 'result.html', {
                'form': form,
                'steps_form': steps_form,
                'share_paths': [os.path.join(settings.MEDIA_URL, os.path.relpath(path, settings.MEDIA_ROOT)) for path in sharesPath],
                'overlay_path': os.path.join(settings.MEDIA_URL, os.path.relpath(overlay_path, settings.MEDIA_ROOT)),
                'original_image_path': os.path.join(settings.MEDIA_URL, image_path),
            })
    else:
        form = ImageUploadForm()
        steps_form = StepsForm()
    return render(request, 'criptografia_visual.html', {'form': form, 'steps_form': steps_form})

from .forms import ImageUploadForm_ 

# Funciones auxiliares
def block_process(img, block_size, func):
    h, w = img.shape
    out = np.zeros((h, w))
    for i in range(0, h, block_size[0]):
        for j in range(0, w, block_size[1]):
            block = img[i:i + block_size[0], j:j + block_size[1]]
            out[i:i + block_size[0], j:j + block_size[1]] = func(block)
    return out

def zigzag(input):
    h, w = input.shape
    return np.concatenate([np.diagonal(input[::-1, :], i)[::(2*(i % 2)-1)] for i in range(1-h, h)])

def dwt2d(block):
    coeffs2 = pywt.dwt2(block, 'haar')
    LL, (LH, HL, HH) = coeffs2
    return np.vstack((np.hstack((LL, LH)), np.hstack((HL, HH))))

def idwt2d(block):
    LL = block[:block.shape[0]//2, :block.shape[1]//2]
    LH = block[:block.shape[0]//2, block.shape[1]//2:]
    HL = block[block.shape[0]//2:, :block.shape[1]//2]
    HH = block[block.shape[0]//2:, block.shape[1]//2:]
    coeffs2 = LL, (LH, HL, HH)
    return pywt.idwt2(coeffs2, 'haar')

# Función principal para procesar la marca de agua
def insert_watermark(imagen_path, marca_path, alfa=0.08, block_size=(8, 8)):
    I = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)
    if I is None:
        raise FileNotFoundError(f"No se pudo encontrar la imagen en la ruta: {imagen_path}")
    I = cv2.resize(I, (256, 256))
    I = img_as_float(I)

    M = cv2.imread(marca_path, cv2.IMREAD_GRAYSCALE)
    if M is None:
        raise FileNotFoundError(f"No se pudo encontrar la imagen en la ruta: {marca_path}")
    M = cv2.resize(M, (256, 256))
    M = img_as_float(M)

    I1 = block_process(I, block_size, dwt2d)
    M1 = block_process(M, block_size, dwt2d)

    mask = np.array([
        [1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ])

    C1 = block_process(I1, block_size, lambda x: mask * x)
    C2 = block_process(M1, block_size, lambda x: mask * x)

    Imr = C1 + alfa * C2

    u, s, v = svd(Imr)
    a, b, c = svd(C1)

    Imarcada = a @ np.diag(s) @ c
    Imarcada = block_process(Imr, block_size, idwt2d)

    output_dir = os.path.join(settings.MEDIA_ROOT, 'output')
    os.makedirs(output_dir, exist_ok=True)
    imagen_marcada_path = os.path.join(output_dir, 'imagenmarcada.jpg')
    cv2.imwrite(imagen_marcada_path, (Imarcada * 255).astype(np.uint8))

    """ # Recuperación de la marca utilizando la imagen marcada generada
    C3 = block_process(Imarcada, block_size, dwt2d)
    C3 = block_process(C3, block_size, lambda x: mask * x)

    e, f, g = svd(C3)
    C2r = ((u @ np.diag(f) @ v) - C1) * 1 / alfa
    marcarecuperada = block_process(C2r, block_size, idwt2d)

    marcarecuperada_path = os.path.join(output_dir, 'marcarecuperada.jpg')
    cv2.imwrite(marcarecuperada_path, (marcarecuperada * 255).astype(np.uint8))
 """
    return imagen_marcada_path

def recover_watermark(original_image_path, marked_image_path, alfa=0.08):
    block_size = (8, 8)

    I = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    if I is None:
        raise FileNotFoundError(f"No se pudo encontrar la imagen en la ruta: {original_image_path}")
    I = cv2.resize(I, (256, 256))
    I = img_as_float(I)

    b = cv2.imread(marked_image_path, cv2.IMREAD_GRAYSCALE)
    if b is None:
        raise FileNotFoundError(f"No se pudo encontrar la imagen en la ruta: {marked_image_path}")
    b = cv2.resize(b, (256, 256))
    A = img_as_float(b)

    mask = np.array([
        [1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ])

    I1 = block_process(I, block_size, dwt2d)
    C1 = block_process(I1, block_size, lambda x: mask * x)
    C3 = block_process(A, block_size, dwt2d)
    C3 = block_process(C3, block_size, lambda x: mask * x)

    u, s, v = svd(C3)
    e, f, g = svd(C3)
    C2r = ((u @ np.diag(f) @ v) - C1) * 1 / alfa
    marcarecuperada = block_process(C2r, block_size, idwt2d)


    output_dir = os.path.join(settings.MEDIA_ROOT, 'output')
    os.makedirs(output_dir, exist_ok=True)
    marcarecuperada_path = os.path.join(output_dir, 'marcarecuperada.jpg')
    cv2.imwrite(marcarecuperada_path, (marcarecuperada * 255).astype(np.uint8))
    #cv2.imwrite("marca_recuperada.jpg", (marcarecuperada * 255).astype(np.uint8))
    return marcarecuperada_path


# Vista para manejar la marca de agua
def marcas_de_agua(request):
    error_message = None

    if request.method == 'POST':
        form = ImageUploadForm_(request.POST, request.FILES)
        if form.is_valid():
            action = form.cleaned_data['action']
            image = form.cleaned_data['image']
            image_path = default_storage.save('uploads/' + image.name, ContentFile(image.read()))
            image_full_path = default_storage.path(image_path)

            if action == 'mark':
                watermark = form.cleaned_data['watermark']
                if not watermark:
                    error_message = "Watermark image is required for marking the image."
                else:
                    watermark_path = default_storage.save('uploads/' + watermark.name, ContentFile(watermark.read()))
                    watermark_full_path = default_storage.path(watermark_path)
                    try:
                        imagen_marcada_path = insert_watermark(image_full_path, watermark_full_path)
                        imagen_marcada_rel_path = default_storage.url(imagen_marcada_path)
                        image_full_path = default_storage.url(image_full_path)
                        watermark_full_path = default_storage.url(watermark_full_path)
                        return render(request, 'result_marcas_de_agua.html', {
                            'form': form,
                            'watermark_full_path': os.path.join(settings.MEDIA_URL, watermark_full_path),
                            'image_full_path': os.path.join(settings.MEDIA_URL, image_full_path),
                            'imagen_marcada_path': os.path.join(settings.MEDIA_URL, imagen_marcada_rel_path),
                        })
                    except FileNotFoundError as e:
                        error_message = str(e)

            elif action == 'recover':
                marked_image = form.cleaned_data['marked_image']
                if not marked_image:
                    error_message = "Watermarked image is required for recovering the watermark."
                else:
                    marked_image_path = default_storage.save('uploads/' + marked_image.name, ContentFile(marked_image.read()))
                    marked_image_full_path = default_storage.path(marked_image_path)
                    try:
                        marcarecuperada_path = recover_watermark(image_full_path, marked_image_full_path)
                        marcarecuperada_rel_path = default_storage.url(marcarecuperada_path)
                        image_full_path = default_storage.url(image_full_path)
                        marked_image_full_path = default_storage.url(marked_image_full_path)
                        return render(request, 'result_marcas_de_agua.html', {
                            'form': form,
                            'image_full_path': os.path.join(settings.MEDIA_URL, image_full_path),
                            'marked_image_full_path': os.path.join(settings.MEDIA_URL, marked_image_full_path),
                            'marcarecuperada_path': os.path.join(settings.MEDIA_URL, marcarecuperada_rel_path),
                        })
                    except FileNotFoundError as e:
                        error_message = str(e)
        else:
            error_message = "Invalid form submission. Please correct the errors and try again."

    else:
        form = ImageUploadForm_()

    return render(request, 'marcas_de_agua.html', {'form': form, 'error': error_message})
