from PIL import Image
import numpy as np
from django.conf import settings
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from .forms import ImageUploadForm, StepsForm
import os

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

def marcas_de_agua(request):
    return render(request, 'marcas_de_agua.html')