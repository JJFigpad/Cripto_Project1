from django.shortcuts import render
from PIL import Image
import numpy as np
import os
from django.conf import settings
from .forms import ImageUploadForm

def index(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            image_path = save_uploaded_file(image)
            binary_image = convert_to_binary(image_path)
            share1, share2 = create_shares(binary_image)
            share1_path, share2_path, overlay_path = save_images(share1, share2)
            return render(request, 'result.html', {
                'share1_path': share1_path.replace(settings.MEDIA_ROOT, settings.MEDIA_URL),
                'share2_path': share2_path.replace(settings.MEDIA_ROOT, settings.MEDIA_URL),
                'overlay_path': overlay_path.replace(settings.MEDIA_ROOT, settings.MEDIA_URL),
                'original_image_path': image_path.replace(settings.MEDIA_ROOT, settings.MEDIA_URL),
            })
    else:
        form = ImageUploadForm()
    return render(request, 'index.html', {'form': form})

def save_uploaded_file(f):
    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    file_path = os.path.join(upload_dir, f.name)
    with open(file_path, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    return file_path

def convert_to_binary(image_path, threshold=128):
    image = Image.open(image_path).convert('L')  # Convertir a escala de grises
    image = image.point(lambda p: p > threshold and 255)  # Binarizar la imagen
    return image

def create_shares(binary_image):
    width, height = binary_image.size
    share1 = Image.new('1', (width * 2, height * 2))
    share2 = Image.new('1', (width * 2, height * 2))

    pixels = binary_image.load()
    pixels1 = share1.load()
    pixels2 = share2.load()

    for y in range(height):
        for x in range(width):
            pixel = pixels[x, y]
            pattern = np.random.randint(0, 2)

            if pixel == 0:  # Negro
                if pattern == 0:
                    pixels1[2 * x, 2 * y] = 0
                    pixels1[2 * x + 1, 2 * y + 1] = 0
                    pixels1[2 * x + 1, 2 * y] = 1
                    pixels1[2 * x, 2 * y + 1] = 1

                    pixels2[2 * x, 2 * y] = 0
                    pixels2[2 * x + 1, 2 * y + 1] = 0
                    pixels2[2 * x + 1, 2 * y] = 1
                    pixels2[2 * x, 2 * y + 1] = 1
                else:
                    pixels1[2 * x, 2 * y] = 1
                    pixels1[2 * x + 1, 2 * y + 1] = 1
                    pixels1[2 * x + 1, 2 * y] = 0
                    pixels1[2 * x, 2 * y + 1] = 0

                    pixels2[2 * x, 2 * y] = 1
                    pixels2[2 * x + 1, 2 * y + 1] = 1
                    pixels2[2 * x + 1, 2 * y] = 0
                    pixels2[2 * x, 2 * y + 1] = 0
            else:  # Blanco
                if pattern == 0:
                    pixels1[2 * x, 2 * y] = 0
                    pixels1[2 * x + 1, 2 * y + 1] = 1
                    pixels1[2 * x + 1, 2 * y] = 0
                    pixels1[2 * x, 2 * y + 1] = 1

                    pixels2[2 * x, 2 * y] = 1
                    pixels2[2 * x + 1, 2 * y + 1] = 0
                    pixels2[2 * x + 1, 2 * y] = 1
                    pixels2[2 * x, 2 * y + 1] = 0
                else:
                    pixels1[2 * x, 2 * y] = 1
                    pixels1[2 * x + 1, 2 * y + 1] = 0
                    pixels1[2 * x + 1, 2 * y] = 1
                    pixels1[2 * x, 2 * y + 1] = 0

                    pixels2[2 * x, 2 * y] = 0
                    pixels2[2 * x + 1, 2 * y + 1] = 1
                    pixels2[2 * x + 1, 2 * y] = 0
                    pixels2[2 * x, 2 * y + 1] = 1

    return share1, share2

def save_images(share1, share2):
    output_dir = os.path.join(settings.MEDIA_ROOT, 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    share1_path = os.path.join(output_dir, 'share1.png')
    share2_path = os.path.join(output_dir, 'share2.png')
    overlay_path = os.path.join(output_dir, 'overlay.png')
    
    share1.save(share1_path)
    share2.save(share2_path)
    
    overlay = overlay_shares(share1, share2)
    overlay.save(overlay_path)
    
    return share1_path, share2_path, overlay_path

def overlay_shares(share1, share2):
    width, height = share1.size
    result = Image.new('1', (width, height))
    pixels1 = share1.load()
    pixels2 = share2.load()
    pixels_result = result.load()

    for y in range(height):
        for x in range(width):
            pixels_result[x, y] = pixels1[x, y] and pixels2[x, y]

    return result
