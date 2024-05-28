from django import forms

class ImageUploadForm(forms.Form):
    image = forms.ImageField()

class StepsForm(forms.Form):
    num_steps = forms.IntegerField(label='Número de pasos', min_value=1, max_value=10, initial=1)


class ImageUploadForm_(forms.Form):
    image = forms.ImageField(label='Imagen Principal')
    watermark = forms.ImageField(label='Imagen de Marca de Agua',required=False)
    marked_image = forms.ImageField(label='Seleccione imagen con marca de agua', required=False)
    action = forms.ChoiceField(
        choices=[('mark', 'Crear marca de agua'), ('recover', 'Recuperar marca de agua')],
        widget=forms.RadioSelect,
        label='Selecciona una opción:'
    )
