from django import forms

class ImageUploadForm(forms.Form):
    image = forms.ImageField()

class StepsForm(forms.Form):
    num_steps = forms.IntegerField(label='Número de pasos', min_value=1, max_value=10, initial=1)
