from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('descripcion/', views.descripcion_app, name='descripcion_app'),
    path('criptografia_visual/', views.criptografia_visual, name='pagina_criptografia_visual'),
    path('marcas_de_agua/', views.marcas_de_agua, name='pagina_marcas_de_agua'),
]