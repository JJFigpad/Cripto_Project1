from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('descripcion/', views.descripcion_app, name='descripcion_app'),
]