from django.urls import path
from . import views

app_name = 'intelligence'

urlpatterns = [
    path('', views.index, name='index'),
    path('calc', views.calc, name='calc'),
    path('result', views.result, name='result')
]