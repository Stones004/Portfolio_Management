from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('efficient_frontier/', views.efficient_frontier, name='efficient_frontier'),


]

