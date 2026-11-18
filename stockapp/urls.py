from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('efficient_frontier/', views.efficient_frontier, name='efficient_frontier'),
    path('login/', auth_views.LoginView.as_view(template_name='stockapp/login.html', redirect_authenticated_user=True), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='login'), name='logout'),
    path('register/', views.register, name='register'),
    path('dcf/', views.dcf_view, name='dcf'),
    path('dcf/portfolio/', views.portfolio_dcf, name='portfolio_dcf'),
    path('dcf/value/', views.dcf_value_api, name='dcf_value_api'),
]

