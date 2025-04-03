from django.urls import path
from . import views


urlpatterns = [
    path('', views.landing, name='landing'),
    path('login/', views.user_login, name='login'),  # Updated
    path('signup/', views.user_signup, name='signup'),
    path('logout/',views.logout_view,name='logout'),
    path('home/', views.home, name="home"),
    path('summarize/', views.summarize, name="summarize"),
]