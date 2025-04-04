from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings


urlpatterns = [
    path('', views.landing, name='landing'),
    path('login/', views.user_login, name='login'),  # Updated
    path('signup/', views.user_signup, name='signup'),
    path('logout/',views.logout_view,name='logout'),
    path('home/', views.home, name="home"),
    path('summarize/', views.summarize, name="summarize"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)