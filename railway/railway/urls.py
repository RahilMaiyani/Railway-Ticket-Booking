from django.contrib import admin
from django.urls import path, include
from django.contrib.auth import views as auth_views
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('bookings.urls')),
    path('accounts/', include('django.contrib.auth.urls')), 
    path('login/', auth_views.LoginView.as_view(template_name='bookings/login.html'), name='login'),
    
]
