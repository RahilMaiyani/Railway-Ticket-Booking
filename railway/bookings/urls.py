from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('search/', views.search_trains, name='search_trains'),
    path('book/<int:train_id>/', views.book_train, name='book_train'),
    path('preview/<int:booking_id>/', views.booking_preview, name='booking_preview'),
    path('mock-pay/<int:booking_id>/', views.mock_pay, name='mock_pay'),
    path('ticket/<int:booking_id>/', views.ticket_success, name='ticket_success'),
    path('ticket/<int:booking_id>/download/', views.download_ticket, name='download_ticket'),
    path('my-bookings/', views.my_bookings, name='my_bookings'),
    path('profile/', views.profile_view, name='profile'),
    path('register/', views.register, name='register'),
    path('admin-dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('admin-dashboard/export-bookings-csv/', views.export_bookings_csv, name='admin_export_bookings_csv'),
    path('admin-dashboard/export-bookings-pdf/', views.export_bookings_pdf, name='admin_export_bookings_pdf'),
]