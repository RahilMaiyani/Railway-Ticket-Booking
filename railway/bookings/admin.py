from django.contrib import admin
from .models import (User, Station, BerthType, Train, TrainBerthAvailability,
                     RouteStop, DailyTrainAvailability, Booking, Passenger, Payment)

admin.site.register(User)
admin.site.register(Station)
admin.site.register(BerthType)
admin.site.register(Train)
admin.site.register(TrainBerthAvailability)
admin.site.register(RouteStop)
admin.site.register(DailyTrainAvailability)
admin.site.register(Booking)
admin.site.register(Passenger)
admin.site.register(Payment)
