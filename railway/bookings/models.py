from django.db import models
from django.conf import settings
from django.contrib.auth.models import AbstractUser
from django.core.validators import RegexValidator, MinValueValidator
from decimal import Decimal


class User(AbstractUser):
    mobile = models.CharField(max_length=10, validators=[RegexValidator(r'^\d{10}$')], blank=True, null=True)
    GENDER_CHOICES = [('M','Male'),('F','Female')]
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES, blank=True, null=True)
    dob = models.DateField(blank=True, null=True)

    def age(self):
        from datetime import date
        if not self.dob: return None
        today = date.today()
        return today.year - self.dob.year - ((today.month, today.day) < (self.dob.month, self.dob.day))

    def __str__(self):
        return self.get_full_name() or self.username

class Station(models.Model):
    name = models.CharField(max_length=255)
    code = models.CharField(max_length=10, unique=True)

    def __str__(self):
        return f"{self.name} ({self.code})"

class BerthType(models.Model):
    name = models.CharField(max_length=50)   # 'Sleeper'
    code = models.CharField(max_length=5)    # 'SL'
    price_per_km = models.DecimalField(max_digits=8, decimal_places=2, validators=[MinValueValidator(Decimal('0.0'))])

    def __str__(self):
        return f"{self.name} ({self.code})"

class Train(models.Model):
    number = models.CharField(max_length=20, unique=True)
    name = models.CharField(max_length=255)
    # Optional: default overall capacity, but we'll use TrainBerthAvailability to manage berth-wise capacity
    def __str__(self):
        return f"{self.number} - {self.name}"

class TrainBerthAvailability(models.Model):
    train = models.ForeignKey(Train, on_delete=models.CASCADE, related_name='berths')
    berth_type = models.ForeignKey(BerthType, on_delete=models.CASCADE)
    capacity = models.PositiveIntegerField(default=0)

    class Meta:
        unique_together = ('train', 'berth_type')

    def __str__(self):
        return f"{self.train} | {self.berth_type.code} | capacity {self.capacity}"

class RouteStop(models.Model):
    train = models.ForeignKey(Train, on_delete=models.CASCADE, related_name='stops')
    station = models.ForeignKey(Station, on_delete=models.CASCADE)
    sequence = models.PositiveIntegerField()
    arrival = models.TimeField(null=True, blank=True)
    departure = models.TimeField(null=True, blank=True)
    distance = models.PositiveIntegerField(help_text="Distance from origin in km")

    class Meta:
        unique_together = ('train', 'sequence')
        ordering = ['sequence']

    def __str__(self):
        return f"{self.train.number} stop {self.sequence} - {self.station.code}"

# availability per train/berth/date for atomic decrements
class DailyTrainAvailability(models.Model):
    train = models.ForeignKey(Train, on_delete=models.CASCADE)
    berth_type = models.ForeignKey(BerthType, on_delete=models.CASCADE)
    date = models.DateField()
    available_seats = models.PositiveIntegerField()

    class Meta:
        unique_together = ('train', 'berth_type', 'date')

    def __str__(self):
        return f"{self.train.number} | {self.berth_type.code} | {self.date} -> {self.available_seats}"

class Booking(models.Model):
    STATUS_CHOICES = [('PENDING','PENDING'), ('CONFIRMED','CONFIRMED'), ('CANCELLED','CANCELLED')]
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='bookings')
    train = models.ForeignKey(Train, on_delete=models.CASCADE)
    source = models.ForeignKey(Station, on_delete=models.CASCADE, related_name='source_bookings')
    destination = models.ForeignKey(Station, on_delete=models.CASCADE, related_name='destination_bookings')
    date_of_journey = models.DateField()
    berth_type = models.ForeignKey(BerthType, on_delete=models.CASCADE)
    passengers_count = models.PositiveIntegerField()
    total_fare = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal('0.00'))
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='PENDING')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Booking {self.pk} {self.user} {self.train.number} {self.date_of_journey}"

class Passenger(models.Model):
    booking = models.ForeignKey(Booking, on_delete=models.CASCADE, related_name='passengers')
    name = models.CharField(max_length=255)
    age = models.PositiveIntegerField()
    gender = models.CharField(max_length=1, choices=User.GENDER_CHOICES)
    seat_number = models.CharField(max_length=20, blank=True, null=True)

    def __str__(self):
        return f"{self.name} ({self.age})"

class Payment(models.Model):
    booking = models.OneToOneField(Booking, on_delete=models.CASCADE)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    status = models.CharField(max_length=10, choices=[('PENDING','PENDING'),('SUCCESS','SUCCESS'),('FAILED','FAILED')], default='PENDING')
    txn_id = models.CharField(max_length=255, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
