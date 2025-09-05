from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta, date
from bookings.models import Train, TrainBerthAvailability, DailyTrainAvailability

class Command(BaseCommand):
    help = 'Populate DailyTrainAvailability for next N days (default 30)'

    def add_arguments(self, parser):
        parser.add_argument('--days', type=int, default=30)

    def handle(self, *args, **options):
        days = options['days']
        today = date.today()
        for d in range(days):
            target = today + timedelta(days=d)
            for t in Train.objects.all():
                for tb in TrainBerthAvailability.objects.filter(train=t):
                    obj, created = DailyTrainAvailability.objects.get_or_create(
                        train=t, berth_type=tb.berth_type, date=target,
                        defaults={'available_seats': tb.capacity}
                    )
                    if created:
                        self.stdout.write(f"Created availability for {t} {tb.berth_type.code} {target}")
        self.stdout.write("Done.")
