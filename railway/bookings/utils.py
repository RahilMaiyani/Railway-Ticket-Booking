
from decimal import Decimal, ROUND_HALF_UP
from django.conf import settings

def calculate_fare(distance_km: int, price_per_km: Decimal, passengers: int):
    base_per = (Decimal(distance_km) * Decimal(price_per_km)).quantize(Decimal('0.01'), ROUND_HALF_UP)
    total_base = (base_per * passengers).quantize(Decimal('0.01'), ROUND_HALF_UP)
    gst = (total_base * Decimal(settings.TAX_CONFIG['GST_PCT']) / Decimal(100)).quantize(Decimal('0.01'), ROUND_HALF_UP)
    service = (total_base * Decimal(settings.TAX_CONFIG['SERVICE_PCT']) / Decimal(100)).quantize(Decimal('0.01'), ROUND_HALF_UP)
    convenience = Decimal(settings.TAX_CONFIG['CONVENIENCE_FEE']).quantize(Decimal('0.01'), ROUND_HALF_UP)
    total = (total_base + gst + service + convenience).quantize(Decimal('0.01'), ROUND_HALF_UP)
    return {
        'base_per': base_per,
        'total_base': total_base,
        'gst': gst,
        'service': service,
        'convenience': convenience,
        'total': total
    }
