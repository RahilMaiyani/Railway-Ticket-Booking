# Railway Ticket Booking System (RTBS)

## Overview
The **Railway Ticket Booking System (RTBS)** is a Django-based web application for searching, booking, and managing train tickets. It includes an Admin Dashboard with integrated machine learning forecasting to analyze and predict passenger booking patterns.

---

## Features

### Passenger Module
- User registration & authentication (login/signup).  
- Train search by source, destination, and date.  
- Real-time ticket booking & seat allocation.  
- Booking history and ticket management.

### Admin Panel
- Interactive dashboard with charts (Chart.js).  
- Manage trains, schedules, and bookings.  
- Revenue, passenger trends, and booking statistics.  
- ML-powered demand forecasting to predict passenger counts and optimize operations.

### Machine Learning Integration
- Historical booking data is processed using pandas/NumPy.  
- Uses scikit-learn (LinearRegression) for short-term passenger forecasting.  
- Zero-booking dates are filtered to improve model accuracy.  
- Simple weekend uplift and lower-bound guards are applied to predictions.

---

## Tech Stack
- **Backend:** Django (Python)  
- **Frontend:** HTML, CSS, Bootstrap, JavaScript (Chart.js)  
- **Database:** SQLite3 (development; swap to PostgreSQL for production)  
- **Machine Learning:** pandas, NumPy, scikit-learn  
- **PDF Export:** xhtml2pdf (for booking exports)  
- **Editor:** VS Code

---

## Project Structure
    RTBS/
    │── bookings/          # Django app for bookings (models, views, templates)
    │── static/            # CSS, JS, images
    │── templates/         # HTML templates
    │── manage.py          # Django entry point
    │── db.sqlite3         # Development DB (should be in .gitignore)
    │── requirements.txt   # Python dependencies
    │── README.md          # This file

---

## Installation & Setup

1. Clone the repository
    git clone https://github.com/YOUR-USERNAME/RTBS_Final.git
    cd RTBS_Final

2. Create and activate a virtual environment
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS / Linux
    # source venv/bin/activate

3. Install dependencies
    pip install -r requirements.txt

4. Apply database migrations
    python manage.py makemigrations
    python manage.py migrate

5. Create an admin user
    python manage.py createsuperuser

6. Run the development server
    python manage.py runserver

Open: http://127.0.0.1:8000/  
Admin dashboard (staff only): http://127.0.0.1:8000/admin-dashboard/

---

## Admin Dashboard (Highlights)
- Dashboard KPIs: total trains, seats, bookings for selected date.  
- Booking trends visualized for recent days; zero-booking days are ignored when producing trend lines.  
- 7-day passenger forecast (linear regression on day number, weekend uplift, clamped to sensible lower bounds).  
- Export bookings for a date range to PDF.

---

## Future Improvements
- Move to PostgreSQL for production.  
- Replace xhtml2pdf with WeasyPrint or another robust renderer if high-quality PDFs are required.  
- Enhance forecasting with ARIMA/Prophet or a small neural net; persist predictions via a scheduled job (cron/Celery).  
- Add payment gateway integration (Razorpay / Stripe) and transactional email/SMS confirmations.  
- Add role-based access and pagination/filtering for admin exports.
