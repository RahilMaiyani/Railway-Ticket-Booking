
# Railway Ticket Booking System (Django + ML)

A Railway Ticket Booking System built with **Django 5**, **SQLite**, and a lightweight **ML forecasting** pipeline using **pandas**, **NumPy**, and **scikit-learn**.  
The system supports train search, real-time availability, booking, and ticketing, with a **rich Admin Dashboard** that includes KPIs, trend charts, and a **7‑day passenger forecast**.

---

## ✨ Features

### User-facing
- **Search Trains** by source, destination, and date.  
  ![Search Trains](Screenshot (17).png)

- **Available Trains** list with berth-wise availability and fare calculation.  
  ![Available Trains](Screenshot (18).png)

- **Booking Form** with passenger details (1–5 passengers).  
  ![Booking Form](Screenshot (19).png)

- **Ticket Preview & Confirmation** with breakdown of fare + GST/fees.  
  ![Ticket Preview](Screenshot (20).png)

- **My Bookings** section for history and ticket management.  
  ![My Bookings](Screenshot (21).png)

- **Login/Registration** with age validation (18+).  
  ![Login Page](Screenshot (24).png)

### Admin Panel (Staff Only)
- **KPIs**: total trains, available seats, bookings.  
  ![Admin Dashboard KPIs](Screenshot (22).png)

- **Forecasting**: passenger demand for the next 7 days using Linear Regression, weekend uplift, and zero-booking filtering.  
  ![Admin Dashboard Forecast](Screenshot (23).png)

- **Bookings Export (PDF)** for selected date ranges.

---

## 🧠 Machine Learning Forecasting

- Uses historical daily bookings for regression.  
- Ignores **zero-booking days** (to avoid bias).  
- Adds a **weekend uplift** (+25%) to reflect demand spikes.  
- Predictions are **clamped to non-negative** and never below current bookings.

---

## 🧱 Tech Stack

- **Backend**: Django 5, Django ORM, Custom User Model  
- **Database**: SQLite (default; can switch to PostgreSQL/MySQL)  
- **ML**: pandas, NumPy, scikit‑learn (LinearRegression)  
- **Frontend**: Django Templates + Bootstrap 5 + Chart.js  
- **PDF Export**: xhtml2pdf  

---

## 🚀 Local Setup

### 1. Create virtual environment
```bash
python -m venv .venv
# Activate
.\.venv\Scripts\activate   # Windows
source .venv/bin/activate     # macOS/Linux
```

### 2. Install dependencies
```bash
pip install -r railway/requirements.txt
```

### 3. Migrate database
```bash
cd railway
python manage.py migrate
python manage.py createsuperuser
```

### 4. Run server
```bash
python manage.py runserver
```
Visit: `http://127.0.0.1:8000/`  
Admin Dashboard: `http://127.0.0.1:8000/admin-dashboard/`

---

## 📸 Screenshots

Below are the main system views:

- Search Trains  
  ![Search Trains](Screenshot (17).png)

- Available Trains  
  ![Available Trains](Screenshot (18).png)

- Booking Form  
  ![Booking Form](Screenshot (19).png)

- Ticket Preview  
  ![Ticket Preview](Screenshot (20).png)

- My Bookings  
  ![My Bookings](Screenshot (21).png)

- Login Page  
  ![Login Page](Screenshot (24).png)

- Admin Dashboard (KPIs)  
  ![Admin Dashboard KPIs](Screenshot (22).png)

- Admin Dashboard (Forecast)  
  ![Admin Dashboard Forecast](Screenshot (23).png)

---

## 🔮 Future Improvements
- Switch DB to PostgreSQL for scalability.  
- Improve forecasting with ARIMA/Prophet models.  
- Add payment gateway integration.  
- Role-based access for staff vs super-admins.
