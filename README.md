# Railway Ticket Booking System (Django + ML)

A Railway Ticket Booking System built with **Django 5**, **SQLite**, and a lightweight **ML forecasting** pipeline using **pandas**, **NumPy**, and **scikit-learn**. The system supports train search, real-time availability, booking, and ticketing, with a **rich Admin Dashboard** that includes KPIs, trend charts, and a **7-day passenger forecast**.

---

## Features

### User Features
- **Search Trains** by source, destination, and date (up to 30 days advance)
- **Real-time Availability** with berth-wise capacity tracking
- **Multi-passenger Booking** (1-5 passengers) with individual details
- **Dynamic Fare Calculation** with GST, service charges, and convenience fees
- **Ticket Management** - Preview, confirm, and download tickets in PDF format
- **User Account System** with age validation (18+) and profile management
- **Booking History** with search and filtering options
- **Secure Authentication** with proper session management

### Admin Features
- **Analytics Dashboard** with comprehensive KPIs and business metrics
- **ML Forecasting** - 7-day passenger demand prediction using Linear Regression
- **Revenue Analytics** by route, top customers, and berth distribution
- **Data Export** in CSV and PDF formats
- **User Management** and system administration
- **Date-based Filtering** for analytics and exports

---

## Machine Learning Forecasting

The system implements intelligent forecasting that predicts passenger demand for the next 7 days:

- Uses historical daily bookings for regression analysis
- Ignores zero-booking days to avoid bias
- Applies weekend uplift (+25%) to reflect demand spikes
- Ensures predictions are non-negative and realistic
- Never predicts below current bookings

---

## Tech Stack

- **Backend**: Django 5, Django ORM, Custom User Model
- **Database**: SQLite (default; can switch to PostgreSQL/MySQL)
- **ML**: pandas, NumPy, scikit-learn (LinearRegression)
- **Frontend**: Django Templates + Bootstrap 5 + Chart.js
- **PDF Export**: xhtml2pdf

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup Steps

1. **Create virtual environment**
   ```bash
   python -m venv .venv
   # Activate
   .\.venv\Scripts\activate   # Windows
   source .venv/bin/activate     # macOS/Linux
   ```

2. **Install dependencies**
   ```bash
   pip install -r railway/requirements.txt
   ```

3. **Migrate database**
   ```bash
   cd railway
   python manage.py migrate
   python manage.py createsuperuser
   ```

4. **Run server**
   ```bash
   python manage.py runserver
   ```

5. **Access the application**
   - Main Application: http://127.0.0.1:8000/
   - Admin Dashboard: http://127.0.0.1:8000/admin-dashboard/

---

## Usage

### For Users
1. **Register** - Create account with valid details (18+ required)
2. **Search Trains** - Enter source, destination, and travel date
3. **Select Train** - Choose from available trains and berth types
4. **Add Passengers** - Enter passenger details (1-5 passengers)
5. **Review & Pay** - Check fare breakdown and complete booking
6. **Download Ticket** - Get ticket in PDF format

### For Administrators
1. **Login** - Use staff/admin credentials
2. **Dashboard** - View KPIs, charts, and analytics
3. **Forecasting** - Check 7-day demand predictions
4. **Export Data** - Download bookings in CSV/PDF format
5. **Manage System** - Add trains, routes, and manage availability

---

## Database Schema

### Core Models
- **User** - Custom user model with mobile, gender, DOB
- **Station** - Railway stations with codes
- **Train** - Train information and details
- **RouteStop** - Train routes with timing and distances
- **BerthType** - Different classes (SL, AC, etc.) with pricing
- **DailyTrainAvailability** - Real-time seat availability
- **Booking** - Booking records with status tracking
- **Passenger** - Individual passenger details
- **Payment** - Payment tracking

### Key Relationships
- User → Booking (One-to-Many)
- Train → RouteStop (One-to-Many)
- Booking → Passenger (One-to-Many)
- Train → DailyTrainAvailability (One-to-Many)

---

## API Endpoints

### User Endpoints
- `GET /` - Home page (redirects to search)
- `GET /search/` - Train search page
- `POST /search/` - Search trains
- `GET /book/<train_id>/` - Booking form
- `POST /book/<train_id>/` - Submit booking
- `GET /preview/<booking_id>/` - Booking preview
- `POST /mock-pay/<booking_id>/` - Mock payment
- `GET /ticket/<booking_id>/` - Ticket display
- `GET /my-bookings/` - User's bookings
- `GET /profile/` - User profile
- `POST /register/` - User registration

### Admin Endpoints
- `GET /admin-dashboard/` - Admin dashboard
- `GET /admin-dashboard/export-bookings-csv/` - Export CSV
- `GET /admin-dashboard/export-bookings-pdf/` - Export PDF

---

## Configuration

### Tax Configuration
```python
TAX_CONFIG = {
    'GST_PCT': 5,           # GST percentage
    'SERVICE_PCT': 2,       # Service charge percentage
    'CONVENIENCE_FEE': 10,  # Fixed convenience fee (INR)
}
```

### Booking Limits
- Maximum passengers per booking: 5
- Advance booking allowed: 30 days
- Age requirement: 18+ years

---

## Testing

### Running Tests
```bash
# Run all tests
python manage.py test

# Run specific app tests
python manage.py test bookings

# Run with coverage
coverage run --source='.' manage.py test
coverage report
```

---

## Future Improvements

- Switch database to PostgreSQL for scalability
- Improve forecasting with ARIMA/Prophet models
- Add payment gateway integration
- Implement role-based access for staff vs super-admins
- Add email notifications for bookings
- Create mobile application
- Implement real-time updates with WebSocket

---
