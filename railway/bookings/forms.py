from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import User
from datetime import date

class RegisterForm(UserCreationForm):
    first_name = forms.CharField(max_length=150, required=False, label="Name")
    email = forms.EmailField(required=True)
    mobile = forms.CharField(required=True)
    dob = forms.DateField(widget=forms.DateInput(attrs={'type': 'date'}))
    gender = forms.ChoiceField(choices=User.GENDER_CHOICES, required=False)

    class Meta:
        model = User
        fields = ('username', 'first_name', 'email', 'mobile', 'gender', 'dob', 'password1', 'password2')

    def clean_mobile(self):
        mobile = self.cleaned_data['mobile']
        if not mobile.isdigit() or len(mobile) != 10:
            raise forms.ValidationError("Mobile number must be exactly 10 digits.")
        return mobile

    def clean_dob(self):
        dob = self.cleaned_data['dob']
        today = date.today()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        if age < 18:
            raise forms.ValidationError("You must be at least 18 years old to register.")
        return dob

    def save(self, commit=True):
        user = super().save(commit=False)
        # set first_name from the form
        user.first_name = self.cleaned_data.get('first_name', '') or ''
        user.email = self.cleaned_data.get('email', '') or ''
        user.mobile = self.cleaned_data.get('mobile', '') or ''
        user.gender = self.cleaned_data.get('gender', None)
        user.dob = self.cleaned_data.get('dob', None)
        if commit:
            user.save()
        return user

class PassengerForm(forms.Form):
    name = forms.CharField(max_length=255)
    age = forms.IntegerField(min_value=0)
    gender = forms.ChoiceField(choices=User.GENDER_CHOICES)
