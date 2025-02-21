from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from .models import CustomUser

class SignupForm(UserCreationForm):
    email = forms.EmailField(required=True)
    full_name = forms.CharField(max_length=255)

    class Meta:
        model = CustomUser
        fields = ['full_name', 'email', 'password1', 'password2']

class LoginForm(AuthenticationForm):
    username = forms.EmailField(label="Email", widget=forms.EmailInput(attrs={"class": "form-control"}))


def save(self, commit=True):
        user = super().save(commit=False)
        user.first_name = self.cleaned_data['full_name']  # Save full name as first_name
        if commit:
            user.save()
        return user