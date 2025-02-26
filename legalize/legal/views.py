from django.shortcuts import render


# Create your views here.
def home(request):
    return render(request, 'home.html')

# def login(request):
#     return render(request, 'login.html')

# def signup(request):
#     return render(request, 'signup.html')

from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from .forms import SignupForm, LoginForm
from django.contrib import messages
from django.contrib.auth import login, logout, authenticate, get_user_model

User = get_user_model()  # Get the custom user model if you are using one

def signup(request):
    if request.method == 'POST':
        form = SignupForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')

            # ✅ Check if the username already exists
            if User.objects.filter(username=username).exists():
                messages.error(request, "Username already taken. Please choose a different one.")
            else:
                user = form.save()
                request.login(user)  # ✅ Fix for Django 5.1
                messages.success(request, "Signup successful!")
                return redirect('home')  # Redirect after signup
        else:
            messages.error(request, "Error in form submission. Please check the details.")
    else:
        form = SignupForm()
    
    return render(request, 'signup.html', {'form': form})



def login(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')

        user = authenticate(request, username=email, password=password)  # Change username to email

        if user is not None:
            login(request, user)
            messages.success(request, "Login successful!")
            return redirect('home')
        else:
            messages.error(request, "Invalid email or password.")

    return render(request, 'login.html')

def logout_view(request):
    logout(request)
    messages.success(request, "You have been logged out.")
    return redirect('login')