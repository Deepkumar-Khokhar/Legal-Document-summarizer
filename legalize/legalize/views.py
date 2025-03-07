from django.http import HttpResponse
from django.shortcuts import render

def landing(request):
    #return  HttpResponse("Hello, world. you are at first_project Home page")
    return render(request,'website/index.html')

def about(request):
    #return  HttpResponse("Hello, world. you are at first_project about page")
    return render(request, 'website/about.html')

def contact(request):
    return  HttpResponse("Hello, world. you are at first_project contact page") 