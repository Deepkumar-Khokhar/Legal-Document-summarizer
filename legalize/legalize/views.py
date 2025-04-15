from django.http import HttpResponse
from django.shortcuts import render

def landing(request):
    #return  HttpResponse("Hello, world. you are at first_project Home page")
    return render(request,'website/index.html')
