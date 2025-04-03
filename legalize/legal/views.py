from django.shortcuts import render


# Create your views here.
def landing(request):
    return render(request, 'landing.html')

# def login(request):
#     return render(request, 'login.html')

# def signup(request):
#     return render(request, 'signup.html')

from django.shortcuts import render
from django.contrib import messages
from django.contrib.auth import get_user_model

User = get_user_model()  # Get the custom user model if you are using one

import os
import torch
import pdfplumber
from docx import Document
from django.shortcuts import render, redirect
from django.core.files.storage import default_storage
from transformers import BartTokenizer, BartForConditionalGeneration

# Load BART model
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
device = torch.device('cpu')

# def summarize(request):
#     return render(request, 'summarize.html')

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    text_array = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_array.append(text.replace('\n', ' '))
    return ' '.join(text_array)

# Function to extract text from DOCX
def extract_text_from_docx(file_path):
    text_array = []
    doc = Document(file_path)
    for para in doc.paragraphs:
        text = para.text.replace('\n', ' ')
        text_array.append(text)
    return ' '.join(text_array)

# Function to summarize text using BART
def summarize_text(text):
    preprocessed_text = text.strip().replace('\n', '')

    # Prepare input for BART
    inputs = tokenizer.encode("summarize: " + preprocessed_text, return_tensors='pt', max_length=1024, truncation=True).to(device)
    summary_ids = model.generate(inputs, min_length=30, max_length=200, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Upload and summarize document
def summarize(request):
    if request.method == "POST":
        uploaded_file = request.FILES.get("document")
        if not uploaded_file:
            return render(request, "index.html", {"error": "Please upload a document."})

        file_path = default_storage.save(uploaded_file.name, uploaded_file)
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()

        # Extract text
        if file_ext == ".pdf":
            extracted_text = extract_text_from_pdf(file_path)
        elif file_ext == ".docx":
            extracted_text = extract_text_from_docx(file_path)
        else:
            extracted_text = "Unsupported file format."

        # Summarize text
        summary = summarize_text(extracted_text)
        word_count = len(extracted_text.split())

        # Redirect to summary page with extracted data
        return render(request, "summarize.html", {
            "extracted_text": extracted_text,
            "summary": summary,
            "word_count": word_count,
            "file_name": uploaded_file.name
        })

    return render(request, "index.html")
from django.contrib.auth.forms import AuthenticationForm

from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm,AuthenticationForm
from django.contrib.auth import login,logout
from .middlewares import auth, guest
# Create your views here.

@guest
def user_signup(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request,user)
            return redirect('home')
    else:
        initial_data = {'username':'', 'password1':'','password2':""}
        form = UserCreationForm(initial=initial_data)
    return render(request, 'signup.html',{'form':form})

@guest
def user_login(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request,user)
            return redirect('home')
    else:
        initial_data = {'username':'', 'password':''}
        form = AuthenticationForm(initial=initial_data)
    return render(request, 'login.html',{'form':form}) 

@auth
def home(request):
    return render(request, 'home.html')


def logout_view(request):
    logout(request)
    return redirect('login')
