import os
import torch
import pdfplumber
import matplotlib.pyplot as plt
from docx import Document
from io import BytesIO
from django.shortcuts import render, redirect
from django.core.files.storage import default_storage
from django.contrib.auth import login, logout, get_user_model
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.conf import settings
from django.http import HttpResponse
from transformers import BartTokenizer, BartForConditionalGeneration
from wordcloud import WordCloud
from .middlewares import auth, guest

# Load BART model for summarization
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
device = torch.device("cpu")

User = get_user_model()  # Get the custom user model if using one


# Function to generate word cloud
def generate_wordcloud(text):
    """Generates a word cloud image and saves it to the media directory."""
    media_path = settings.MEDIA_ROOT
    os.makedirs(media_path, exist_ok=True)  # Ensure media directory exists
    wordcloud_path = os.path.join(media_path, "wordcloud.png")

    # Generate and save the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    wordcloud.to_file(wordcloud_path)

    return settings.MEDIA_URL + "wordcloud.png"  # Return the URL


# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    text_array = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_array.append(text.replace("\n", " "))
    return " ".join(text_array)


# Function to extract text from DOCX
def extract_text_from_docx(file_path):
    text_array = []
    doc = Document(file_path)
    for para in doc.paragraphs:
        text_array.append(para.text.replace("\n", " "))
    return " ".join(text_array)


# Function to summarize text using BART
def summarize_text(text):
    """Summarizes the extracted text using BART."""
    preprocessed_text = text.strip().replace("\n", "")

    inputs = tokenizer.encode("summarize: " + preprocessed_text, return_tensors="pt", max_length=1024, truncation=True).to(device)
    summary_ids = model.generate(inputs, min_length=30, max_length=200, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# Function to handle document upload and summarization
def summarize(request):
    if request.method == "POST":
        uploaded_file = request.FILES.get("document")
        if not uploaded_file:
            return render(request, "index.html", {"error": "Please upload a document."})

        file_path = default_storage.save(uploaded_file.name, uploaded_file)
        file_path = os.path.join(settings.MEDIA_ROOT, file_path)  # Ensure absolute path
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()

        # Extract text from file
        if file_ext == ".pdf":
            extracted_text = extract_text_from_pdf(file_path)
        elif file_ext == ".docx":
            extracted_text = extract_text_from_docx(file_path)
        else:
            extracted_text = "Unsupported file format."

        # Summarize extracted text
        summary = summarize_text(extracted_text)
        word_count = len(extracted_text.split())

        # Generate Word Cloud
        wordcloud_url = generate_wordcloud(extracted_text)

        # Save summary to session for later use
        request.session["summary"] = summary
        request.session["extracted_text"] = extracted_text

        return render(
            request,
            "summarize.html",
            {
                "extracted_text": extracted_text,
                "summary": summary,
                "word_count": word_count,
                "file_name": uploaded_file.name,
                "wordcloud_url": wordcloud_url,
            },
        )

    return render(request, "index.html")


# Landing Page View
def landing(request):
    return render(request, "landing.html")


# User Signup View
@guest
def user_signup(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect("home")
    else:
        form = UserCreationForm()
    return render(request, "signup.html", {"form": form})


# User Login View
@guest
def user_login(request):
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect("home")
    else:
        form = AuthenticationForm()
    return render(request, "login.html", {"form": form})


# Home View (Requires Authentication)
@auth
def home(request):
    return render(request, "home.html")


# User Logout View
def logout_view(request):
    logout(request)
    return redirect("login")
