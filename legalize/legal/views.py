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
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from .middlewares import auth, guest
from .models import UploadHistory

# Load BART model for summarization
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
device = torch.device("cpu")

User = get_user_model()


# Function to generate word cloud
def generate_wordcloud(text):
    media_path = settings.MEDIA_ROOT
    os.makedirs(media_path, exist_ok=True)
    wordcloud_path = os.path.join(media_path, "wordcloud.png")

    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    wordcloud.to_file(wordcloud_path)

    return settings.MEDIA_URL + "wordcloud.png"


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
    preprocessed_text = text.strip().replace("\n", "")

    inputs = tokenizer.encode("summarize: " + preprocessed_text, return_tensors="pt", max_length=1024, truncation=True).to(device)
    summary_ids = model.generate(inputs, min_length=30, max_length=200, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# Summarize view
def summarize(request):
    if request.method == "POST":
        uploaded_file = request.FILES.get("document")
        if not uploaded_file:
            return render(request, "index.html", {"error": "Please upload a document."})

        file_path = default_storage.save(uploaded_file.name, uploaded_file)
        file_path = os.path.join(settings.MEDIA_ROOT, file_path)
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()

        # Extract text
        if file_ext == ".pdf":
            extracted_text = extract_text_from_pdf(file_path)
        elif file_ext == ".docx":
            extracted_text = extract_text_from_docx(file_path)
        else:
            extracted_text = "Unsupported file format."

        # Summarize
        summary = summarize_text(extracted_text)
        word_count = len(extracted_text.split())
        wordcloud_url = generate_wordcloud(extracted_text)

        # Save summary and file name to session
        request.session["summary"] = summary
        request.session["extracted_text"] = extracted_text
        request.session["file_name"] = uploaded_file.name

        # Save history to DB
        if request.user.is_authenticated:
            UploadHistory.objects.create(
                user=request.user,
                file_name=uploaded_file.name,
                file_path=file_path
            )

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


# Download Summary as PDF view
def download_summary_pdf(request):
    summary = request.session.get("summary", "")
    file_name = request.session.get("file_name", "summary.pdf")

    if not summary:
        return HttpResponse("No summary available.", status=400)

    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    text_object = p.beginText(40, height - 50)
    text_object.setFont("Helvetica", 12)

    lines = summary.split("\n")
    for line in lines:
        for part in [line[i:i+90] for i in range(0, len(line), 90)]:
            text_object.textLine(part)

    p.drawText(text_object)
    p.showPage()
    p.save()

    buffer.seek(0)
    response = HttpResponse(buffer, content_type="application/pdf")
    response["Content-Disposition"] = f'attachment; filename="{file_name.split(".")[0]}_summary.pdf"'
    return response


# Landing page
def landing(request):
    return render(request, "landing.html")


# Signup view
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


# Login view
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


# Logout view
def logout_view(request):
    logout(request)
    return redirect("landing")


# Home view
@auth
def home(request):
    return render(request, "home.html")


# About page
def about(request):
    return render(request, "about.html")


# History view
@auth
def history(request):
    uploads = UploadHistory.objects.filter(user=request.user).order_by("-uploaded_at")
    return render(request, "history.html", {"uploads": uploads})
