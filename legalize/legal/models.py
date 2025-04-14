# from django.db import models
from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()

class UploadHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file_name = models.CharField(max_length=255)
    file_path = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.file_name} uploaded by {self.user.username}"
