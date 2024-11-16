#from django.db import models

# Create your models here.
from django.db import models

class TryOnRequest(models.Model):
    person_image = models.ImageField(upload_to='person_images/')
    garment_image = models.ImageField(upload_to='garment_images/')
    garment_description = models.TextField()
    result_image = models.ImageField(upload_to='result_images/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Try-on Request {self.id} - {self.created_at}"