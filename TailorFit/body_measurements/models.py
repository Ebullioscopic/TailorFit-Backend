#from django.db import models

# Create your models here.
from django.db import models
from django.contrib.auth.models import User

class BodyMeasurement(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
    image = models.ImageField(upload_to='measurements/')
    processed_image = models.ImageField(upload_to='processed/', null=True)
    shoulder_width = models.FloatField(null=True)
    chest_circumference = models.FloatField(null=True)
    waist_circumference = models.FloatField(null=True)
    hip_circumference = models.FloatField(null=True)
    left_bicep_circumference = models.FloatField(null=True)
    right_bicep_circumference = models.FloatField(null=True)
    left_forearm_circumference = models.FloatField(null=True)
    right_forearm_circumference = models.FloatField(null=True)
    left_thigh_circumference = models.FloatField(null=True)
    right_thigh_circumference = models.FloatField(null=True)
    left_calf_circumference = models.FloatField(null=True)
    right_calf_circumference = models.FloatField(null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Measurement {self.id} - {self.created_at}"