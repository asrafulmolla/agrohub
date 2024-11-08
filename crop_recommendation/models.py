# models.py
from django.db import models

class CropPrediction(models.Model):
    nitrogen = models.FloatField()
    phosphorus = models.FloatField()
    potassium = models.FloatField()
    temperature = models.FloatField()
    humidity = models.FloatField()
    ph = models.FloatField()
    rainfall = models.FloatField()
    predicted_crop = models.CharField(max_length=100)
    timestamp = models.DateTimeField(auto_now_add=True)  

    def __str__(self):
        return f"{self.timestamp} - {self.predicted_crop}"
