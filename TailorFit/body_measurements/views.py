from django.shortcuts import render

# Create your views here.
from pydantic import ValidationError
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from django.core.files.base import ContentFile
import cv2
import numpy as np
from .models import BodyMeasurement
from .serializers import BodyMeasurementSerializer
from .utils import BodyMeasurementProcessor

class BodyMeasurementViewSet(viewsets.ModelViewSet):
    queryset = BodyMeasurement.objects.all()
    serializer_class = BodyMeasurementSerializer
    parser_classes = (MultiPartParser, FormParser)
    
    def perform_create(self, serializer):
        user = self.request.user if self.request.user.is_authenticated else None
        image = self.request.data.get('image')
        if not image:
            raise ValidationError("Image is required")
            
        # Save the initial instance with the original image
        instance = serializer.save(user=user)
        
        # Process the image
        processor = BodyMeasurementProcessor()
        measurements, processed_image = processor.process_image(instance.image.path)
        
        if measurements and processed_image is not None:
            # Convert processed image to file and save
            success, buffer = cv2.imencode('.jpg', processed_image)
            if success:
                processed_image_file = ContentFile(buffer.tobytes())
                instance.processed_image.save(f'processed_{image.name}', processed_image_file, save=False)
            
            # Update measurements
            for key, value in measurements.items():
                setattr(instance, key, value)
            
            instance.save()
            print(instance)
            return instance
        else:
            instance.delete()
            raise ValidationError("Could not process image. Please ensure the person is clearly visible.")
