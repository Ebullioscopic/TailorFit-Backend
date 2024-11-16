from rest_framework import serializers
from .models import BodyMeasurement

class BodyMeasurementSerializer(serializers.ModelSerializer):
    class Meta:
        model = BodyMeasurement
        fields = '__all__'
        read_only_fields = ('processed_image', 'user')