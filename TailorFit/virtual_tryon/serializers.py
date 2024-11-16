from rest_framework import serializers
from .models import TryOnRequest

class TryOnRequestSerializer(serializers.ModelSerializer):
    class Meta:
        model = TryOnRequest
        fields = '__all__'
