from rest_framework import serializers
from .models import TryOnRequest

class TryOnRequestSerializer(serializers.ModelSerializer):
    class Meta:
        model = TryOnRequest
        fields = '__all__'

# virtual_tryon/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import TryOnViewSet

router = DefaultRouter()
router.register(r'tryon', TryOnViewSet)

urlpatterns = [
    path('', include(router.urls)),
]