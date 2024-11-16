from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import TryOnViewSet

router = DefaultRouter()
router.register(r'tryon', TryOnViewSet)

urlpatterns = [
    path('', include(router.urls)),
]