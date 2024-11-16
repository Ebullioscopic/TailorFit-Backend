# In views.py
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import action
from django.core.files.base import ContentFile
import io
from PIL import Image

from .models import TryOnRequest
from .serializers import TryOnRequestSerializer
from .utils import setup_virtual_tryon

class TryOnViewSet(viewsets.ModelViewSet):
    queryset = TryOnRequest.objects.all()
    serializer_class = TryOnRequestSerializer
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tryon = setup_virtual_tryon()
    
    @action(detail=False, methods=['POST'])
    def try_on(self, request):
        # ... validation code ...
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            try:
                # Create TryOnRequest instance
                try_on_request = serializer.save()
                result_image, mask_image = self.tryon.process_images(
                    person_image=try_on_request.person_image,
                    garment_image=try_on_request.garment_image,
                    garment_description=try_on_request.garment_description,
                    auto_mask=True,  # or get from request
                    auto_crop=False  # or get from request
                )
                if result_image:
                    # Convert PIL image to bytes
                    img_io = io.BytesIO()
                    result_image.save(img_io, format='PNG')
                    img_content = ContentFile(img_io.getvalue())
                    
                    # Save to model
                    try_on_request.result_image.save(
                        f'result_{try_on_request.id}.png',
                        img_content
                    )
                    try_on_request.save()
                
                return Response(
                    self.serializer_class(try_on_request).data,
                    status=status.HTTP_200_OK
                )
        # ... save result code ...
            except Exception as e:
                return Response(
                    {'error': str(e)},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
        return Response(
            serializer.errors,
            status=status.HTTP_400_BAD_REQUEST
        )