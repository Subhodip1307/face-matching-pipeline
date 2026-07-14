from django.db import models
# NOTE: We defined a custom User model above, so don't import Django's
# built-in one. Import your custom model (or use settings.AUTH_USER_MODEL)
# in your own project.
from django.contrib.auth.models import User
from pgvector.django import VectorField
from pgvector.django import HnswIndex


class UploadImages(models.Model):
    # Meta data of an uploaded Image
    image = models.FileField(upload_to='upload_images/%Y/%m/%d/', null=True, blank=True)
    created_by = models.ForeignKey(User, related_name="user_upload_img", on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

class ImageEmbedding(models.Model):
    image=models.ForeignKey(UploadImages,on_delete=models.CASCADE)
    embedding=VectorField(dimensions=512,null=True,blank=True)
    class Meta:
        indexes = [
            HnswIndex(
                name="imageembedding_hnsw_idx",
                fields=["embedding"],
                m=16,
                ef_construction=64,
                opclasses=["vector_cosine_ops"],
            )
        ]


