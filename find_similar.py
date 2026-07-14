from pgvector.django import CosineDistance
from .models import User,ImageEmbedding
# NOTE: We defined a custom User model above, so don't import Django's
# built-in one. Import your custom model (or use settings.AUTH_USER_MODEL)
# in your own project.

def find_matching_images(profile_id, threshold=0.4):
    """
    Find UploadImages containing a face similar to the user's profile face.

    threshold: max cosine distance (0 = identical, 2 = opposite).
               For InsightFace buffalo_l, ~0.3-0.5 is the typical
               "same person" range. Tune it on your own data.
    """
    profile = User.objects.get(pk=profile_id)

    if profile.embedding is None:
        print(f"Profile {profile_id} has no embedding yet")
        return []

    matches = (
        ImageEmbedding.objects
        .exclude(embedding__isnull=True)
        .annotate(distance=CosineDistance("embedding", profile.embedding))
        .filter(distance__lt=threshold).select_related("image")
        .order_by("distance")[:100]
    )

    image_ids = list(dict.fromkeys(m.image_id for m in matches)) 
    return image_ids #returning matched Images Ids