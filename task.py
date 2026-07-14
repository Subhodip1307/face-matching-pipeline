
from celery import shared_task
from .models import UploadImages,ImageEmbedding,User
from .SimpleFace import FaceExtractor,FaceAnalysis

_extractor = FaceExtractor(FaceAnalysis(name="buffalo_l"))


@shared_task
def user_profile_embedding(profile_id):
   """
   Generate and store the embedding for a user's profile picture.
   We count all detected faces on purpose:
   a second face in the photo — even a tiny background one 
   means we reject it rather than risk embedding the wrong person.
   """
    profile = User.objects.get(pk=profile_id)
    result = _extractor.get_face_embeddings(profile.profile_file.path)

    if result["error"]:
        print(f"Profile {profile_id}: {result['error']}")
        return "error"

    face_count = result["face_count"]

    if face_count == 1:
        profile.embedding = result["faces"][0]["embedding"]  # plain 512-float list
        profile.save(update_fields=["embedding"])
        return "ok"
    elif face_count == 0:
        print(f"Profile {profile_id}: no face detected")
        
    else:
        print(f"Profile {profile_id}: {face_count} faces detected, expected 1")
    
    profile.embedding = None
    profile.save(update_fields=["embedding"])
    return 


@shared_task
def bulk_image_embedding(image_ids: list):
    up_imgs = UploadImages.objects.filter(id__in=image_ids)
    rows = []

    for img in up_imgs:
        result = _extractor.get_face_embeddings(img.image.path)
        if result["error"]:
            print(f"Skipping image {img.id}: {result['error']}")
            continue

        print(f"Image {img.id}: {result['face_count']} faces detected")
        for face in result["faces"]:
            rows.append(ImageEmbedding(
                image=img,
                embedding=face["embedding"],   # plain list of 512 floats
            ))

    created = ImageEmbedding.objects.bulk_create(rows)
    return len(created)








