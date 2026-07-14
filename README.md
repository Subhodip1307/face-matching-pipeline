# Face Matching Pipeline

Educational project demonstrating AI-powered face matching using **Django**, **InsightFace**, **PostgreSQL**, and **pgvector** — the same core technique used by applications like Google Photos and Immich. Faces are detected with InsightFace's `buffalo_l` model, converted into 512-dimensional embeddings, stored in PostgreSQL, and matched using pgvector's cosine similarity search.

📖 **Read the full article:** [How Google Photos Finds Your Face: Building It from Scratch with Django and pgvector](https://medium.com/@rogue-mind/how-google-photos-finds-your-face-building-it-from-scratch-with-django-and-pgvector-bb8a57f65ea8?sharedUserId=rogue-mind)

> **Note:** This repository is a reference implementation, not a plug-and-play project. To keep the focus on the face-matching pipeline, application-specific boilerplate (routing, serializers, views, project scaffolding) has been intentionally omitted. Adapt the snippets to your own project structure.

## FAQ

### Is this production-ready code?

No. This project is written for learning. Several production concerns are deliberately out of scope, for example:

- Re-running `bulk_image_embedding` on the same image (e.g. a Celery retry) creates duplicate embedding rows — a real application would make the task idempotent.
- Errors are logged with `print` statements instead of proper logging, retries, and monitoring.
- There is no automatic face clustering, the profile picture is used as the search reference to keep the pipeline easy to follow.

### Can I use this for personal or commercial purposes?

The code in this repository is free to use for both. However, the main heavy lifter here is **InsightFace's `buffalo_l` model**, which is provided for **non-commercial / educational use**. If you plan to use this in a commercial product, please check the [InsightFace license terms](https://github.com/deepinsight/insightface) and obtain the appropriate permissions or use a commercially licensed alternative.


## Implementation Notes

- **Docker volume path:** the compose file mounts `/var/lib/postgresql`, which is correct for the `pgvector/pgvector:pg18` image. If you use the `pg16` or `pg17` image instead, change the mount to `/var/lib/postgresql/data`.
- **Model loading:** `FaceExtractor` is created at module level on purpose, so each Celery worker loads the `buffalo_l` model once at startup instead of once per task. `ctx_id=0` uses GPU 0 if `onnxruntime-gpu` is installed and falls back to CPU otherwise.
- **Custom User model:** the project uses a custom `User(AbstractUser)` model. In your own project, set `AUTH_USER_MODEL` in settings before running the first migration, and reference it via `settings.AUTH_USER_MODEL` in foreign keys.
- **Match threshold:** for `buffalo_l`, a cosine-distance threshold between **0.3 and 0.5** is a good starting point — smaller is stricter, larger is more forgiving. Tune it on your own dataset.