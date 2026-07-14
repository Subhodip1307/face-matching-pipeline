import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceExtractor:
    def __init__(self, app: FaceAnalysis):
        """
        Args:
            app: An initialized InsightFace FaceAnalysis instance
                 (e.g. FaceAnalysis(name="buffalo_l"))
        """
        self.app = app
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    # ---------- image loading helpers ----------

    def _load_from_path(self, input_image_path: str) -> np.ndarray:
        img = cv2.imread(input_image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found or unreadable: {input_image_path}")
        return img

    def _load_from_bytes(self, byte_image) -> np.ndarray:
        # Rewind in case the stream was already read (common with Django UploadedFile)
        if hasattr(byte_image, "seek"):
            byte_image.seek(0)
        file_bytes = np.frombuffer(byte_image.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image from bytes")
        return img

    # ---------- counting ----------

    def count_faces(self, input_image_path: str) -> int:
        """Return the number of faces in an image file."""
        img = self._load_from_path(input_image_path)
        return len(self.app.get(img))

    def count_faces_bytes(self, byte_image) -> int:
        """Return the number of faces in an in-memory image (file-like object)."""
        img = self._load_from_bytes(byte_image)
        return len(self.app.get(img))

    # ---------- embeddings (what you need for pgvector) ----------

    def get_face_embeddings(
        self,
        source,
        min_face_size: int = 30,
        min_det_score: float = 0.5,
    ) -> dict:
        """
        Detect all faces and return a 512-dim embedding for each.

        Args:
            source: str path OR file-like object (e.g. request.FILES['image'])
            min_face_size: skip faces smaller than this (px) — usually noise
            min_det_score: skip low-confidence detections

        Returns:
            {
                'face_count': int,          # total faces detected
                'faces': [                  # one entry per usable face
                    {
                        'face_number': int,
                        'embedding': list[float],   # 512 floats, L2-normalized
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float,
                        'size': 'WxH',
                    },
                    ...
                ],
                'error': str | None,
            }
        """
        result = {'face_count': 0, 'faces': [], 'error': None}

        try:
            if isinstance(source, str):
                img = self._load_from_path(source)
            else:
                img = self._load_from_bytes(source)

            faces = self.app.get(img)
            result['face_count'] = len(faces)

            for i, face in enumerate(faces):
                x1, y1, x2, y2 = face.bbox.astype(int)
                w, h = x2 - x1, y2 - y1

                if w < min_face_size or h < min_face_size:
                    continue
                if face.det_score < min_det_score:
                    continue

                # normed_embedding is L2-normalized -> works directly with
                # cosine distance / inner product in pgvector
                embedding = face.normed_embedding
                if embedding is None:
                    # Fallback: normalize the raw embedding
                    raw = face.embedding
                    embedding = raw / np.linalg.norm(raw)

                result['faces'].append({
                    'face_number': i + 1,
                    'embedding': embedding.astype(np.float32).tolist(),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(face.det_score),
                    'size': f"{w}x{h}",
                })

        except Exception as e:
            result['error'] = str(e)

        return result