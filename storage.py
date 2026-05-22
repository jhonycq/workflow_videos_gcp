"""
Google Cloud Storage utilities for the video generation pipeline.

Handles upload, download, and listing of assets in GCS.
"""
import logging
from pathlib import Path
from typing import List, Optional
import json

from google.cloud import storage

from config import BUCKET_NAME, PROJECT_ID, get_gcs_scene_path, get_gcs_manifest_path

logger = logging.getLogger(__name__)

# Global client instance (lazy initialized)
_storage_client: Optional[storage.Client] = None


def get_client() -> storage.Client:
    """Get or create the GCS client."""
    global _storage_client
    if _storage_client is None:
        _storage_client = storage.Client(project=PROJECT_ID)
    return _storage_client


def get_bucket(bucket_name: str = BUCKET_NAME) -> storage.Bucket:
    """Get a bucket reference."""
    client = get_client()
    return client.bucket(bucket_name)


def upload_bytes(
    data: bytes,
    gcs_path: str,
    bucket_name: str = BUCKET_NAME,
    content_type: str = "application/octet-stream"
) -> str:
    """
    Upload bytes directly to GCS.
    
    Args:
        data: Bytes to upload
        gcs_path: Path within the bucket (without gs:// prefix)
        bucket_name: GCS bucket name
        content_type: MIME type of the content
        
    Returns:
        Full GCS URI (gs://bucket/path)
    """
    logger.info(f"Uploading {len(data)} bytes to gs://{bucket_name}/{gcs_path}")
    
    bucket = get_bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    
    blob.upload_from_string(data, content_type=content_type)
    
    gcs_uri = f"gs://{bucket_name}/{gcs_path}"
    logger.info(f"Upload complete: {gcs_uri}")
    
    return gcs_uri


def upload_file(
    local_path: Path,
    gcs_path: str,
    bucket_name: str = BUCKET_NAME
) -> str:
    """
    Upload a local file to GCS.
    
    Args:
        local_path: Local file path
        gcs_path: Path within the bucket
        bucket_name: GCS bucket name
        
    Returns:
        Full GCS URI
    """
    local_path = Path(local_path)
    logger.info(f"Uploading {local_path} to gs://{bucket_name}/{gcs_path}")
    
    bucket = get_bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    
    blob.upload_from_filename(str(local_path))
    
    gcs_uri = f"gs://{bucket_name}/{gcs_path}"
    logger.info(f"Upload complete: {gcs_uri}")
    
    return gcs_uri


def download_file(
    gcs_path: str,
    local_path: Path,
    bucket_name: str = BUCKET_NAME
) -> Path:
    """
    Download a file from GCS to local filesystem.
    
    Args:
        gcs_path: Path within the bucket (can be full gs:// URI or just path)
        local_path: Local destination path
        bucket_name: GCS bucket name (used if gcs_path is not a full URI)
        
    Returns:
        Local path where file was saved
    """
    # Handle full GCS URI
    if gcs_path.startswith("gs://"):
        parts = gcs_path[5:].split("/", 1)
        bucket_name = parts[0]
        gcs_path = parts[1] if len(parts) > 1 else ""
    
    local_path = Path(local_path)
    logger.info(f"Downloading gs://{bucket_name}/{gcs_path} to {local_path}")
    
    # Create parent directories
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    bucket = get_bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    
    blob.download_to_filename(str(local_path))
    
    logger.info(f"Download complete: {local_path}")
    return local_path


def download_bytes(
    gcs_path: str,
    bucket_name: str = BUCKET_NAME
) -> bytes:
    """
    Download file content as bytes from GCS.
    
    Args:
        gcs_path: Path within the bucket
        bucket_name: GCS bucket name
        
    Returns:
        File content as bytes
    """
    # Handle full GCS URI
    if gcs_path.startswith("gs://"):
        parts = gcs_path[5:].split("/", 1)
        bucket_name = parts[0]
        gcs_path = parts[1] if len(parts) > 1 else ""
    
    logger.info(f"Downloading bytes from gs://{bucket_name}/{gcs_path}")
    
    bucket = get_bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    
    return blob.download_as_bytes()


def list_scene_videos(
    run_id: str,
    bucket_name: str = BUCKET_NAME
) -> List[str]:
    """
    List all video files for a given run.
    
    Args:
        run_id: The run identifier
        bucket_name: GCS bucket name
        
    Returns:
        List of GCS URIs for video files, sorted by scene number
    """
    prefix = f"mvp/{run_id}/scenes/"
    logger.info(f"Listing videos in gs://{bucket_name}/{prefix}")
    
    bucket = get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    
    video_uris = []
    for blob in blobs:
        if blob.name.endswith(".mp4"):
            video_uris.append(f"gs://{bucket_name}/{blob.name}")
    
    # Sort by scene number
    video_uris.sort()
    
    logger.info(f"Found {len(video_uris)} videos")
    return video_uris


def upload_manifest(
    manifest_dict: dict,
    run_id: str,
    bucket_name: str = BUCKET_NAME
) -> str:
    """
    Upload the manifest JSON to GCS.
    
    Args:
        manifest_dict: Manifest as dictionary
        run_id: The run identifier
        bucket_name: GCS bucket name
        
    Returns:
        GCS URI of the manifest
    """
    gcs_path = get_gcs_manifest_path(run_id)
    manifest_json = json.dumps(manifest_dict, indent=2, ensure_ascii=False)
    
    return upload_bytes(
        data=manifest_json.encode('utf-8'),
        gcs_path=gcs_path,
        bucket_name=bucket_name,
        content_type="application/json"
    )


def download_manifest(
    run_id: str,
    bucket_name: str = BUCKET_NAME
) -> dict:
    """
    Download and parse the manifest JSON from GCS.
    
    Args:
        run_id: The run identifier
        bucket_name: GCS bucket name
        
    Returns:
        Manifest as dictionary
    """
    gcs_path = get_gcs_manifest_path(run_id)
    data = download_bytes(gcs_path, bucket_name)
    return json.loads(data.decode('utf-8'))


def upload_scene_image(
    image_bytes: bytes,
    scene_idx: int,
    run_id: str,
    bucket_name: str = BUCKET_NAME
) -> str:
    """
    Upload a scene's image to GCS.
    
    Args:
        image_bytes: Image data
        scene_idx: Scene index
        run_id: Run identifier
        bucket_name: GCS bucket name
        
    Returns:
        GCS URI of the uploaded image
    """
    gcs_path = f"{get_gcs_scene_path(run_id, scene_idx)}/image.png"
    
    return upload_bytes(
        data=image_bytes,
        gcs_path=gcs_path,
        bucket_name=bucket_name,
        content_type="image/png"
    )


def check_file_exists(
    gcs_path: str,
    bucket_name: str = BUCKET_NAME
) -> bool:
    """
    Check if a file exists in GCS.
    
    Args:
        gcs_path: Path within the bucket
        bucket_name: GCS bucket name
        
    Returns:
        True if file exists
    """
    # Handle full GCS URI
    if gcs_path.startswith("gs://"):
        parts = gcs_path[5:].split("/", 1)
        bucket_name = parts[0]
        gcs_path = parts[1] if len(parts) > 1 else ""
    
    bucket = get_bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    
    return blob.exists()


if __name__ == "__main__":
    # Test storage module
    logging.basicConfig(level=logging.INFO)
    
    print("Storage module loaded successfully.")
    print(f"Default bucket: {BUCKET_NAME}")
    print("\nAvailable functions:")
    print("  - upload_bytes(data, gcs_path)")
    print("  - upload_file(local_path, gcs_path)")
    print("  - download_file(gcs_path, local_path)")
    print("  - list_scene_videos(run_id)")
    print("  - upload_manifest(manifest_dict, run_id)")
