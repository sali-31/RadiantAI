import os
from typing import Dict

import boto3
from botocore.exceptions import BotoCoreError, ClientError


def s3_bucket_name() -> str:
    return os.getenv("S3_BUCKET_NAME") or os.getenv("AWS_S3_BUCKET", "")


def s3_region() -> str:
    return os.getenv("AWS_REGION", "us-east-1")


def s3_prefix() -> str:
    return os.getenv("S3_UPLOAD_PREFIX", "uploads").strip("/")


def is_s3_configured() -> bool:
    return bool(s3_bucket_name())


def s3_client():
    return boto3.client("s3", region_name=s3_region())


def upload_image_to_s3(image_bytes: bytes, filename: str, content_type: str = "image/png") -> Dict[str, str]:
    bucket = s3_bucket_name()
    if not bucket:
        raise RuntimeError("S3_BUCKET_NAME is not configured")

    prefix = s3_prefix()
    key = f"{prefix}/{filename}" if prefix else filename

    try:
        s3_client().put_object(
            Bucket=bucket,
            Key=key,
            Body=image_bytes,
            ContentType=content_type,
            ServerSideEncryption="AES256",
        )
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"S3 upload failed: {exc}") from exc

    return {
        "provider": "s3",
        "bucket": bucket,
        "key": key,
        "url": presigned_s3_url(key),
    }


def presigned_s3_url(key: str) -> str:
    bucket = s3_bucket_name()
    if not bucket:
        return ""

    expires = int(os.getenv("S3_PRESIGNED_URL_EXPIRES_SECONDS", "3600"))
    try:
        return s3_client().generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires,
        )
    except (BotoCoreError, ClientError):
        return ""


def looks_like_s3_key(key: str) -> bool:
    prefix = s3_prefix()
    return bool(prefix and key.startswith(f"{prefix}/"))
