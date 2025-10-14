from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import boto3
from io import BytesIO
from .base import BaseStorage
from ..storage_utils import get_partitioned_path


class S3Storage(BaseStorage):
    """S3-based storage for cloud training"""

    def __init__(self, bucket: str, prefix: str = 'data/inputs', region: str = 'us-east-1'):
        """
        Initialize S3 storage.

        Args:
            bucket: S3 bucket name
            prefix: Prefix path within bucket
            region: AWS region
        """
        self.bucket = bucket
        self.prefix = prefix.rstrip('/')
        self.region = region
        self.s3_client = boto3.client('s3', region_name=region)

    def _get_s3_key(self, data_type: str, identifier: str) -> str:
        """
        Generate S3 key for data.

        Args:
            data_type: Type of data
            identifier: Unique identifier

        Returns:
            S3 key path
        """
        path = get_partitioned_path(data_type, identifier)
        return f"{self.prefix}/{data_type}/{path.name}"

    def save(
        self,
        data: Any,
        data_type: str,
        identifier: str,
        **kwargs
    ) -> None:
        """
        Save data to S3 as Parquet.

        Args:
            data: Data to save (dict or DataFrame)
            data_type: Type of data
            identifier: Unique identifier
        """
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data

        buffer = BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)

        s3_key = self._get_s3_key(data_type, identifier)
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=s3_key,
            Body=buffer.getvalue()
        )

    def load(
        self,
        data_type: str,
        filters: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Load data from S3 with optional filters.

        Args:
            data_type: Type of data to load
            filters: Optional filters (e.g., {'start_date': '20240101', 'end_date': '20240331'})

        Returns:
            DataFrame containing loaded data
        """
        start_date = filters.get('start_date') if filters else None
        end_date = filters.get('end_date') if filters else None

        s3_prefix = f"{self.prefix}/{data_type}/"

        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=s3_prefix
        )

        if 'Contents' not in response:
            return pd.DataFrame()

        keys = []
        for obj in response['Contents']:
            key = obj['Key']
            if not key.endswith('.parquet'):
                continue

            if start_date or end_date:
                file_name = Path(key).stem
                parts = file_name.split('_')
                if len(parts) >= 2:
                    file_date = parts[-1]
                    if start_date and file_date < start_date:
                        continue
                    if end_date and file_date > end_date:
                        continue

            keys.append(key)

        if not keys:
            return pd.DataFrame()

        dfs = []
        for key in keys:
            obj = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            buffer = BytesIO(obj['Body'].read())
            df = pd.read_parquet(buffer)
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)

    def exists(self, data_type: str, identifier: str) -> bool:
        """
        Check if data exists in S3.

        Args:
            data_type: Type of data
            identifier: Unique identifier

        Returns:
            True if data exists
        """
        s3_key = self._get_s3_key(data_type, identifier)
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except:
            return False

    def upload_directory(self, local_dir: str, s3_prefix: str) -> None:
        """
        Upload entire local directory to S3.

        Args:
            local_dir: Local directory path
            s3_prefix: S3 prefix for upload
        """
        local_path = Path(local_dir)
        for file_path in local_path.rglob('*.parquet'):
            relative_path = file_path.relative_to(local_path)
            s3_key = f"{self.prefix}/{s3_prefix}/{relative_path}".replace('\\', '/')
            self.s3_client.upload_file(str(file_path), self.bucket, s3_key)

    def download_directory(self, s3_prefix: str, local_dir: str) -> None:
        """
        Download entire S3 prefix to local directory.

        Args:
            s3_prefix: S3 prefix to download
            local_dir: Local directory path
        """
        full_prefix = f"{self.prefix}/{s3_prefix}/"
        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=full_prefix
        )

        if 'Contents' not in response:
            return

        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)

        for obj in response['Contents']:
            key = obj['Key']
            relative_key = key[len(full_prefix):]
            local_file = local_path / relative_key
            local_file.parent.mkdir(parents=True, exist_ok=True)
            self.s3_client.download_file(self.bucket, key, str(local_file))

    def save_model(self, model_path: str, s3_key: str) -> None:
        """
        Upload trained model to S3.

        Args:
            model_path: Local model file path
            s3_key: S3 key for model
        """
        self.s3_client.upload_file(model_path, self.bucket, s3_key)

    def load_model(self, s3_key: str, local_path: str) -> None:
        """
        Download model from S3.

        Args:
            s3_key: S3 key for model
            local_path: Local destination path
        """
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        self.s3_client.download_file(self.bucket, s3_key, local_path)
