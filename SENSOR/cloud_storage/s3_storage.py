"""
S3 Storage Module
Handles AWS S3 operations for model and data storage
"""

import os
import sys
import pickle
import json
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import io

import boto3
from botocore.exceptions import ClientError, ConnectionError, Timeout

from SENSOR.exception import SensorException
from SENSOR.constant.s3_bucket import (
    S3_BUCKET_NAME,
    S3_REGION,
    S3_MAX_RETRIES,
    S3_UPLOAD_TIMEOUT,
    S3_DOWNLOAD_TIMEOUT,
    S3_ACL,
    S3_SERVER_SIDE_ENCRYPTION,
    S3_STORAGE_CLASS,
)
from SENSOR.constant.env_variable import env_var


class S3Storage:
    """
    AWS S3 Storage Handler
    Manages upload, download, and deletion of files in S3
    """

    def __init__(
        self,
        bucket_name: str = S3_BUCKET_NAME,
        region_name: str = S3_REGION,
        access_key: str = env_var.AWS_ACCESS_KEY_ID,
        secret_key: str = env_var.AWS_SECRET_ACCESS_KEY
    ):
        """
        Initialize S3Storage with AWS credentials
        
        Args:
            bucket_name: S3 bucket name
            region_name: AWS region
            access_key: AWS access key ID
            secret_key: AWS secret access key
        """
        try:
            self.bucket_name = bucket_name
            self.region_name = region_name
            self.logger = logging.getLogger(__name__)
            
            # Create S3 client
            self.s3_client = boto3.client(
                's3',
                region_name=region_name,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                config=boto3.session.Config(
                    retries={'max_attempts': S3_MAX_RETRIES},
                    connect_timeout=S3_UPLOAD_TIMEOUT,
                    read_timeout=S3_DOWNLOAD_TIMEOUT,
                )
            )
            
            # Verify bucket exists
            self._verify_bucket_exists()
            self.logger.info(f"S3Storage initialized with bucket: {bucket_name}")
            
        except Exception as e:
            raise SensorException(e, sys)

    def _verify_bucket_exists(self) -> bool:
        """Verify that the S3 bucket exists and is accessible"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            self.logger.info(f"Verified access to S3 bucket: {self.bucket_name}")
            return True
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                raise SensorException(f"S3 bucket not found: {self.bucket_name}", sys)
            elif error_code == '403':
                raise SensorException(f"Access denied to S3 bucket: {self.bucket_name}", sys)
            else:
                raise SensorException(f"Error accessing S3 bucket: {str(e)}", sys)

    def upload_file(
        self,
        file_path: str,
        s3_key: str,
        extra_args: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Upload a file to S3
        
        Args:
            file_path: Local file path
            s3_key: S3 object key (path in bucket)
            extra_args: Additional arguments for upload
            
        Returns:
            bool: True if upload successful
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Default extra arguments
            if extra_args is None:
                extra_args = {
                    'ACL': S3_ACL,
                    'ServerSideEncryption': S3_SERVER_SIDE_ENCRYPTION,
                    'StorageClass': S3_STORAGE_CLASS,
                }
            
            file_size = os.path.getsize(file_path)
            self.logger.info(f"Uploading file: {file_path} to s3://{self.bucket_name}/{s3_key} (Size: {file_size} bytes)")
            
            # Upload file
            self.s3_client.upload_file(
                file_path,
                self.bucket_name,
                s3_key,
                ExtraArgs=extra_args,
            )
            
            self.logger.info(f"Successfully uploaded: s3://{self.bucket_name}/{s3_key}")
            return True
            
        except FileNotFoundError as e:
            self.logger.error(f"File not found: {str(e)}")
            raise SensorException(e, sys)
        except ClientError as e:
            self.logger.error(f"AWS error during upload: {str(e)}")
            raise SensorException(e, sys)
        except Exception as e:
            self.logger.error(f"Error uploading file: {str(e)}")
            raise SensorException(e, sys)

    def download_file(
        self,
        s3_key: str,
        file_path: str
    ) -> bool:
        """
        Download a file from S3
        
        Args:
            s3_key: S3 object key
            file_path: Local file path to save to
            
        Returns:
            bool: True if download successful
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            self.logger.info(f"Downloading file from s3://{self.bucket_name}/{s3_key} to {file_path}")
            
            # Download file
            self.s3_client.download_file(
                self.bucket_name,
                s3_key,
                file_path
            )
            
            file_size = os.path.getsize(file_path)
            self.logger.info(f"Successfully downloaded: {file_path} (Size: {file_size} bytes)")
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                self.logger.error(f"File not found in S3: {s3_key}")
            else:
                self.logger.error(f"AWS error during download: {str(e)}")
            raise SensorException(e, sys)
        except Exception as e:
            self.logger.error(f"Error downloading file: {str(e)}")
            raise SensorException(e, sys)

    def upload_object(
        self,
        python_object: Any,
        s3_key: str,
        object_type: str = 'pickle'
    ) -> bool:
        """
        Upload a Python object (model, dict, etc.) to S3
        
        Args:
            python_object: Python object to upload
            s3_key: S3 object key
            object_type: Type of serialization ('pickle' or 'json')
            
        Returns:
            bool: True if upload successful
        """
        try:
            self.logger.info(f"Uploading {object_type} object to s3://{self.bucket_name}/{s3_key}")
            
            # Serialize object
            if object_type == 'pickle':
                object_bytes = pickle.dumps(python_object)
                content_type = 'application/octet-stream'
            elif object_type == 'json':
                object_bytes = json.dumps(python_object, indent=4).encode('utf-8')
                content_type = 'application/json'
            else:
                raise ValueError(f"Unsupported object type: {object_type}")
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=object_bytes,
                ContentType=content_type,
                ServerSideEncryption=S3_SERVER_SIDE_ENCRYPTION,
                StorageClass=S3_STORAGE_CLASS,
                ACL=S3_ACL,
            )
            
            self.logger.info(f"Successfully uploaded {object_type} object: s3://{self.bucket_name}/{s3_key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error uploading {object_type} object: {str(e)}")
            raise SensorException(e, sys)

    def download_object(
        self,
        s3_key: str,
        object_type: str = 'pickle'
    ) -> Any:
        """
        Download a Python object (model, dict, etc.) from S3
        
        Args:
            s3_key: S3 object key
            object_type: Type of deserialization ('pickle' or 'json')
            
        Returns:
            Any: Deserialized Python object
        """
        try:
            self.logger.info(f"Downloading {object_type} object from s3://{self.bucket_name}/{s3_key}")
            
            # Download object
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            
            object_bytes = response['Body'].read()
            
            # Deserialize object
            if object_type == 'pickle':
                python_object = pickle.loads(object_bytes)
            elif object_type == 'json':
                python_object = json.loads(object_bytes.decode('utf-8'))
            else:
                raise ValueError(f"Unsupported object type: {object_type}")
            
            self.logger.info(f"Successfully downloaded {object_type} object: s3://{self.bucket_name}/{s3_key}")
            return python_object
            
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                self.logger.error(f"Object not found in S3: {s3_key}")
            else:
                self.logger.error(f"AWS error during download: {str(e)}")
            raise SensorException(e, sys)
        except Exception as e:
            self.logger.error(f"Error downloading {object_type} object: {str(e)}")
            raise SensorException(e, sys)

    def delete_file(self, s3_key: str) -> bool:
        """
        Delete a file from S3
        
        Args:
            s3_key: S3 object key
            
        Returns:
            bool: True if delete successful
        """
        try:
            self.logger.info(f"Deleting file from s3://{self.bucket_name}/{s3_key}")
            
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            
            self.logger.info(f"Successfully deleted: s3://{self.bucket_name}/{s3_key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting file: {str(e)}")
            raise SensorException(e, sys)

    def list_files(self, prefix: str = "", max_results: int = 1000) -> List[Dict[str, Any]]:
        """
        List files in S3 bucket with given prefix
        
        Args:
            prefix: S3 prefix to filter files
            max_results: Maximum number of results
            
        Returns:
            List[Dict]: List of file metadata
        """
        try:
            self.logger.info(f"Listing files in s3://{self.bucket_name}/{prefix}")
            
            files = []
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=prefix,
                PaginationConfig={'PageSize': min(max_results, 1000)}
            )
            
            for page in pages:
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    files.append({
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'modified': obj['LastModified'].isoformat(),
                        'storage_class': obj['StorageClass']
                    })
                
                if len(files) >= max_results:
                    break
            
            self.logger.info(f"Found {len(files)} files with prefix: {prefix}")
            return files[:max_results]
            
        except Exception as e:
            self.logger.error(f"Error listing files: {str(e)}")
            raise SensorException(e, sys)

    def file_exists(self, s3_key: str) -> bool:
        """
        Check if a file exists in S3
        
        Args:
            s3_key: S3 object key
            
        Returns:
            bool: True if file exists
        """
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            self.logger.info(f"File exists: s3://{self.bucket_name}/{s3_key}")
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                self.logger.warning(f"File not found: s3://{self.bucket_name}/{s3_key}")
                return False
            else:
                raise SensorException(e, sys)

    def get_file_size(self, s3_key: str) -> int:
        """
        Get file size in S3
        
        Args:
            s3_key: S3 object key
            
        Returns:
            int: File size in bytes
        """
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            file_size = response['ContentLength']
            self.logger.info(f"File size: {s3_key} = {file_size} bytes")
            return file_size
        except Exception as e:
            self.logger.error(f"Error getting file size: {str(e)}")
            raise SensorException(e, sys)

    def copy_file(self, source_key: str, destination_key: str) -> bool:
        """
        Copy a file within S3
        
        Args:
            source_key: Source S3 object key
            destination_key: Destination S3 object key
            
        Returns:
            bool: True if copy successful
        """
        try:
            self.logger.info(f"Copying s3://{self.bucket_name}/{source_key} to s3://{self.bucket_name}/{destination_key}")
            
            copy_source = {'Bucket': self.bucket_name, 'Key': source_key}
            
            self.s3_client.copy_object(
                CopySource=copy_source,
                Bucket=self.bucket_name,
                Key=destination_key,
                ServerSideEncryption=S3_SERVER_SIDE_ENCRYPTION,
                StorageClass=S3_STORAGE_CLASS,
                ACL=S3_ACL,
            )
            
            self.logger.info(f"Successfully copied to: s3://{self.bucket_name}/{destination_key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error copying file: {str(e)}")
            raise SensorException(e, sys)

    def sync_folder(self, local_folder: str, s3_prefix: str) -> int:
        """
        Sync all files from local folder to S3
        
        Args:
            local_folder: Local folder path
            s3_prefix: S3 prefix for destination
            
        Returns:
            int: Number of files uploaded
        """
        try:
            if not os.path.exists(local_folder):
                raise FileNotFoundError(f"Folder not found: {local_folder}")
            
            uploaded_count = 0
            
            for root, dirs, files in os.walk(local_folder):
                for file in files:
                    local_file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_file_path, local_folder)
                    s3_key = f"{s3_prefix}/{relative_path}".replace("\\", "/")
                    
                    self.upload_file(local_file_path, s3_key)
                    uploaded_count += 1
            
            self.logger.info(f"Synced {uploaded_count} files from {local_folder} to s3://{self.bucket_name}/{s3_prefix}")
            return uploaded_count
            
        except Exception as e:
            self.logger.error(f"Error syncing folder: {str(e)}")
            raise SensorException(e, sys)

    def get_file_url(self, s3_key: str, expiration: int = 3600) -> str:
        """
        Generate a presigned URL for an S3 object
        
        Args:
            s3_key: S3 object key
            expiration: URL expiration time in seconds (default: 1 hour)
            
        Returns:
            str: Presigned URL
        """
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            
            self.logger.info(f"Generated presigned URL for: {s3_key} (expires in {expiration}s)")
            return url
            
        except Exception as e:
            self.logger.error(f"Error generating presigned URL: {str(e)}")
            raise SensorException(e, sys)
