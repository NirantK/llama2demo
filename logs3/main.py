import os
import datetime

import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, Depends, FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

app = FastAPI()
security = HTTPBearer()


class LogItem(BaseModel):
    log: str


# Replace with your S3 credentials and bucket name
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_KEY")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "llamachatlogs")
SECRET_TOKEN = os.environ.get("SECRET_TOKEN")

s3_client = boto3.client(
    "s3", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY
)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if token != SECRET_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    return True


@app.post("/log")
async def log_to_s3(log_item: LogItem, authenticated: bool = Depends(verify_token)):
    # Request body should be in the format {"log": "A new log line from llama-chat"}
    log_line = log_item.log

    if not log_line:
        return {"message": "No log provided."}

    # Get the current date in the format YYYY-MM-DD
    current_date = datetime.date.today().isoformat()

    # Construct the log file name with the current date
    S3_FILE_NAME = f"llamalogs_{current_date}.txt"

    # Read the existing file from S3
    try:
        logger.info("Fetched existing log from S3.")
        existing_log = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=S3_FILE_NAME)
        existing_log_contents = existing_log["Body"].read().decode("utf-8")
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            logger.info("No existing log found in S3.")
            existing_log_contents = ""

    # Concatenate the new log with the existing content
    updated_log_contents = existing_log_contents + log_line + "\n"

    # Replace the previous file with the new file to S3
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=S3_FILE_NAME,
            Body=updated_log_contents.encode("utf-8"),
            ACL="bucket-owner-full-control",
        )
        logger.info(f"Log appended to S3: {log_line}")
        return {"message": "Log appended to S3 successfully."}
    except Exception as e:
        logger.error(f"Error appending log to S3: {e}")
        return {"message": "Error appending log to S3.", "error": str(e)}
