import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError

def show_file_content(bucket_name, file_key):
    s3 = boto3.client('s3')
    try:
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        file_content = response['Body'].read().decode('utf-8')
        print(f"Content of {file_key}:\n")
        print(file_content)
    except ClientError as e:
        print(f"Error getting {file_key} from {bucket_name}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while reading {file_key}: {e}")

def list_files_in_bucket(bucket_name):
    s3 = boto3.client('s3')
    try:
        response = s3.list_objects_v2(Bucket=bucket_name)
        if 'Contents' in response:
            print(f"Files in {bucket_name}:")
            file_keys = [obj['Key'] for obj in response['Contents']]
            for key in file_keys:
                print(f"  - {key}")
            
            # Ask for a file name to view
            chosen_file = input("Enter the file name you want to view: ").strip()
            if chosen_file in file_keys:
                show_file_content(bucket_name, chosen_file)
            else:
                print("The file you entered does not exist in this bucket.")
        else:
            print(f"No files found in {bucket_name} or the bucket is empty.")
    except ClientError as e:
        print(f"Error listing objects in {bucket_name}: {e}")

def check_s3_connection():
    try:
        s3 = boto3.client('s3')
        response = s3.list_buckets()

        if 'Buckets' in response:
            print("Connected to S3! Buckets available:")
            buckets = [bucket['Name'] for bucket in response['Buckets']]
            for b in buckets:
                print(f"  - {b}")
            
            # Prompt user to choose a bucket
            if buckets:
                chosen_bucket = input("Enter the name of the bucket you want to list files from: ").strip()
                if chosen_bucket in buckets:
                    list_files_in_bucket(chosen_bucket)
                else:
                    print("The bucket you entered does not exist in your account.")
        else:
            print("Connected to S3 but no buckets found.")

    except NoCredentialsError:
        print("No AWS credentials found. Please configure them before running this code.")
    except PartialCredentialsError:
        print("Incomplete AWS credentials. Please provide both Access Key and Secret Key.")
    except ClientError as e:
        print(f"Client error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    check_s3_connection()
