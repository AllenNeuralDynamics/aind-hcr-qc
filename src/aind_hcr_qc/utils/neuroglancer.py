import boto3
import json
from IPython.display import display, HTML

def get_ng_link_from_s3(bucket_name, prefix, ng_type: str = "spots") -> str:
    """
    Download multichannel_spot_annotation_ng_link.json from S3 and extract ng_link.
    
    Args:
        bucket_name (str): S3 bucket name (e.g., "aind-open-data")
        prefix (str): S3 prefix/folder path (e.g., "HCR_767018_2025-08-20_14-00-00_processed_2025-09-10_00-33-24")
        ng_type (str): Type of neuroglancer link, default is "spots"
    
    Returns:
        str: The ng_link URL if found, None otherwise
    """
    s3_client = boto3.client('s3')

    if ng_type != "spots":
        raise ValueError("Currently, only 'spots' ng_type is supported.")

    if "_processed_" in prefix:
        raw_folder = prefix.split("_processed_")[0]
    # map of type to filename
    type_file_map = {
        "spots": "multichannel_spot_annotation_ng_link.json",

        # proteomics pipeline ngs are stored in prefix/prefix
        "camera_aligned": f"{prefix}/camera_aligned_neuroglancer.json",
        "radial_correction": f"{prefix}/radially_corrected_neuroglancer.json",
        "raw": f"{raw_folder}/raw_neuroglancer.json", # stored in (raw prefix)/raw_neuroglancer.json

        # Future types can be added here
    }

    s3_key = f"{prefix}/{type_file_map[ng_type]}"
    
    try:
        # Download the file
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        file_content = response['Body'].read().decode('utf-8')
        
        data = json.loads(file_content)
        ng_link = data.get('ng_link')
        
        if ng_link:
            print(f"Found ng_link for {prefix}:")
            # Display as clickable hyperlink
            display(HTML(f'<a href="{ng_link}" target="_blank">{ng_link}</a>'))
            return ng_link
        else:
            print(f"No 'ng_link' key found in the JSON file for {prefix}")
            return None
            
    except s3_client.exceptions.NoSuchKey:
        print(f"File multichannel_spot_annotation_ng_link.json not found for prefix: {prefix}")
        return None
    except Exception as e:
        print(f"Error downloading or parsing file for {prefix}: {str(e)}")
        return None