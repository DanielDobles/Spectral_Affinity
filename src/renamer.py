import re

def clean_filename(filename):
    """
    Cleans audio filenames by removing common prefixes and UUID suffixes.
    
    Args:
        filename (str): Original filename (e.g., "Slavic-Song_Name-uuid.wav")
        
    Returns:
        str: Cleaned filename (e.g., "Song Name.wav")
    """
    name_body, ext = filename.rsplit('.', 1)
    
    # 1. Remove common prefixes
    prefixes = [
        r"^Slavic-",
        r"^Theme_OST-",
        r"^My_Workspace-",
        r"^audio-"
    ]
    
    for prefix in prefixes:
        name_body = re.sub(prefix, "", name_body, flags=re.IGNORECASE)
        
    # 2. Remove UUID suffixes (standard 8-4-4-4-12 hex chars)
    # Pattern: -[0-9a-f]{8}-[0-9a-f]{4}-...
    uuid_pattern = r"-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
    name_body = re.sub(uuid_pattern, "", name_body)
    
    # 3. Replace underscores with spaces
    name_body = name_body.replace("_", " ")
    
    # 4. Clean extra spaces
    name_body = re.sub(r"\s+", " ", name_body).strip()
    
    return f"{name_body}.{ext}"
