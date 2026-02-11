import re

def clean_filename(filename):
    if '.' not in filename: return filename
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
    uuid_pattern = r"-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
    name_body = re.sub(uuid_pattern, "", name_body)
    
    # 3. Handle specific patterns like Slavic-Untitled-...
    # If the name starts with Untitled-..., it might have another UUID
    # But let's stick to the user's specific examples first
    
    # 4. Replace underscores with spaces
    name_body = name_body.replace("_", " ")
    
    # 5. Clean extra spaces and hyphens at start/end
    name_body = re.sub(r"\s+", " ", name_body).strip(" -")
    
    return f"{name_body}.{ext}"

# Test cases from user's list
test_files = [
    "My_Workspace-Zorya_(Remastered)-4f1903f0-e6cb-4570-ade2-e06905570242.wav",
    "Slavic-Kolo_(Remastered)-6f01353d-23aa-4b08-afb4-f47727d4e9dd.wav",
    "Theme_OST-Blóð_dýp-1fa048c8-d0f0-4b6c-9944-7642100a1449.wav",
    "Slavic-Percival_-_Sargon-3cbe8082-eb6d-4e91-b055-6a2d140bb7a3.wav"
]

print("Testing Renamer:")
for f in test_files:
    print(f"Original: {f}")
    print(f"Cleaned:  {clean_filename(f)}")
    print("-" * 20)
