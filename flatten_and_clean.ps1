$baseDir = "c:\Users\armon\DEV_main\Spectral_Affinity\data_set"

# 1. Move all files to the base directory and rename them
$files = Get-ChildItem -Path $baseDir -File -Recurse

foreach ($file in $files) {
    $currentName = $file.Name
    # Pattern: "01 - [05A - 127BPM] Name.flac" -> "Name.flac"
    # Regex: Start with digits, space, dash, space, brackets with anything inside, then space(s)
    $newName = $currentName -replace "^\d{2}\s-\s\[.*\]\s+", ""
    
    $destPath = Join-Path -Path $baseDir -ChildPath $newName
    
    # Handle filename collisions if they occur (add suffix if necessary)
    $count = 1
    $finalDestPath = $destPath
    while (Test-Path -Path $finalDestPath) {
        if ($file.FullName -eq $finalDestPath) { break } # It's the same file (already in root)
        $extension = [System.IO.Path]::GetExtension($newName)
        $baseName = [System.IO.Path]::GetFileNameWithoutExtension($newName)
        $finalDestPath = Join-Path -Path $baseDir -ChildPath "$($baseName)_$count$($extension)"
        $count++
    }

    if ($file.FullName -ne $finalDestPath) {
        Write-Host "Moving and renaming: $($file.Name) -> $($Path::GetFileName($finalDestPath))"
        Move-Item -Path $file.FullName -Destination $finalDestPath -Force
    }
}

# 2. Remove empty subdirectories
Get-ChildItem -Path $baseDir -Directory | Remove-Item -Recurse -Force
