$files = Get-ChildItem -Path 'c:\Users\armon\DEV_main\Spectral_Affinity\Slavic Data_Set' -File -Filter *.mp3;
$counts = @{};

# Mapping table
$mapping = @{
    'Master' = 'Grom';
    'Bela_Noc' = 'Noc';
    'Chernozem' = 'Zem';
    'Drum_pattern' = 'Rit';
    'Witcher' = 'Vuk';
    'Ogień_Gore' = 'Og';
    'REF2' = 'Odj';
    'Sardon' = 'Srd';
    'Slavic' = 'Rod';
    'Tika' = 'Tik';
    'Trzask' = 'Trz';
    'Zorya' = 'Zor';
    'Anielka' = 'An';
    'Put' = 'Put';
    'Duh' = 'Duh';
    'Gora' = 'Gor';
    'Istina' = 'Ist';
    'Plamen' = 'Plam';
    'Grob' = 'Grob';
    'Izvor' = 'Izv';
    'Krug' = 'Kru';
    'Krv' = 'Krv';
    'Zlato' = 'Zla';
    'Most' = 'Mos';
    'Mrak' = 'Mra';
    'Svedok' = 'Sve';
    'Noć' = 'Noc';
    'Vidik' = 'Vid';
    'Odjek' = 'Odj';
    'Oganj' = 'Og';
    'Ognjište' = 'Dom';
    'Vetra' = 'Vet';
    'Platno' = 'Nit';
    'Polje' = 'Pol';
    'Čas' = 'Cas';
    'Povratak' = 'Pov';
    'Soba' = 'Sob';
    'Korak' = 'Kor';
    'Pusta' = 'Pus';
    'Reka' = 'Rek';
    'Horizont' = 'Hor';
    'Sjen' = 'Sje';
    'Staklo' = 'Led';
    'Kamena' = 'Src';
    'Hrast' = 'Hra';
    'Zavet' = 'Zav';
    'Stranac' = 'Gos';
    'Obala' = 'Rub';
    'Voda' = 'Vod';
    'Tišina' = 'Tis';
    'San' = 'San';
    'Grad' = 'Gra';
    'Vera' = 'Ver';
    'Zid' = 'Zid';
    'Zima' = 'Zim';
    'Jesen' = 'Jes';
    'Peska' = 'Pes';
    'Mir' = 'Mir';
    'Baba' = 'Zit';
    'Untitled' = 'Taj' # 'Taj' as in Tajna (Secreto)
}

foreach ($f in $files) {
    $found = $false;
    $newName = 'Slav';
    
    foreach ($key in $mapping.Keys) {
        if ($f.Name -match $key) {
            $newName = $mapping[$key];
            $found = $true;
            break;
        }
    }
    
    if ($counts.ContainsKey($newName)) {
        $counts[$newName]++
        $finalName = $newName + '_' + $counts[$newName] + '.mp3'
    } else {
        $counts[$newName] = 1
        $finalName = $newName + '.mp3'
    }
    
    $dest = Join-Path -Path $f.DirectoryName -ChildPath $finalName
    if ($f.FullName -ne $dest) {
        Write-Host "Renaming $($f.Name) to $finalName"
        Rename-Item -Path $f.PSPath -NewName $finalName -Force
    }
}
