$files = Get-ChildItem -Path 'c:\Users\armon\DEV_main\Spectral_Affinity\Slavic Data_Set' -File -Filter *.mp3;

$uniqueNames = @(
    "Zora_Vecernja", "Svarog_Kovac", "Perunov_Grom", "Mokos_Tkalo", "Moranin_Dah", "Velesov_Vrt", "Dazdbog_Dar", "Stribogov_Let", "Radegast_Dom", "Bjelobogov_Sjaj",
    "Triglav_Vid", "Rod_Pradavni", "Simarglov_Let", "Lesijev_Cuvaj", "Domovoj_Duh", "Rusalkin_Zal", "Vodenjakov_Mir", "Karacun_Zima", "Zduhac_Boj", "Vilin_Ples",
    "Alatir_Kamen", "Bujan_Otok", "Irij_Vrt", "Prav_Zakon", "Jav_Prisut", "Nav_Sjena", "Jarilov_Zar", "Kupalov_Oganj", "Kostroma_Plac", "Lada_Sloga",
    "Lelov_Pev", "Polevik_Polja", "Ovinnik_Iskra", "Bannik_Para", "Bolotnik_Mah", "Gamajun_Prica", "Alkonost_Rad", "Sirin_Tuga", "Rarog_Zar", "Vedmak_Klet",
    "Vedma_Znanje", "Koscej_Besmrt", "Vasilisa_Lijep", "Ivanov_Put", "Svatogor_Gor", "Ilja_Snaga", "Dobrinja_Cist", "Aljosa_Mudr", "Sadko_Put", "Volkh_Mijen",
    "Zmej_Oganj", "Koljada_Snijeg", "Maslenica_Sun", "Rusalka_Varka", "Bogatir_Stit", "Kolovrat_Krug", "Svargov_Put", "Irij_Obzor", "Viraj_Duh", "Dolja_Sreca",
    "Nedolja_Klet", "Sudice_Nit", "Rodjanice_Dar", "Krsnik_Moc", "Vedogon_San", "Strzyga_Znak", "Upir_Zedj", "Vampir_Kandz", "Topielec_Dub", "Poludnica_Dan",
    "Nocnica_Noc", "Likho_Oko", "Chrt_Varka", "Bes_Sapat", "Shishiga_Guz", "Kikimora_Igla", "Polevoj_Vrt", "Lugovoj_Dah", "Morel_Zvon", "Zimnik_Mraz",
    "Vesnik_Cvat", "Letnik_Topl", "Osennik_Kise", "Soko_Let", "Orel_Vrh", "Jelen_Rog", "Medved_Moc", "Vukov_Krik", "Lisica_Mudr", "Zaba_Izvor",
    "Ptica_Eho", "Suma_Sjena", "Polje_Zlato", "More_Dubina", "Nebo_Vis", "Zemlja_Kor"
)

for ($i = 0; $i -lt $files.Count; $i++) {
    if ($i -lt $uniqueNames.Count) {
        $newName = $uniqueNames[$i] + ".mp3"
        $oldFile = $files[$i]
        Write-Host "Renaming $($oldFile.Name) to $newName"
        Rename-Item -Path $oldFile.PSPath -NewName $newName -Force
    }
}
