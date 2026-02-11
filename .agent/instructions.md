# Spectral Affinity: Flujo de Trabajo Técnico

Este proyecto implementa una "Huella Digital de Audio" para agrupar canciones sin escucharlas, analizando sus características matemáticas.

## Características de Audio Extraídas

1.  **MFCCs (Mel-frequency cepstral coefficients):** Capturan el timbre y color del sonido (ej. diferencia entre guitarra y piano). Extraemos el promedio y la varianza de los primeros 13 coeficientes.
2.  **Spectral Centroid:** Indica el "centro de masa" del espectro. Valores altos sugieren sonidos brillantes (agudos), valores bajos sonidos oscuros (graves).
3.  **Tempo (BPM):** Estima la velocidad rítmica de la pista.

## Optimizaciones de Rendimiento

-   **Análisis Central:** Se cargan únicamente 30 segundos centrales de cada pista para captar la esencia musical sin procesar el archivo completo.
-   **Procesamiento Paralelo:** Utiliza `joblib` para extraer características de múltiples archivos simultáneamente, aprovechando todos los núcleos de la CPU.

## Lógica de Agrupación (Clustering)

-   **Escalado:** Las características se normalizan usando `StandardScaler` para que el Tempo (ej. 120) no eclipse a los MFCCs (valores pequeños).
-   **K-Means:** Agrupa los vectores de características en `N` grupos definidos por el usuario.

## Seguridad

-   **Modo Copia:** Por defecto, el script utiliza `shutil.copy2` para duplicar los archivos en carpetas organizadas, preservando los originales y sus metadatos.