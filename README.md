# Proyecto ARGOS 👁️

Revisor automático de requisiciones escaneadas contra la base SAFISS.

ARGOS toma requisiciones en PDF/imagen, extrae los ítems con OCR (opcionalmente
asistido por Gemini AI), y las cruza con el Excel de movimientos de SAFISS
para detectar diferencias de cantidad, códigos faltantes, documentos fuera del
período consultado y errores de lectura.

## Requisitos

- Python 3.10+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) instalado en el sistema
- Opcional: `GOOGLE_API_KEY` en un archivo `.env` para habilitar Gemini AI

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

```bash
streamlit run app.py
```

En la barra lateral:

1. **Requisiciones escaneadas** — carpeta local o subir PDFs/imágenes.
2. **Base SAFISS** — Excel/CSV con los movimientos (acepta el formato original de SAFISS o el normalizado).
3. **Ejecutar** — corre el pipeline y muestra:
   - Resumen por documento (OK / faltantes / diferencias / fuera de rango)
   - Detalle por ítem con posibilidad de abrir el PDF original
   - Reproceso selectivo con Gemini para verificar inconsistencias
   - Descarga del reporte en Excel

## Cómo se hace el cruce

- **Clave**: `no_doc + codigo` — cada ítem se identifica con el número de documento (OCR/nombre de archivo) y el código de material.
- Solo se comparan los documentos que se escanearon (SAFISS se filtra por `no_doc`).
- Solo se compara `cant_despachada`; los ítems de SAFISS sin despacho se ignoran.
- Un documento es **OK** si todos sus ítems despachados coinciden en cantidad.

## Resolución de número de documento

El número de doc se determina por consenso entre tres fuentes:

- **Nombre de archivo** (`4931159608_...`)
- **OCR local** (Tesseract sobre la imagen deskeada)
- **Gemini AI** (si está habilitado)

Cuando las fuentes discrepan, se aplican heurísticas para preferir la versión
completa sobre la abreviada, corregir confusiones OCR de 1 dígito y reconciliar
documentos con nombre de archivo mal tipeado.

## Reconciliación post-OCR

Antes de comparar contra SAFISS, se corrigen dos tipos de error frecuentes:

1. **Número mal leído/tipeado** — si el `no_doc` escaneado no existe en SAFISS
   pero hay uno a distancia ≤1 dígito con los mismos códigos, se remapea.
2. **PDF mal nombrado** — si el `no_doc` sí existe pero sus códigos no cuadran
   y `no_doc_ocr_local` apunta a otro doc de SAFISS con solape alto, se remapea.

## Estructura

- [app.py](app.py) — Streamlit UI y lógica de comparación
- [extract.py](extract.py) — Pipeline de OCR y extracción de tablas
- [requirements.txt](requirements.txt) — dependencias Python

## Datos

Los archivos de datos reales (requisiciones escaneadas, movimientos SAFISS,
catálogo de códigos) están excluidos del repositorio por contener información
sensible del hospital.
