"""Interfaz web para extraer datos de escaneos y compararlos con una base externa."""
import io
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
import zipfile
from difflib import SequenceMatcher
from pathlib import Path

SUPPORTED_SCAN_EXTS = {'.pdf', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

import fitz
import pandas as pd
import streamlit as st

# En Streamlit Cloud los secrets se configuran en el dashboard y sólo se
# exponen vía `st.secrets`; el resto del código lee `os.environ`, así que
# los replicamos ahí antes de importar cualquier módulo que consulte la key.
try:
    for _secret_key in ('GOOGLE_API_KEY',):
        if _secret_key in st.secrets and not os.environ.get(_secret_key):
            os.environ[_secret_key] = str(st.secrets[_secret_key])
except Exception:
    # Local sin `secrets.toml`: la key se toma del `.env` o del shell y no
    # hay nada que replicar.
    pass

import extract as extract_mod
from extract import iter_extract_folder, list_input_files, order_columns, process_pdf

import json as _json

# Checkpoint en disco: `session_state` vive en la RAM del container y muere
# con él (OOM, reboot, deploy). `/tmp` en Streamlit Cloud sobrevive muchos
# reinicios, así que persistimos ahí después de cada archivo procesado.
CHECKPOINT_PATH = Path(tempfile.gettempdir()) / 'argos_extraction_checkpoint.json'


def _save_checkpoint(rows: list[dict]) -> None:
    try:
        CHECKPOINT_PATH.write_text(_json.dumps(rows, default=str))
    except Exception:
        # Falla al persistir no debe frenar la extracción; el checkpoint es
        # best-effort, y la corrida sigue con lo que quede en memoria.
        pass


def _load_checkpoint() -> list[dict]:
    if not CHECKPOINT_PATH.is_file():
        return []
    try:
        data = _json.loads(CHECKPOINT_PATH.read_text())
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _clear_checkpoint() -> None:
    try:
        CHECKPOINT_PATH.unlink(missing_ok=True)
    except Exception:
        pass


@st.cache_data(show_spinner=False)
def _runtime_versions() -> dict[str, str]:
    """Reúne versiones y hashes de las libs sensibles para OCR.

    El delta más impactante para lecturas distintas suele ser el archivo
    `spa.traineddata` (modelo LSTM del español): distintas distribuciones
    empaquetan versiones diferentes con precisión desigual, y no hay forma
    de detectarlo por número de versión — hay que mirar tamaño/hash.
    """
    import hashlib
    import importlib.metadata as _im
    import platform
    import shutil
    versions: dict[str, str] = {
        'Sistema': f'{platform.system()} {platform.release()} ({platform.machine()})',
        'Python': platform.python_version(),
    }
    try:
        import pytesseract
        versions['Tesseract'] = str(pytesseract.get_tesseract_version())
        langs = pytesseract.get_languages(config='')
        versions['Idiomas Tesseract'] = ', '.join(sorted(langs)) or '(ninguno)'
        # Hash del modelo LSTM del español: si difiere entre local y Cloud,
        # tenés un modelo distinto entrenado con datos distintos y la
        # precisión sobre dígitos/acentos cambia — es la diferencia real.
        traineddata_paths = [
            '/opt/homebrew/share/tessdata/spa.traineddata',
            '/usr/local/share/tessdata/spa.traineddata',
            '/usr/share/tesseract-ocr/5/tessdata/spa.traineddata',
            '/usr/share/tesseract-ocr/4.00/tessdata/spa.traineddata',
            '/usr/share/tessdata/spa.traineddata',
        ]
        found_path = next((p for p in traineddata_paths if Path(p).is_file()), None)
        if found_path:
            data = Path(found_path).read_bytes()
            versions['spa.traineddata'] = (
                f'{len(data)//1024} KB, sha256={hashlib.sha256(data).hexdigest()[:12]}'
            )
        else:
            versions['spa.traineddata'] = 'no encontrado en rutas estándar'
    except Exception as e:
        versions['Tesseract'] = f'error: {type(e).__name__}: {e}'
    try:
        import cv2
        # `opencv-python` y `opencv-python-headless` instalan el mismo módulo
        # `cv2`; hay que preguntar al metadata de la distro para saber cuál.
        installed = []
        for dist_name in ('opencv-python-headless', 'opencv-python',
                          'opencv-contrib-python-headless', 'opencv-contrib-python'):
            try:
                installed.append(f'{dist_name}=={_im.version(dist_name)}')
            except _im.PackageNotFoundError:
                continue
        # Buscamos "GUI: NONE" o similar en el build info: ausencia de HighGUI
        # es señal fiable de que es la variante headless.
        build_info = cv2.getBuildInformation()
        gui_line = next(
            (line.strip() for line in build_info.splitlines() if 'GUI:' in line),
            'GUI: ?',
        )
        pkg_label = ' + '.join(installed) or 'desconocido'
        versions['OpenCV'] = f'{cv2.__version__} [{pkg_label}] {gui_line}'
    except Exception as e:
        versions['OpenCV'] = f'error: {e}'
    try:
        versions['PyMuPDF'] = fitz.__version__
    except Exception as e:
        versions['PyMuPDF'] = f'error: {e}'
    try:
        import numpy as _np
        versions['NumPy'] = _np.__version__
    except Exception:
        pass
    try:
        versions['Pandas'] = pd.__version__
    except Exception:
        pass
    try:
        versions['Streamlit'] = st.__version__
    except Exception:
        pass
    tess_binary = shutil.which('tesseract')
    if tess_binary:
        versions['Binario Tesseract'] = tess_binary
    return versions


KEYS = ['no_doc', 'codigo']
COMPARE_FIELDS = ['cant_despachada']

SAFISS_MAP = {
    'Documento material':      'no_doc',
    'Material':                 'codigo',
    'Texto breve de material':  'nombre_producto',
    'Almacén':                  'almacen',
    'Clase de movimiento':      'mov',
    'Un.medida de entrada':     'presentacion',
    'Centro de coste':          'depend_destinataria',
    'Fe.contabilización':       'fecha',
    'Nombre 1':                 'centro_receptor',
    'Nombre del usuario':       'usuario_safiss',
}
SAFISS_SIGNATURE = {'Material', 'Documento material', 'Ctd.en UM entrada'}


def _clean_id(v):
    if pd.isna(v):
        return ''
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


def is_safiss_format(df: pd.DataFrame) -> bool:
    return SAFISS_SIGNATURE.issubset(df.columns)


def normalize_safiss(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns=SAFISS_MAP).copy()
    for col in ('no_doc', 'codigo', 'mov', 'depend_destinataria'):
        if col in out.columns:
            out[col] = out[col].map(_clean_id)
    if 'Ctd.en UM entrada' in out.columns:
        out['cant_despachada'] = out['Ctd.en UM entrada'].abs()
    return out

STATUS_STYLE = {
    'ok':             ('background-color: rgba(45, 200, 100, 0.18)',  '✅', 'OK'),
    'solo_externa':   ('background-color: rgba(255, 193, 7, 0.28)',   '⚠️', 'Solo en SAFISS'),
    'solo_escaneada': ('background-color: rgba(59, 130, 246, 0.28)',  '📄', 'Solo escaneada'),
    'fuera_rango':    ('background-color: rgba(100, 116, 139, 0.30)', '📅', 'Fuera de rango SAFISS'),
}
DIFF_STYLE = ('background-color: rgba(239, 68, 68, 0.32)', '❌', 'Cantidad difiere')


st.set_page_config(page_title='Proyecto ARGOS', page_icon='👁️', layout='wide')
st.title('Proyecto ARGOS')
st.caption(
    'El guardián de los cien ojos: observa cada requisición, contrasta sus datos '
    'con SAFISS y señala oportunamente cualquier inconsistencia.'
)
with st.expander('👁️ Acerca de ARGOS, el guardián', expanded=False):
    st.markdown(
        '**ARGOS** toma su nombre de Argos Panoptes, el guardián de los cien ojos. '
        'Su vigilancia constante representa el propósito del proyecto: revisar cada '
        'documento y cada ítem con atención, sin perder de vista los detalles.\n\n'
        'El sistema combina lectura de documentos escaneados, validación asistida '
        'por inteligencia artificial y comparación con la base de SAFISS. Así ayuda '
        'a identificar diferencias de cantidades, códigos faltantes, documentos fuera '
        'del período consultado y posibles errores de lectura antes de emitir el reporte.\n\n'
        '**Misión:** brindar una revisión clara, trazable y confiable que facilite la '
        'verificación de requisiciones y permita concentrar la atención humana donde '
        'realmente se necesita.'
    )
with st.expander('🚀 Cómo usar ARGOS (paso a paso)', expanded=False):
    st.markdown(
        '**1. Cargá tus requisiciones escaneadas** (panel izquierdo)\n\n'
        '   Podés apuntar a una **carpeta local** con los PDFs o subirlos '
        'directamente. Se aceptan PDF e imágenes (PNG, JPG, TIFF).\n\n'
        '**2. Subí el Excel de SAFISS** (panel izquierdo)\n\n'
        '   Sirve el Excel original tal como lo descargás de SAFISS. ARGOS '
        'reconoce automáticamente sus columnas (`Material`, `Documento '
        'material`, `Ctd.en UM entrada`, etc.).\n\n'
        '**3. Presioná "Procesar y comparar"**\n\n'
        '   ARGOS va a leer cada requisición, extraer los ítems y compararlos '
        'con SAFISS. Vas a ver el progreso en tiempo real.\n\n'
        '**4. Revisá los resultados** (pestaña *Comparación*)\n\n'
        '   Al final aparece un resumen con los KPIs principales y una tabla '
        'de detalle. Podés hacer clic en una fila para ver el PDF original.\n\n'
        '**5. Descargá el reporte en Excel** para archivar o compartir.'
    )
with st.expander('🎨 Qué significa cada color/estado', expanded=False):
    st.markdown(
        '- ✅ **OK** — el ítem coincide en cantidad con SAFISS.\n'
        '- ❌ **Cantidad difiere** — el código está en ambos lados pero las '
        'cantidades no coinciden. Revisá el escaneo original.\n'
        '- 📄 **Solo escaneada** — el ítem aparece en la requisición pero no '
        'está en SAFISS (posible faltante de registro en SAFISS o error de '
        'lectura del código).\n'
        '- ⚠️ **Solo en SAFISS** — SAFISS tiene un movimiento que no aparece '
        'en la requisición escaneada (puede faltar el escaneo o el OCR omitió '
        'la fila).\n'
        '- 📅 **Fuera de rango SAFISS** — la fecha del documento cae fuera del '
        'período que cargaste, así que no hay contra qué compararlo.'
    )
with st.expander('🔧 Cómo se hace el cruce (detalle técnico)', expanded=False):
    st.markdown(
        '- **Identificador**: cada ítem se identifica por el **número de '
        'documento** y el **código de material**. El número de documento se '
        'toma del OCR del PDF y se contrasta con el nombre del archivo y con '
        'la lectura de la IA (si está activa) para elegir el más confiable.\n'
        '- **Alcance**: solo se comparan los documentos que aparecen en tus '
        'escaneos (SAFISS se filtra automáticamente).\n'
        '- **Qué se compara**: la cantidad despachada. Los ítems de SAFISS con '
        'cantidad cero (que no generaron movimiento) se ignoran. Un documento '
        'se marca OK si **todos** sus ítems despachados coinciden.\n'
        '- **Correcciones automáticas**: cuando el número de documento leído no '
        'existe en SAFISS pero hay uno muy parecido (1 dígito distinto o '
        'nombre truncado) con los mismos códigos, ARGOS lo remapea '
        'automáticamente para evitar falsos positivos.'
    )


def load_external(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    df = pd.read_csv(uploaded) if name.endswith('.csv') else pd.read_excel(uploaded)
    if is_safiss_format(df):
        st.toast('Formato SAFISS detectado — columnas normalizadas.', icon='✅')
        return normalize_safiss(df)
    return df


FILENAME_DATE_RE = re.compile(
    r'(?<!\d)(\d{1,2})[.-](\d{1,2})[.-](\d{4}|\d{2})(?!\d)'
)
COMPACT_FILENAME_DATE_RE = re.compile(
    r'(?<!\d)(\d{1,2})[.-](\d{2})(\d{2})(?!\d)'
)
FILENAME_YEAR_RE = re.compile(r'[.-](\d{4}|\d{2})(?=\D*$)')


def _date_from_archivo(name):
    """Obtiene fechas DD.MM.YYYY, DD-MM-YYYY o DD-MM-YY del archivo."""
    if not isinstance(name, str):
        return None
    m = FILENAME_DATE_RE.search(name) or COMPACT_FILENAME_DATE_RE.search(name)
    if not m:
        return None
    d, mo, y = m.groups()
    try:
        year = int(y)
        if year < 100:
            year += 2000
        return pd.Timestamp(year=year, month=int(mo), day=int(d))
    except ValueError:
        return None


def _date_range_from_archivo(name) -> tuple[pd.Timestamp, pd.Timestamp, str] | None:
    """Devuelve fecha exacta o, como respaldo, el rango anual del archivo."""
    exact = _date_from_archivo(name)
    if exact is not None:
        return exact, exact, exact.strftime('%d.%m.%Y')
    if not isinstance(name, str):
        return None
    match = FILENAME_YEAR_RE.search(name)
    if not match:
        return None
    year = int(match.group(1))
    if year < 100:
        year += 2000
    if not 2000 <= year <= 2100:
        return None
    return (
        pd.Timestamp(year=year, month=1, day=1),
        pd.Timestamp(year=year, month=12, day=31),
        f'año {year}',
    )


def _safiss_date_range(externa: pd.DataFrame):
    if 'fecha' not in externa.columns:
        return None
    dates = pd.to_datetime(externa['fecha'], errors='coerce').dropna()
    if dates.empty:
        return None
    return dates.min(), dates.max()


def _open_file(path: Path) -> tuple[bool, str]:
    """Abre un archivo con el visor default del OS."""
    try:
        if sys.platform == 'darwin':
            subprocess.Popen(['open', str(path)])
        elif sys.platform.startswith('linux'):
            subprocess.Popen(['xdg-open', str(path)])
        elif sys.platform == 'win32':
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            return False, f'plataforma no soportada: {sys.platform}'
        return True, ''
    except Exception as e:
        return False, str(e)


def _document_path(no_doc, source_folder: str, escaneada: pd.DataFrame) -> Path | None:
    """Busca el archivo original asociado a un número de documento."""
    if 'no_doc' not in escaneada.columns or 'archivo' not in escaneada.columns:
        return None
    matches = escaneada.loc[
        escaneada['no_doc'].map(_clean_id) == _clean_id(no_doc), 'archivo'
    ].dropna()
    if matches.empty:
        return None
    return Path(source_folder) / str(matches.iloc[0])


@st.cache_data(show_spinner=False)
def _render_pdf_page(path: str, modified_ns: int, page_index: int) -> bytes:
    """Renderiza una página a PNG; modified_ns invalida la caché si cambia el PDF."""
    del modified_ns
    with fitz.open(path) as pdf:
        page = pdf.load_page(page_index)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(1.7, 1.7), alpha=False)
        return pixmap.tobytes('png')


@st.dialog('📄 Documento original', width='large')
def _show_document(path: Path) -> None:
    """Muestra el PDF o imagen en una ventana modal."""
    header, download, action = st.columns([5, 1, 1])
    header.caption(path.name)
    download.download_button(
        '⬇️',
        data=path.read_bytes(),
        file_name=path.name,
        mime='application/pdf' if path.suffix.lower() == '.pdf' else None,
        help='Descargar documento',
        width='stretch',
    )
    if action.button('Cerrar', key='close_document_preview', width='stretch'):
        st.session_state.pop('document_preview', None)
        st.rerun()

    if path.suffix.lower() == '.pdf':
        with fitz.open(path) as pdf:
            page_count = pdf.page_count

        current = min(max(int(st.session_state.get('pdf_page', 1)), 1), page_count)
        st.session_state['pdf_page'] = current
        previous, page_label, following = st.columns([1, 3, 1])
        if previous.button('← Anterior', disabled=current == 1, width='stretch'):
            st.session_state['pdf_page'] = current - 1
            st.rerun()
        page_label.markdown(
            f"<p style='text-align:center; margin:.45rem 0 0'>Página <b>{current}</b> de {page_count}</p>",
            unsafe_allow_html=True,
        )
        if following.button('Siguiente →', disabled=current == page_count, width='stretch'):
            st.session_state['pdf_page'] = current + 1
            st.rerun()

        page_png = _render_pdf_page(str(path), path.stat().st_mtime_ns, current - 1)
        st.image(page_png, width='stretch')
    else:
        st.image(str(path), caption=path.name, width='stretch')


def _fecha_por_doc(
    escaneada: pd.DataFrame,
) -> dict[str, tuple[pd.Timestamp, pd.Timestamp, str]]:
    if 'archivo' not in escaneada.columns or 'no_doc' not in escaneada.columns:
        return {}
    out = {}
    for _, r in escaneada[['no_doc', 'archivo']].dropna().drop_duplicates().iterrows():
        date_range = _date_range_from_archivo(r['archivo'])
        if date_range is not None:
            out[_clean_id(r['no_doc'])] = date_range
    return out


def _to_number(v):
    """Convierte cantidad (float, '100', '100.00', '1,234') a float. Devuelve None si no se puede."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(',', '').replace('|', '')
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _aggregate_by_key(df: pd.DataFrame) -> pd.DataFrame:
    """Colapsa filas repetidas por (no_doc, codigo) sumando cant_despachada.
    SAFISS suele tener el mismo material varias veces por documento (despachos parciales)."""
    if df.empty:
        return df
    agg = {c: 'first' for c in df.columns if c not in KEYS and c != 'cant_despachada'}
    if 'cant_despachada' in df.columns:
        agg['cant_despachada'] = 'sum'
    return df.groupby(KEYS, as_index=False).agg(agg)


def _codes_one_edit_apart(a: str, b: str) -> bool:
    """True para una sustitución, inserción o eliminación de un dígito."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    shorter, longer = (a, b) if len(a) < len(b) else (b, a)
    i = j = differences = 0
    while i < len(shorter) and j < len(longer):
        if shorter[i] == longer[j]:
            i += 1
            j += 1
        else:
            differences += 1
            j += 1
            if differences > 1:
                return False
    return True


def _normal_text(value) -> str:
    """Normaliza descripciones para comparar SAFISS con texto OCR ruidoso."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ''
    text = unicodedata.normalize('NFKD', str(value))
    text = ''.join(ch for ch in text if not unicodedata.combining(ch)).upper()
    return ' '.join(re.findall(r'[A-Z0-9]+', text))


def _description_similarity(a, b) -> float:
    a, b = _normal_text(a), _normal_text(b)
    if not a or not b:
        return 0.0
    sequence = SequenceMatcher(None, a, b).ratio()
    tokens_a, tokens_b = set(a.split()), set(b.split())
    token_overlap = len(tokens_a & tokens_b) / max(1, min(len(tokens_a), len(tokens_b)))
    return max(sequence, token_overlap)


def _code_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _reconcile_ocr_codes(externa: pd.DataFrame, escaneada: pd.DataFrame) -> pd.DataFrame:
    """Corrige un código OCR a partir de SAFISS cuando la coincidencia es inequívoca.

    Exige la misma cantidad despachada. Acepta un error de un dígito o una
    coincidencia fuerte de descripción + código; esta segunda vía resuelve
    lecturas truncadas/fusionadas. Si el código correcto ya fue extraído en otra
    pasada OCR, descarta la lectura errónea para no duplicar la cantidad.
    """
    if externa.empty or escaneada.empty or 'cant_despachada' not in escaneada.columns:
        return escaneada

    corrected = escaneada.copy()
    drop_indexes = []
    external_by_doc = {
        doc: group for doc, group in externa.groupby('no_doc', dropna=False)
    }

    for idx, row in corrected.iterrows():
        doc = row['no_doc']
        scanned_code = row['codigo']
        candidates_df = external_by_doc.get(doc)
        if candidates_df is None or scanned_code in set(candidates_df['codigo']):
            continue
        if not scanned_code.isdigit():
            continue

        scanned_qty = _to_number(row.get('cant_despachada'))
        if scanned_qty is None or scanned_qty <= 0:
            continue

        candidates = []
        for _, external_row in candidates_df.iterrows():
            external_code = external_row['codigo']
            external_qty = _to_number(external_row.get('cant_despachada'))
            if (not external_code.isdigit() or external_qty is None
                    or abs(external_qty - scanned_qty) > 1e-6):
                continue
            one_edit = _codes_one_edit_apart(scanned_code, external_code)
            product_similarity = _description_similarity(
                row.get('nombre_producto'), external_row.get('nombre_producto'),
            )
            code_similarity = _code_similarity(scanned_code, external_code)
            score = 0.70 * product_similarity + 0.30 * code_similarity
            # Las fusiones OCR pueden mover/duplicar varios dígitos y aun así
            # conservar casi toda la secuencia (902000857 -> 910200085).
            strong_code = one_edit or code_similarity >= 0.84
            if strong_code or (product_similarity >= 0.55 and score >= 0.60):
                candidates.append((external_code, score, strong_code))

        if not candidates:
            continue
        candidates.sort(key=lambda item: (item[2], item[1]), reverse=True)
        if len(candidates) > 1:
            first, second = candidates[0], candidates[1]
            # Una corrección por descripción debe ser claramente mejor que la
            # alternativa siguiente; evitamos adivinar entre productos parecidos.
            if not first[2] and first[1] - second[1] < 0.12:
                continue
            if first[2] and second[2]:
                continue

        corrected_code = candidates[0][0]
        already_present = (
            (corrected['no_doc'] == doc)
            & (corrected['codigo'] == corrected_code)
            & corrected['cant_despachada'].map(
                lambda value: (_to_number(value) is not None
                               and abs(_to_number(value) - scanned_qty) <= 1e-6)
            )
        ).any()
        if already_present:
            drop_indexes.append(idx)
        else:
            corrected.at[idx, 'codigo'] = corrected_code

    return corrected.drop(index=drop_indexes)


def _select_best_scanned_copy(escaneada: pd.DataFrame) -> pd.DataFrame:
    """Conserva una sola copia por documento para no duplicar cantidades.

    Los PDFs pueden ser copias visualmente iguales con bytes distintos, por lo
    que el hash no basta. Se prioriza la copia con más códigos, más cantidades
    legibles y cuyo número de archivo coincide con el documento resuelto.
    """
    required = {'no_doc', 'archivo', 'codigo'}
    if escaneada.empty or not required.issubset(escaneada.columns):
        return escaneada

    scored = escaneada.copy()
    scored['_cantidad_legible'] = (
        scored['cant_despachada'].map(_to_number).notna()
        if 'cant_despachada' in scored.columns else False
    )
    if 'no_doc_archivo' in scored.columns:
        scored['_nombre_coincide'] = (
            scored['no_doc_archivo'].map(_clean_id)
            == scored['no_doc'].map(_clean_id)
        )
    else:
        scored['_nombre_coincide'] = False

    quality = scored.groupby(['no_doc', 'archivo'], as_index=False).agg(
        codigos=('codigo', 'nunique'),
        cantidades_legibles=('_cantidad_legible', 'sum'),
        nombre_coincide=('_nombre_coincide', 'max'),
    )
    selected = (
        quality.sort_values(
            ['no_doc', 'codigos', 'cantidades_legibles', 'nombre_coincide', 'archivo'],
            ascending=[True, False, False, False, True],
        )
        .drop_duplicates(subset=['no_doc'], keep='first')[['no_doc', 'archivo']]
    )
    filtered = scored.merge(selected, on=['no_doc', 'archivo'], how='inner')
    return filtered.drop(columns=['_cantidad_legible', '_nombre_coincide'])


def _reconcile_scanned_doc_numbers(
    externa: pd.DataFrame, escaneada: pd.DataFrame,
) -> pd.DataFrame:
    """Corrige el `no_doc` del escaneo cuando SAFISS tiene otro documento cuyos
    códigos coinciden mejor con los ítems escaneados.

    Cubre dos casos:
      1. Doc escaneado inexistente en SAFISS + doc en SAFISS a ≤1 dígito con
         los mismos códigos (p. ej. `4930686996` → `4930686990`).
      2. Doc escaneado sí presente en SAFISS pero con solape de códigos casi
         nulo y `no_doc_ocr_local` apuntando a otro doc de SAFISS con solape
         alto (típico de PDF mal nombrado; p. ej. filename `4930843220` pero
         adentro `4930842933`).
    """
    if externa.empty or escaneada.empty:
        return escaneada

    externa_codes_by_doc: dict[str, set[str]] = {
        doc: set(group['codigo']) for doc, group in externa.groupby('no_doc')
    }
    externa_docs = set(externa_codes_by_doc)
    scanned_docs = set(escaneada['no_doc'].unique())
    ocr_local_col = 'no_doc_ocr_local'
    has_ocr_local = ocr_local_col in escaneada.columns

    def _codes_for(scanned_doc: str) -> set[str]:
        rows = escaneada.loc[escaneada['no_doc'] == scanned_doc]
        if 'cant_despachada' in rows.columns:
            # Los ítems con cero (o sin cantidad) están en la requisición pero
            # no fueron despachados y SAFISS no genera movimiento. Contarlos
            # aquí ensucia el score de solape al comparar sets de códigos.
            quantities = rows['cant_despachada'].map(_to_number)
            rows = rows[quantities.fillna(0) > 0]
        codes = set(rows['codigo'])
        codes.discard('')
        return codes

    def _overlap_score(scanned_codes: set[str], external_codes: set[str]) -> float:
        if not scanned_codes or not external_codes:
            return 0.0
        return len(scanned_codes & external_codes) / max(
            len(scanned_codes), len(external_codes),
        )

    remap: dict[str, str] = {}
    for scanned_doc in scanned_docs:
        if not scanned_doc:
            continue
        scanned_codes = _codes_for(scanned_doc)
        if not scanned_codes:
            continue

        current_score = _overlap_score(
            scanned_codes, externa_codes_by_doc.get(scanned_doc, set()),
        )
        # Si ya matchea bien contra su propio no_doc, no hay nada que reasignar.
        if current_score >= 0.5:
            continue

        candidates: list[tuple[str, float, int, int]] = []

        def _consider(external_doc: str) -> None:
            if not external_doc or external_doc == scanned_doc:
                return
            if external_doc not in externa_codes_by_doc:
                return
            # Evitamos que dos remaps colisionen en el mismo doc; pero permitimos
            # remapear a un doc ya escaneado (típico: mismo PDF con distinto
            # nombre). El dedup posterior por (no_doc, codigo) evita duplicar
            # cantidades y prefiere la fila canónica no remapeada.
            if external_doc in remap.values():
                return
            score = _overlap_score(scanned_codes, externa_codes_by_doc[external_doc])
            if score < 0.7:
                return
            edit = (
                sum(a != b for a, b in zip(scanned_doc, external_doc))
                if len(scanned_doc) == len(external_doc) else -1
            )
            overlap = len(scanned_codes & externa_codes_by_doc[external_doc])
            candidates.append((external_doc, score, overlap, edit))

        # Candidatos explícitos desde las lecturas alternativas (OCR local,
        # Gemini). Cubren casos de PDF mal nombrado o filename truncado.
        for alt_col in (ocr_local_col, 'no_doc_gemini'):
            if alt_col not in escaneada.columns:
                continue
            for value in (
                escaneada.loc[escaneada['no_doc'] == scanned_doc, alt_col]
                .dropna().map(_clean_id).unique()
            ):
                _consider(value)

        # Candidatos por edit distance ≤ 1 (misma longitud) o cuando uno es
        # sufijo del otro (filename abreviado vs número SAFISS completo).
        for external_doc in externa_codes_by_doc:
            if len(external_doc) == len(scanned_doc):
                if sum(a != b for a, b in zip(scanned_doc, external_doc)) <= 1:
                    _consider(external_doc)
            elif (external_doc.endswith(scanned_doc)
                  or scanned_doc.endswith(external_doc)):
                _consider(external_doc)

        if not candidates:
            continue

        # El mismo doc puede llegar por varias vías (OCR local, Gemini, sufijo).
        # Deduplicamos por external_doc antes del chequeo de ambigüedad, si no,
        # dos entradas idénticas se leen como "empate" y descartan el remap.
        best_by_doc: dict[str, tuple[str, float, int, int]] = {}
        for candidate in candidates:
            existing = best_by_doc.get(candidate[0])
            if existing is None or candidate[1] > existing[1]:
                best_by_doc[candidate[0]] = candidate
        candidates = list(best_by_doc.values())

        # Un remap sobre un doc que ya existe en SAFISS es más agresivo: exigimos
        # que el candidato supere claramente el solape actual para hacerlo.
        if scanned_doc in externa_docs:
            candidates = [c for c in candidates if c[1] >= current_score + 0.3]
            if not candidates:
                continue

        candidates.sort(
            key=lambda item: (item[1], item[2], -item[3] if item[3] >= 0 else -99),
            reverse=True,
        )
        if len(candidates) > 1 and candidates[0][1] - candidates[1][1] < 0.1:
            continue
        remap[scanned_doc] = candidates[0][0]

    if not remap:
        return escaneada

    escaneada = escaneada.copy()
    escaneada['no_doc_original'] = escaneada['no_doc']
    escaneada['no_doc'] = escaneada['no_doc'].map(lambda d: remap.get(d, d))
    # Después del remap puede haber (no_doc, codigo) duplicado si el target ya
    # estaba escaneado por otro PDF. Preferimos la fila canónica (no remapeada)
    # para no sumar cantidades de una copia extra del mismo documento.
    escaneada['_is_remapped'] = escaneada['no_doc'] != escaneada['no_doc_original']
    escaneada = (
        escaneada.sort_values('_is_remapped', kind='stable')
        .drop_duplicates(subset=['no_doc', 'codigo'], keep='first')
        .drop(columns='_is_remapped')
    )
    escaneada.attrs['doc_number_remap'] = remap
    return escaneada


def compare(externa: pd.DataFrame, escaneada: pd.DataFrame) -> pd.DataFrame:
    for df in (externa, escaneada):
        for k in KEYS:
            if k not in df.columns:
                raise ValueError(f"Falta la columna '{k}' en uno de los datasets")
            df[k] = df[k].map(_clean_id)

    externa = externa[externa['no_doc'].astype(bool) & externa['codigo'].astype(bool)].copy()
    escaneada = escaneada[escaneada['no_doc'].astype(bool) & escaneada['codigo'].astype(bool)].copy()

    # Un mismo PDF puede existir con varios nombres. Se conserva una sola fila
    # por archivo físico y código para no sumar copias idénticas como despachos.
    if 'archivo_hash' in escaneada.columns:
        escaneada = escaneada.drop_duplicates(
            subset=['archivo_hash', 'no_doc', 'codigo'], keep='first',
        )
    escaneada = _select_best_scanned_copy(escaneada)

    escaneada = _reconcile_scanned_doc_numbers(externa, escaneada)

    docs_escaneados = set(escaneada['no_doc'].unique())
    if docs_escaneados:
        externa = externa[externa['no_doc'].isin(docs_escaneados)].copy()

    if 'cant_despachada' in externa.columns:
        # SAFISS representa las salidas de almacén con signo negativo. Algunos
        # archivos llegan con el formato original y otros ya traen las columnas
        # normalizadas; por eso el valor absoluto debe aplicarse siempre aquí,
        # en la frontera de comparación, y no depender del formato de entrada.
        externa['cant_despachada'] = externa['cant_despachada'].map(
            lambda value: (
                abs(number) if (number := _to_number(value)) is not None else None
            )
        )
        externa = externa[externa['cant_despachada'].fillna(0) > 0]
    if 'cant_despachada' in escaneada.columns:
        escaneada['cant_despachada'] = escaneada['cant_despachada'].map(_to_number)
        # Una línea con cero pertenece a la requisición, pero no fue despachada;
        # SAFISS no genera movimiento para ella y no debe contarse como faltante.
        escaneada = escaneada[
            escaneada['cant_despachada'].isna()
            | (escaneada['cant_despachada'] != 0)
        ].copy()

    escaneada = _reconcile_ocr_codes(externa, escaneada)

    externa = _aggregate_by_key(externa)
    escaneada = _aggregate_by_key(escaneada)

    merged = externa.merge(
        escaneada, on=KEYS, how='outer',
        suffixes=('_ext', '_esc'), indicator=True,
    )

    def categorize(r):
        if r['_merge'] == 'left_only':
            return 'solo_externa'
        if r['_merge'] == 'right_only':
            return 'solo_escaneada'
        issues = []
        for f in COMPARE_FIELDS:
            a, b = _to_number(r.get(f'{f}_ext')), _to_number(r.get(f'{f}_esc'))
            if a is None or b is None:
                continue
            if abs(a - b) > 1e-6:
                issues.append(f)
        return ' + '.join(f'{i}_difiere' for i in issues) if issues else 'ok'

    merged['estado'] = merged.apply(categorize, axis=1)
    merged = merged.drop(columns=['_merge'])

    lead = ['estado', 'no_doc', 'codigo']
    pairs = []
    for f in COMPARE_FIELDS + ['nombre_producto', 'presentacion']:
        for suf in ('_ext', '_esc'):
            c = f + suf
            if c in merged.columns:
                pairs.append(c)
    rest = [c for c in merged.columns if c not in lead + pairs]
    return merged[lead + pairs + rest]


MAX_STYLED_ROWS = 500

COLUMN_LABELS = {
    'estado':                    'Estado',
    'no_doc':                    'No. documento',
    'no_doc_gemini':             'No. documento (Gemini)',
    'no_doc_ocr_local':          'No. documento (OCR local)',
    'no_doc_archivo':            'No. documento (archivo)',
    'no_doc_fuente':             'Fuente del documento',
    'no_doc_coincide_archivo':   'Coincide con archivo',
    'no_doc_coincide_ocr':       'Gemini coincide con OCR',
    'codigo':                    'Código',
    'cant_despachada_ext':       'Cant. despachada (SAFISS)',
    'cant_despachada_esc':       'Cant. despachada (escaneo)',
    'nombre_producto_ext':       'Producto (SAFISS)',
    'nombre_producto_esc':       'Producto (escaneo)',
    'presentacion_ext':          'Presentación (SAFISS)',
    'presentacion_esc':          'Presentación (escaneo)',
    'almacen_ext':               'Almacén (SAFISS)',
    'almacen_esc':               'Almacén (escaneo)',
    'mov_ext':                   'Movimiento (SAFISS)',
    'mov_esc':                   'Movimiento (escaneo)',
    'depend_destinataria_ext':   'Dependencia (SAFISS)',
    'depend_destinataria_esc':   'Dependencia (escaneo)',
    'fecha':                     'Fecha SAFISS',
    'centro_receptor':           'Centro receptor',
    'usuario_safiss':            'Usuario SAFISS',
    'Centro':                    'Centro',
    'archivo':                   'Archivo',
    'cant_solicitada':           'Cant. solicitada',
    'total_despachada':          'Total despachado',
    'total_no_despachada':       'Total no despachado',
    'solicito':                  'Solicitó',
    'autorizo':                  'Autorizó',
    'recibio':                   'Recibió',
    'preparado_por':             'Preparado por',
    'comprobado_por':            'Comprobado por',
    'elaborado':                 'Elaborado',
    'idem':                      'Ítem',
    'lote':                      'Lote',
    'año':                       'Año',
    'mes':                       'Mes',
    'items':                     'Ítems totales',
    'ok':                        'Ítems OK',
    'con_problemas':             'Ítems con problemas',
    'diagnóstico':               'Diagnóstico',
}


def style_merged(df: pd.DataFrame):
    def per_row(row):
        status_value = str(row['estado'])
        status_key = next(
            (key for key, (_, icon, label) in STATUS_STYLE.items()
             if status_value in (key, f'{icon} {label}')),
            None,
        )
        base, _, _ = STATUS_STYLE.get(status_key, DIFF_STYLE)
        styles = [base] * len(row)
        if 'difiere' in str(row['estado']):
            for f in COMPARE_FIELDS:
                a, b = _to_number(row.get(f'{f}_ext')), _to_number(row.get(f'{f}_esc'))
                if a is None or b is None or abs(a - b) <= 1e-6:
                    continue
                for suf in ('_ext', '_esc'):
                    col = f + suf
                    if col in df.columns:
                        styles[df.columns.get_loc(col)] = DIFF_STYLE[0] + '; font-weight: 600'
        return styles
    return df.style.apply(per_row, axis=1)


def estado_label(e: str) -> str:
    if e in STATUS_STYLE:
        return f"{STATUS_STYLE[e][1]} {STATUS_STYLE[e][2]}"
    return f"{DIFF_STYLE[1]} {e}"


@st.cache_data(show_spinner=False)
def resumen_por_documento(merged: pd.DataFrame) -> pd.DataFrame:
    def clasificar(g):
        estados = set(g['estado'])
        if estados == {'ok'}:
            return 'OK'
        issues = []
        if any('difiere' in e for e in estados):
            issues.append('cantidad ≠')
        if 'solo_externa' in estados:
            issues.append('faltan escaneadas')
        if 'solo_escaneada' in estados:
            issues.append('faltan en SAFISS')
        return ', '.join(issues) or 'revisar'

    grouped = merged.groupby('no_doc', dropna=False).apply(
        lambda g: pd.Series({
            'items': len(g),
            'ok': (g['estado'] == 'ok').sum(),
            'con_problemas': (g['estado'] != 'ok').sum(),
            'diagnóstico': clasificar(g),
        }),
        include_groups=False,
    ).reset_index()
    return grouped.sort_values('con_problemas', ascending=False)


def to_excel_bytes(dfs: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as w:
        for name, df in dfs.items():
            df.to_excel(w, sheet_name=name[:31], index=False)
    return buf.getvalue()


with st.sidebar:
    st.header('1. Requisiciones escaneadas')
    # En entornos donde no existe la carpeta por default (p. ej. Streamlit
    # Cloud), arranca en modo "Subir archivos" para que la app sea usable de
    # entrada sin tener que cambiar el modo manualmente.
    default_local_folder = Path(__file__).parent / 'documentos'
    default_mode_index = 0 if default_local_folder.exists() else 1
    src_mode = st.radio(
        'Origen', ['Carpeta local', 'Subir archivos'],
        index=default_mode_index,
    )
    folder_path = uploaded_files = None
    if src_mode == 'Carpeta local':
        folder_path = st.text_input(
            'Ruta de la carpeta',
            value=str(default_local_folder),
            help='Carpeta con las requisiciones escaneadas (PDF o imagen)',
        )
    else:
        uploaded_files = st.file_uploader(
            'Archivos escaneados',
            type=['pdf', 'png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp', 'zip'],
            accept_multiple_files=True,
            help=('Podés subir PDFs/imágenes sueltos o un ZIP con la '
                  'carpeta comprimida (recomendado si son más de 50 archivos).'),
        )

    limit_enabled = st.checkbox('Limitar cantidad de archivos', value=False,
                                help='Útil para probar rápido con una muestra')
    file_limit = st.number_input('Máx. archivos a procesar', min_value=1, value=5, step=1,
                                 disabled=not limit_enabled) if True else None

    has_gemini_key = bool(os.getenv('GOOGLE_API_KEY'))
    use_gemini = st.checkbox(
        '🤖 Potenciar con IA',
        value=has_gemini_key,
        disabled=not has_gemini_key,
        help=('Revisa la tabla dos veces (300 y 400 DPI), corrige códigos '
              'confusos, recupera filas que el OCR no detectó y valida celdas '
              'dudosas. Se recomienda dejarlo activo para máxima precisión.'),
    )
    if not has_gemini_key:
        st.caption('⚠️ El asistente de IA no está configurado en este servidor.')
    if has_gemini_key and st.button('🔎 Probar IA', width='stretch',
                                    help='Hace una llamada mínima a Gemini para confirmar que la API key funciona.'):
        ok_ping, detail = extract_mod.gemini_ping()
        if ok_ping:
            st.success(f'✅ IA respondiendo. {detail}')
        else:
            st.error(f'❌ IA no responde. {detail}')

    with st.expander('🔧 Diagnóstico del entorno', expanded=True):
        _versions = _runtime_versions()
        for label, value in _versions.items():
            st.markdown(f'**{label}:** `{value}`')
        # También lo enviamos a stderr para poder verlo desde los logs del
        # servidor sin necesidad de abrir el expander en la UI.
        if not st.session_state.get('_diag_logged'):
            import sys as _sys_diag
            for _label, _value in _versions.items():
                print(f'[diag] {_label}: {_value}', file=_sys_diag.stderr, flush=True)
            st.session_state['_diag_logged'] = True

    st.header('2. Base SAFISS')
    external_file = st.file_uploader(
        'Excel o CSV a comparar',
        type=['csv', 'xlsx', 'xls'],
        help='Debe tener el mismo formato que dataset.xlsx (mismas columnas)',
    )

    st.header('3. Ejecutar')
    _partial_rows_here = (
        st.session_state.get('extraction_partial_rows') or _load_checkpoint()
    )
    if _partial_rows_here:
        partial_count = len({
            r.get('archivo') for r in _partial_rows_here if r.get('archivo')
        })
        st.warning(
            f'♻️ Hay una corrida interrumpida con {partial_count} archivo(s) '
            'ya procesados. Al presionar **Procesar y comparar** se retoma '
            'desde ahí; usá **Descartar** si querés empezar limpio.'
        )
        try:
            partial_bytes = to_excel_bytes({
                'parcial': order_columns(pd.DataFrame(_partial_rows_here)),
            })
            st.download_button(
                '⬇️ Descargar filas parciales (Excel)',
                data=partial_bytes,
                file_name='argos_parcial.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                width='stretch',
                help='Salvavidas: descargá lo extraído hasta ahora, sin esperar '
                     'a que termine la corrida completa.',
            )
        except Exception as _exc:
            st.caption(f'No se pudo armar el Excel parcial: {_exc}')
        if st.button('🗑️ Descartar corrida anterior', width='stretch'):
            st.session_state.pop('extraction_partial_rows', None)
            _clear_checkpoint()
            st.rerun()
    run = st.button('Procesar y comparar', type='primary', width='stretch')


def _stream_extraction(source_dir: Path, limit: int | None = None,
                       use_gemini: bool = False,
                       externa: pd.DataFrame | None = None) -> pd.DataFrame | None:
    files = list_input_files(source_dir)
    if not files:
        st.error(f'No hay PDFs ni imágenes en {source_dir}')
        return None
    extract_mod.reset_gemini_stats()

    total_files = len(files)
    effective = min(limit, total_files) if limit else total_files
    if limit and limit < total_files:
        st.caption(f'🔢 Procesando **{effective}** de {total_files} archivos (límite activo).')
    if use_gemini:
        st.caption('🤖 Gemini AI activado para celdas difíciles.')

    live_result = st.empty()
    with live_result.container():
        st.subheader('📊 Resultados')
        status = st.empty()
        progress = st.progress(0, text=f'0 / {effective}')
        metric_columns = st.columns(4)
        files_metric = metric_columns[0].empty()
        docs_metric = metric_columns[1].empty()
        ok_metric = metric_columns[2].empty()
        issues_metric = metric_columns[3].empty()
        files_metric.metric('📂 Archivos procesados', 0)
        docs_metric.metric('📄 Documentos únicos', 0)
        if externa is not None:
            ok_metric.metric('✅ Documentos OK', 0)
            issues_metric.metric('⚠️ Con inconsistencias', 0)
        else:
            ok_metric.metric('📦 Ítems extraídos', 0)
            issues_metric.metric('⚠️ Con inconsistencias', '—')
        st.markdown('#### Resumen por requisición')
        summary_slot = st.empty()
        st.markdown('#### Detalle por ítem')
        detail_slot = st.empty()

    import gc

    # Reanudación: si una corrida previa se cayó por OOM, no re-procesamos los
    # archivos que ya tenían filas extraídas. El usuario ve los KPIs desde 0
    # pero el trabajo real (OCR + IA) no se repite. Primero probamos con el
    # session_state (rerun sin crash) y como fallback con el checkpoint en
    # disco (sobrevive OOM/reboot del container en la mayoría de los casos).
    already_done = set()
    rows: list[dict] = list(st.session_state.get('extraction_partial_rows', []))
    if not rows:
        rows = _load_checkpoint()
    if rows:
        already_done = {r.get('archivo') for r in rows if r.get('archivo')}
        st.info(
            f'♻️ Retomando corrida anterior: {len(already_done)} archivo(s) ya procesados, '
            f'{len(rows)} filas ya extraídas. Se saltean.'
        )
    # `compare()` es lento cuando corre en cada iteración; para lotes grandes
    # actualizamos el detalle cada `refresh_every` archivos y siempre al final.
    refresh_every = 1 if effective <= 20 else 5

    def _refresh_display(current_rows: list[dict]) -> None:
        if not current_rows:
            return
        df_partial = order_columns(pd.DataFrame(current_rows))
        unique_docs = (
            df_partial['no_doc'].map(_clean_id).replace('', pd.NA).nunique()
            if 'no_doc' in df_partial.columns else 0
        )
        docs_metric.metric('📄 Documentos únicos', unique_docs)

        if externa is not None and all(key in df_partial.columns for key in KEYS):
            partial_merged = compare(externa.copy(), df_partial.copy())
            partial_summary = resumen_por_documento(partial_merged).copy()
            docs_ok = int((partial_summary['ok'] == partial_summary['items']).sum())
            docs_issues = int(len(partial_summary) - docs_ok)
            ok_metric.metric('✅ Documentos OK', docs_ok)
            issues_metric.metric('⚠️ Con inconsistencias', docs_issues)
            summary_slot.dataframe(
                partial_summary,
                width='stretch',
                hide_index=True,
                height=min(240, 60 + 35 * len(partial_summary)),
                column_config=COLUMN_LABELS,
            )
            partial_display = partial_merged.copy()
            partial_display['estado'] = partial_display['estado'].map(estado_label)
            detail_slot.dataframe(
                partial_display,
                width='stretch',
                hide_index=True,
                height=420,
                column_config=COLUMN_LABELS,
            )
        else:
            ok_metric.metric('📦 Ítems extraídos', len(df_partial))
            if externa is not None:
                issues_metric.metric('⚠️ Con inconsistencias', 0)
            else:
                issues_metric.metric('⚠️ Con inconsistencias', '—')
            summary_columns = [
                column for column in ('archivo', 'no_doc')
                if column in df_partial.columns
            ]
            partial_summary = df_partial[summary_columns].drop_duplicates()
            summary_slot.dataframe(
                partial_summary,
                width='stretch',
                hide_index=True,
                height=min(240, 60 + 35 * len(partial_summary)),
                column_config=COLUMN_LABELS,
            )
            detail_slot.dataframe(
                df_partial,
                width='stretch',
                hide_index=True,
                height=420,
                column_config=COLUMN_LABELS,
            )

    # Render inicial cuando reanudamos: si no lo hacemos, los KPIs y tablas
    # quedan en cero durante toda la fase de skips y el usuario no ve el
    # trabajo hecho hasta que se procese algún archivo nuevo.
    if rows:
        _refresh_display(rows)

    processed = 0
    for _, _, f, new_rows in iter_extract_folder(
        source_dir, use_gemini=use_gemini, skip_names=already_done,
    ):
        processed += 1
        progress.progress(processed / effective, text=f'{processed} / {effective}')
        files_metric.metric('📂 Archivos procesados', processed)
        if f.name in already_done:
            status.info(f'⏭️ [{processed}/{effective}] {f.name} — saltado (ya procesado)')
            if processed >= effective:
                break
            continue
        rows.extend(new_rows)
        # Persistimos en session_state (rápido, en RAM) Y en /tmp (sobrevive
        # reinicios del container) después de cada archivo, para que si el
        # container muere por OOM podamos retomar sin perder todo el trabajo.
        st.session_state['extraction_partial_rows'] = rows
        _save_checkpoint(rows)
        # Liberamos el buffer local de la iteración y forzamos GC: el pipeline
        # OCR deja imágenes grandes (300/400 DPI) que sin esto tardan varios
        # segundos en ser recolectadas por ciclos.
        del new_rows
        gc.collect()
        status.info(f'📄 [{processed}/{effective}] {f.name} — {len(rows)} filas extraídas hasta ahora')
        is_last = processed >= effective
        should_refresh = is_last or processed % refresh_every == 0
        if should_refresh:
            _refresh_display(rows)
            gc.collect()
        if is_last:
            break

    live_result.empty()
    # La corrida terminó completa: descartamos el checkpoint (RAM + disco)
    # para que la próxima ejecución arranque limpia y no ofrezca retomar.
    st.session_state.pop('extraction_partial_rows', None)
    _clear_checkpoint()
    st.toast(f'✅ Extracción lista: {processed} archivos, {len(rows)} filas', icon='✅')
    if use_gemini:
        stats = extract_mod.GEMINI_STATS
        total_calls = stats['ok'] + stats['error']
        if total_calls == 0:
            st.warning(
                '🤖 IA activa pero no se hizo ninguna llamada a Gemini. '
                'Puede que los OCR locales no hayan encontrado celdas dudosas.'
            )
        elif stats['error']:
            st.warning(
                f"🤖 IA: {stats['ok']} llamadas OK, **{stats['error']} con error**. "
                'Revisá los logs del servidor (Manage app → Logs) para el detalle.'
            )
        else:
            st.info(f"🤖 IA: {stats['ok']} llamadas exitosas, 0 errores.")
    result = order_columns(pd.DataFrame(rows)) if rows else pd.DataFrame()
    if (externa is None and not result.empty
            and 'no_doc_coincide_archivo' in result.columns):
        mismatches = result[result['no_doc_coincide_archivo'].eq(False)]
        if not mismatches.empty:
            columns = [
                column for column in
                ('archivo', 'no_doc_archivo', 'no_doc', 'no_doc_fuente')
                if column in mismatches.columns
            ]
            pairs = mismatches[columns].drop_duplicates()
            st.warning(
                f'⚠️ {len(pairs)} archivo(s) tienen un número distinto entre '
                'el nombre y el documento impreso.'
            )
            with st.expander(f'Ver detalle de números distintos ({len(pairs)})'):
                st.dataframe(
                    pairs,
                    width='stretch',
                    hide_index=True,
                    height=min(420, 55 + 35 * len(pairs)),
                    column_config=COLUMN_LABELS,
                )
    if not result.empty and 'no_doc_coincide_ocr' in result.columns:
        disagreements = result[result['no_doc_coincide_ocr'].eq(False)]
        if not disagreements.empty:
            columns = [
                column for column in
                ('archivo', 'no_doc_ocr_local', 'no_doc_gemini', 'no_doc', 'no_doc_fuente')
                if column in disagreements.columns
            ]
            pairs = disagreements[columns].drop_duplicates()
            st.warning(
                f'⚠️ Gemini y el OCR local discrepan en {len(pairs)} archivo(s).'
            )
            with st.expander(f'Ver detalle de discrepancias OCR/Gemini ({len(pairs)})'):
                st.dataframe(
                    pairs,
                    width='stretch',
                    hide_index=True,
                    height=min(520, 55 + 35 * len(pairs)),
                    column_config=COLUMN_LABELS,
                )
    return result


def run_extraction(externa: pd.DataFrame | None = None) -> pd.DataFrame | None:
    lim = int(file_limit) if limit_enabled else None
    if src_mode == 'Carpeta local':
        p = Path(folder_path or '')
        if not p.exists():
            st.error(f'Carpeta no existe: {p}')
            return None
        st.session_state['source_folder'] = str(p)
        return _stream_extraction(
            p, limit=lim, use_gemini=use_gemini, externa=externa,
        )
    st.session_state.pop('source_folder', None)
    if not uploaded_files:
        st.error('Subí al menos un archivo escaneado')
        return None
    uploaded_file_data: dict[str, bytes] = {}
    zip_count = 0
    for uploaded in uploaded_files:
        if uploaded.name.lower().endswith('.zip'):
            zip_count += 1
            try:
                with zipfile.ZipFile(io.BytesIO(uploaded.getbuffer())) as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        base_name = Path(info.filename).name
                        if not base_name or base_name.startswith('.'):
                            continue
                        if Path(base_name).suffix.lower() not in SUPPORTED_SCAN_EXTS:
                            continue
                        # Ante colisión de nombre (misma requisición en dos ZIPs
                        # o dentro de subcarpetas homónimas), la primera copia
                        # que llegue gana; las siguientes se descartan.
                        if base_name in uploaded_file_data:
                            continue
                        uploaded_file_data[base_name] = zf.read(info)
            except zipfile.BadZipFile:
                st.error(f'`{uploaded.name}` no es un ZIP válido.')
                return None
        else:
            uploaded_file_data[uploaded.name] = bytes(uploaded.getbuffer())
    if not uploaded_file_data:
        st.error('No se encontraron PDFs ni imágenes en los archivos subidos.')
        return None
    if zip_count:
        st.caption(
            f'📦 {zip_count} ZIP(s) descomprimido(s) → '
            f'{len(uploaded_file_data)} archivo(s) listos para procesar.'
        )
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)
        for name, data in uploaded_file_data.items():
            (tp / name).write_bytes(data)
        # Liberamos los bytes de RAM en cuanto están en disco. En lotes de
        # 100+ archivos, mantener el dict cargado consume cientos de MB en
        # el container de Streamlit Community Cloud (~1 GB de límite).
        uploaded_file_data.clear()
        del uploaded_file_data
        return _stream_extraction(
            tp, limit=lim, use_gemini=use_gemini, externa=externa,
        )


def reprocess_documents_with_gemini(selected_docs: list[str]) -> tuple[int, list[str]]:
    """Reprocesa con Gemini los archivos asociados y sustituye sólo sus filas."""
    escaneada = st.session_state.get('escaneada')
    externa = st.session_state.get('externa')
    if escaneada is None or externa is None or 'archivo' not in escaneada.columns:
        return 0, ['No están disponibles los datos necesarios para reprocesar.']

    selected_ids = {_clean_id(doc) for doc in selected_docs}
    selected_rows = escaneada[
        escaneada['no_doc'].map(_clean_id).isin(selected_ids)
    ]
    file_names = selected_rows['archivo'].dropna().astype(str).unique().tolist()
    if not file_names:
        return 0, ['No se encontraron archivos para los documentos seleccionados.']

    source_folder = st.session_state.get('source_folder')
    status = st.empty()
    progress = st.progress(0, text=f'0 / {len(file_names)}')
    replacement_rows: list[dict] = []
    replaced_files: list[str] = []
    failures: list[str] = []

    with tempfile.TemporaryDirectory() as td:
        temporary_folder = Path(td)
        for index, file_name in enumerate(file_names, start=1):
            status.info(
                f'🤖 Reprocesando [{index}/{len(file_names)}] {file_name} con Gemini'
            )
            if source_folder:
                file_path = Path(source_folder) / file_name
            else:
                # Cuando la fuente fue upload no guardamos los bytes en
                # session_state (ahorro de RAM), así que reprocesar con
                # Gemini solo aplica al modo "Carpeta local".
                failures.append(
                    f'{file_name}: reprocesar con Gemini solo está disponible '
                    'si la fuente fue una carpeta local.'
                )
                progress.progress(index / len(file_names), text=f'{index} / {len(file_names)}')
                continue

            if not file_path.exists():
                failures.append(f'{file_name}: archivo no encontrado')
                progress.progress(index / len(file_names), text=f'{index} / {len(file_names)}')
                continue

            new_rows = process_pdf(file_path, use_gemini=True)
            valid_rows = [row for row in new_rows if row.get('codigo')]
            if not valid_rows:
                error = new_rows[0].get('error', 'sin filas válidas') if new_rows else 'sin resultados'
                failures.append(f'{file_name}: {error}')
            else:
                replacement_rows.extend(new_rows)
                replaced_files.append(file_name)
            progress.progress(index / len(file_names), text=f'{index} / {len(file_names)}')

    status.empty()
    progress.empty()
    if not replaced_files:
        return 0, failures

    preserved = escaneada[~escaneada['archivo'].astype(str).isin(replaced_files)]
    updated = order_columns(pd.concat(
        [preserved, pd.DataFrame(replacement_rows)], ignore_index=True,
    ))
    st.session_state['escaneada'] = updated
    st.session_state['merged'] = compare(externa.copy(), updated.copy())
    return len(replaced_files), failures


if run:
    for key in ('escaneada', 'externa', 'merged', 'safiss_range'):
        st.session_state.pop(key, None)

    externa = None
    if external_file is not None:
        try:
            externa = load_external(external_file)
            missing_keys = [key for key in KEYS if key not in externa.columns]
            if missing_keys:
                raise ValueError(
                    'faltan las columnas requeridas: ' + ', '.join(missing_keys)
                )
        except Exception as e:
            st.error(f'Base SAFISS no válida: {e}')
            st.stop()

    escaneada = run_extraction(externa=externa)
    if escaneada is None or escaneada.empty:
        st.warning('No se extrajo nada.')
        st.stop()
    st.session_state['escaneada'] = escaneada
    if externa is not None:
        try:
            st.session_state['externa'] = externa
            st.session_state['safiss_range'] = _safiss_date_range(externa)
            st.session_state['merged'] = compare(externa.copy(), escaneada.copy())
        except Exception as e:
            st.error(f'Error comparando: {e}')
    else:
        st.info('Subí una base externa si querés comparar.')


if 'merged' in st.session_state:
    merged = st.session_state['merged']
    tab_cmp, tab_esc, tab_ext = st.tabs(['Comparación', 'Requisiciones escaneadas', 'Base SAFISS'])

    with tab_cmp:
        resumen = resumen_por_documento(merged).copy()

        escaneada_actual = st.session_state['escaneada']
        doc_number_mismatches = pd.DataFrame()
        if 'no_doc_coincide_archivo' in escaneada_actual.columns:
            doc_number_mismatches = escaneada_actual[
                escaneada_actual['no_doc_coincide_archivo'].eq(False)
            ].copy()
            if not doc_number_mismatches.empty:
                doc_number_mismatches = doc_number_mismatches[
                    ['no_doc', 'no_doc_archivo', 'archivo', 'no_doc_fuente']
                ].drop_duplicates(subset=['no_doc'])

        rng = st.session_state.get('safiss_range')
        fechas_docs = _fecha_por_doc(st.session_state['escaneada']) if rng else {}
        docs_fuera_rango: list[tuple[str, str]] = []
        if rng and fechas_docs:
            lo, hi = rng
            # El Excel puede comenzar con movimientos varios días después de
            # la requisición. La cobertura se evalúa por mes contable, no contra
            # el primer/ultimo día exacto presente en SAFISS.
            coverage_start = lo.to_period('M').start_time
            coverage_end = hi.to_period('M').end_time
            for i, row in resumen.iterrows():
                document_range = fechas_docs.get(_clean_id(row['no_doc']))
                doc_rows = merged[
                    merged['no_doc'].map(_clean_id) == _clean_id(row['no_doc'])
                ]
                has_safiss_rows = (
                    'cant_despachada_ext' in doc_rows.columns
                    and doc_rows['cant_despachada_ext'].notna().any()
                )
                if document_range is None or has_safiss_rows:
                    continue
                doc_start, doc_end, date_label = document_range
                if doc_end < coverage_start or doc_start > coverage_end:
                    resumen.at[i, 'diagnóstico'] = 'fuera de rango SAFISS'
                    docs_fuera_rango.append((row['no_doc'], date_label))

        mismatch_docs = {
            _clean_id(value) for value in doc_number_mismatches.get('no_doc', [])
        }
        if mismatch_docs:
            for i, row in resumen.iterrows():
                if _clean_id(row['no_doc']) not in mismatch_docs:
                    continue
                current = str(row['diagnóstico'])
                resumen.at[i, 'diagnóstico'] = (
                    f'{current}; número de archivo ≠ escaneo'
                )

        detail_merged = merged.copy()
        out_of_range_doc_ids = {
            _clean_id(doc) for doc, _ in docs_fuera_rango
        }
        if out_of_range_doc_ids:
            detail_merged.loc[
                detail_merged['no_doc'].map(_clean_id).isin(out_of_range_doc_ids),
                'estado',
            ] = 'fuera_rango'

        total_docs = len(resumen)
        docs_ok = (resumen['ok'] == resumen['items']).sum()
        docs_out = resumen['diagnóstico'].str.contains('fuera de rango SAFISS', na=False).sum()
        docs_mismatch = len(mismatch_docs)
        docs_bad = total_docs - docs_ok - docs_out
        docs_diff = resumen['diagnóstico'].str.contains('cantidad', na=False).sum()
        docs_faltan_safiss = resumen['diagnóstico'].str.contains('faltan en SAFISS', na=False).sum()
        docs_faltan_esc = resumen['diagnóstico'].str.contains('faltan escaneadas', na=False).sum()

        file_columns = [
            column for column in
            ('archivo', 'archivo_hash', 'no_doc', 'no_doc_archivo', 'no_doc_fuente')
            if column in escaneada_actual.columns
        ]
        file_docs = escaneada_actual[file_columns].drop_duplicates(
            subset=['archivo']
        ).sort_values(['no_doc', 'archivo'])
        files_processed = len(file_docs)
        # Dos escaneos del mismo documento pueden tener hashes distintos por
        # compresión o metadatos; el no_doc identifica la copia semántica.
        duplicate_key = 'no_doc'
        file_counts = file_docs.groupby(duplicate_key)['archivo'].transform('count')
        duplicate_files = file_docs[file_counts > 1].copy()
        if not duplicate_files.empty:
            duplicate_files['copias'] = file_counts[file_counts > 1].values
        files_per_copy = file_docs.dropna(subset=[duplicate_key]).groupby(duplicate_key).size()
        duplicate_copies = int((files_per_copy - 1).clip(lower=0).sum())

        def kpi_detail(column, key: str, data: pd.DataFrame, empty_message: str):
            with column.popover('Ver detalle', width='stretch', key=key):
                if data.empty:
                    st.info(empty_message)
                else:
                    st.dataframe(
                        data,
                        width='stretch',
                        hide_index=True,
                        height=min(420, 55 + 35 * len(data)),
                        column_config=COLUMN_LABELS,
                    )

        o1, o2, o3, o4 = st.columns(4)
        o1.metric('📂 Archivos procesados', files_processed,
                  help='Cantidad de archivos PDF o imagen que se procesaron')

        o2.metric('📄 Documentos únicos', total_docs,
                  help='Cantidad de no_doc distintos encontrados dentro de los escaneos')

        o3.metric('📑 Copias duplicadas', max(0, duplicate_copies),
                  help='Archivos adicionales asociados a un no_doc ya procesado')

        ok_documents = resumen[resumen['ok'] == resumen['items']]
        o4.metric('✅ Documentos OK', int(docs_ok),
                  delta=f"{docs_ok/total_docs:.0%}" if total_docs else None,
                  help='no_doc cuyos ítems despachados coinciden 100% con SAFISS')

        d1, d2, d3, d4 = st.columns(4)
        kpi_detail(d1, 'detail_files', file_docs, 'No se procesaron archivos.')
        kpi_detail(d2, 'detail_unique_docs', resumen,
                   'No se identificaron documentos.')
        kpi_detail(d3, 'detail_duplicates', duplicate_files,
                   'No se encontraron documentos duplicados.')
        kpi_detail(d4, 'detail_ok_docs', ok_documents,
                   'No hay documentos completamente coincidentes.')

        s1, s2, s3, s4, s5 = st.columns(5)
        diff_documents = resumen[
            resumen['diagnóstico'].str.contains('cantidad', na=False)
        ]
        s1.metric('❌ Diferencias en cantidades', int(docs_diff),
                  delta=f"-{docs_diff}" if docs_diff else None, delta_color='inverse')

        missing_safiss = resumen[
            resumen['diagnóstico'].str.contains('faltan en SAFISS', na=False)
        ]
        s2.metric('📄 Faltantes en SAFISS', int(docs_faltan_safiss),
                  delta=f"-{docs_faltan_safiss}" if docs_faltan_safiss else None,
                  delta_color='inverse')

        missing_scan = resumen[
            resumen['diagnóstico'].str.contains('faltan escaneadas', na=False)
        ]
        s3.metric('⚠️ Faltantes en escaneo', int(docs_faltan_esc),
                  delta=f"-{docs_faltan_esc}" if docs_faltan_esc else None,
                  delta_color='inverse')

        out_documents = resumen[
            resumen['diagnóstico'].str.contains('fuera de rango SAFISS', na=False)
        ]
        s4.metric('📅 Fuera de rango SAFISS', int(docs_out),
                  delta=f"-{docs_out}" if docs_out else None, delta_color='inverse')

        corrected_documents = resumen[
            resumen['diagnóstico'].str.contains('número de archivo ≠ escaneo', na=False)
        ]
        s5.metric('🪪 Número corregido', int(docs_mismatch),
                  help='documentos cuyo número impreso no coincide con el nombre del archivo')

        t1, t2, t3, t4, t5 = st.columns(5)
        kpi_detail(t1, 'detail_quantity_diff', diff_documents,
                   'No hay diferencias de cantidades.')
        kpi_detail(t2, 'detail_missing_safiss', missing_safiss,
                   'No hay faltantes en SAFISS.')
        kpi_detail(t3, 'detail_missing_scan', missing_scan,
                   'No hay faltantes en los escaneos.')
        kpi_detail(t4, 'detail_out_of_range', out_documents,
                   'No hay documentos fuera del rango SAFISS.')
        kpi_detail(t5, 'detail_corrected_number', corrected_documents,
                   'No hubo números de documento corregidos.')

        total_items = len(detail_merged)
        docs_comparables = total_docs - docs_out
        if docs_bad == 0 and docs_comparables > 0:
            st.success(f'✅ Los {docs_comparables} documentos comparables coinciden con SAFISS ({total_items} ítems).')
        elif docs_comparables > 0:
            st.warning(f'⚠️ {docs_bad} de {docs_comparables} documentos comparables con inconsistencias. Total: {total_items} ítems comparados.')

        if docs_fuera_rango:
            lo, hi = rng
            st.info(
                f'📅 **{docs_out} documento(s) fuera del rango de SAFISS** '
                f'(cobertura: {lo:%m.%Y} – {hi:%m.%Y}).'
            )
            out_of_range_detail = pd.DataFrame(
                docs_fuera_rango, columns=['no_doc', 'fecha_documento'],
            )
            with st.expander(f'Ver documentos fuera de rango ({docs_out})'):
                st.dataframe(
                    out_of_range_detail,
                    width='stretch',
                    hide_index=True,
                    height=min(520, 55 + 35 * len(out_of_range_detail)),
                    column_config={
                        **COLUMN_LABELS,
                        'fecha_documento': 'Fecha del documento',
                    },
                )

        if not doc_number_mismatches.empty:
            st.info(
                f'🪪 **{docs_mismatch} documento(s) con número corregido desde el escaneo.**'
            )
            with st.expander(f'Ver números corregidos ({docs_mismatch})'):
                st.dataframe(
                    doc_number_mismatches,
                    width='stretch',
                    hide_index=True,
                    height=min(520, 55 + 35 * len(doc_number_mismatches)),
                    column_config=COLUMN_LABELS,
                )

        st.subheader('Resumen por requisición')
        st.dataframe(resumen, width='stretch', hide_index=True,
                     height=min(240, 60 + 35 * len(resumen)),
                     column_config=COLUMN_LABELS)

        feedback = st.session_state.pop('gemini_reprocess_feedback', None)
        if feedback:
            replaced_count, previous_failures = feedback
            st.success(
                f'✅ Gemini reprocesó {replaced_count} archivo(s) y actualizó la comparación.'
            )
            if previous_failures:
                st.warning('⚠️ ' + ' | '.join(previous_failures))

        reprocessable = resumen[
            (resumen['con_problemas'] > 0)
            & ~resumen['diagnóstico'].str.contains('fuera de rango SAFISS', na=False)
        ].copy()
        if not reprocessable.empty:
            with st.container(border=True):
                st.markdown('#### 🤖 Verificar inconsistencias con Gemini')
                st.caption(
                    'Seleccioná uno o varios documentos. ARGOS reprocesará sólo '
                    'sus archivos y volverá a compararlos con SAFISS.'
                )
                diagnostic_by_doc = {
                    _clean_id(row['no_doc']): str(row['diagnóstico'])
                    for _, row in reprocessable.iterrows()
                }
                reprocess_options = list(diagnostic_by_doc)
                reprocess_generation = st.session_state.get(
                    'gemini_reprocess_generation', 0,
                )
                selected_for_reprocess = st.multiselect(
                    'Documentos con inconsistencias',
                    reprocess_options,
                    format_func=lambda doc: f'{doc} — {diagnostic_by_doc[doc]}',
                    key=f'gemini_reprocess_docs_{reprocess_generation}',
                    placeholder='Seleccioná los documentos a verificar',
                )
                reprocess_button = st.button(
                    '🤖 Reprocesar seleccionados con Gemini',
                    type='primary',
                    width='stretch',
                    disabled=not selected_for_reprocess or not has_gemini_key,
                    key=f'gemini_reprocess_button_{reprocess_generation}',
                )
                if not has_gemini_key:
                    st.warning(
                        'Configurá `GOOGLE_API_KEY` para habilitar el reproceso con Gemini.'
                    )
                if reprocess_button:
                    replaced_count, failures = reprocess_documents_with_gemini(
                        selected_for_reprocess
                    )
                    if replaced_count:
                        st.session_state['gemini_reprocess_feedback'] = (
                            replaced_count, failures,
                        )
                        st.session_state['gemini_reprocess_generation'] = (
                            reprocess_generation + 1
                        )
                        st.rerun()
                    else:
                        st.error(
                            'No se pudo reprocesar ningún archivo. '
                            + (' | '.join(failures) if failures else '')
                        )

        source_folder = st.session_state.get('source_folder')
        escaneada_df = st.session_state.get('escaneada')
        if source_folder and escaneada_df is not None and 'archivo' in escaneada_df.columns:
            archivos_docs = (escaneada_df[['no_doc', 'archivo']]
                             .dropna().drop_duplicates()
                             .sort_values('no_doc'))
            if not archivos_docs.empty:
                opts = {f"{r['no_doc']} — {r['archivo']}": r['archivo']
                        for _, r in archivos_docs.iterrows()}
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    sel = st.selectbox('📂 Abrir PDF de una requisición',
                                       list(opts.keys()), key='pdf_open_sel')
                with col_b:
                    st.write('')
                    if st.button('Abrir', width='stretch', key='pdf_open_btn'):
                        path = Path(source_folder) / opts[sel]
                        if not path.exists():
                            st.error(f'Archivo no encontrado: {path}')
                        else:
                            ok, err = _open_file(path)
                            if ok:
                                st.toast(f'Abriendo {path.name}', icon='📂')
                            else:
                                st.error(f'No se pudo abrir: {err}')
        elif escaneada_df is not None and 'archivo' in escaneada_df.columns:
            st.caption('💡 Para abrir un PDF directamente, usá el modo **Carpeta local** en la barra lateral.')

        st.subheader('Detalle por ítem')
        f1, f2, f3 = st.columns([2, 1, 1])
        with f1:
            estados = ['(todos)'] + sorted(detail_merged['estado'].unique().tolist())
            estado_sel = f1.selectbox('Filtrar por estado', estados,
                                      format_func=lambda e: '(todos)' if e == '(todos)' else estado_label(e))
        with f2:
            only_bad = st.checkbox('Solo inconsistencias', value=(docs_bad > 0))
        with f3:
            doc_filter = st.text_input('Buscar no_doc / codigo', '')

        view = detail_merged.copy()
        if estado_sel != '(todos)':
            view = view[view['estado'] == estado_sel]
        elif only_bad:
            view = view[view['estado'] != 'ok']
        if doc_filter.strip():
            q = doc_filter.strip().lower()
            view = view[view['no_doc'].astype(str).str.lower().str.contains(q)
                        | view['codigo'].astype(str).str.lower().str.contains(q)]

        st.caption(f'Mostrando **{len(view)}** de **{total_items}** ítems')
        report_data = to_excel_bytes({
            'comparacion': detail_merged,
            'resumen_por_doc': resumen,
            'escaneada': st.session_state['escaneada'],
            'externa': st.session_state['externa'],
        })
        toolbar_slot = st.empty()

        def render_detail_toolbar(document_path: Path | None = None,
                                  selected_doc: str | None = None):
            with toolbar_slot.container():
                view_button, download_button, spacer = st.columns([1.25, 1.25, 4])
                with view_button:
                    label = (f'📄 Ver documento {selected_doc}'
                             if selected_doc else '📄 Ver documento')
                    if st.button(
                        label,
                        type='primary',
                        disabled=document_path is None,
                        width='stretch',
                        key='view_selected_document',
                    ):
                        st.session_state['document_preview'] = str(document_path)
                        st.session_state['pdf_page'] = 1
                with download_button:
                    st.download_button(
                        '⬇️ Descargar reporte Excel',
                        data=report_data,
                        file_name='reporte_argos.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        width='stretch',
                    )

        if len(view) == 0:
            render_detail_toolbar()
            st.info('No hay filas que mostrar con los filtros actuales.')
        else:
            display = view.copy()
            display['estado'] = display['estado'].map(estado_label)
            table_data = display
            if len(display) > MAX_STYLED_ROWS:
                st.caption(f'⚡ Coloreado desactivado ({len(display)} filas > {MAX_STYLED_ROWS}). Usá los filtros o descargá el Excel para ver todo con formato.')
            else:
                table_data = style_merged(display)

            st.caption('Seleccioná una fila para ver su documento original.')
            table_event = st.dataframe(
                table_data,
                width='stretch',
                hide_index=True,
                height=520,
                column_config=COLUMN_LABELS,
                on_select='rerun',
                selection_mode='single-row',
                key='detalle_items_table',
            )

            selected_rows = table_event.selection.rows
            document_path = None
            selected_doc = None
            document_error = None
            if selected_rows:
                selected = view.iloc[selected_rows[0]]
                selected_doc = _clean_id(selected['no_doc'])
                source_folder = st.session_state.get('source_folder')
                escaneada_df = st.session_state.get('escaneada')

                if source_folder and escaneada_df is not None:
                    document_path = _document_path(selected_doc, source_folder, escaneada_df)
                    if document_path is None or not document_path.exists():
                        document_path = None
                        document_error = f'No se encontró el archivo del documento {selected_doc}.'
                else:
                    document_error = (
                        'La vista del documento está disponible usando el origen '
                        '**Carpeta local**.'
                    )

            render_detail_toolbar(document_path, selected_doc)
            if document_error:
                st.warning(document_error)

            preview = st.session_state.get('document_preview')
            if preview:
                preview_path = Path(preview)
                if preview_path.exists():
                    _show_document(preview_path)
                else:
                    st.session_state.pop('document_preview', None)

    with tab_esc:
        st.dataframe(st.session_state['escaneada'], width='stretch', hide_index=True,
                     column_config=COLUMN_LABELS)
    with tab_ext:
        st.dataframe(st.session_state['externa'], width='stretch', hide_index=True,
                     column_config=COLUMN_LABELS)

elif 'escaneada' in st.session_state:
    st.subheader('Dataset extraído')
    st.dataframe(st.session_state['escaneada'], width='stretch', hide_index=True,
                 column_config=COLUMN_LABELS)

else:
    st.info('Configurá origen en la barra lateral y presioná **Procesar y comparar**.')
