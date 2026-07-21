"""Extrae datos de formularios de requisición escaneados (PDF/imagen) a un dataset.

Pipeline:
1. OCR local con Tesseract (doble pasada 300 + 400 DPI).
2. Fallback local por celda con filtrado por componentes conectados (anti X-marks).
3. Enhancement opcional con Gemini AI (opt-in via `use_gemini=True` + GOOGLE_API_KEY).
"""
import argparse
import base64
import hashlib
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import fitz
import numpy as np
import pandas as pd
import pytesseract
import requests
from pytesseract import Output


# ─── Setup: carga .env y regex globales ─────────────────────────────────────
_ENV_FILE = Path(__file__).parent / '.env'
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith('#') or '=' not in _line:
            continue
        _k, _, _v = _line.partition('=')
        os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))


POSITION_TAG_RE = re.compile(r'^[\(\[]\d+[\-\u2013]?\d*[\)\]]$')
ITEM_CODE_RE = re.compile(r'^\d{6,10}$')


def is_position_tag(text: str) -> bool:
    return bool(POSITION_TAG_RE.match(text.strip()))


# ─── Carga de imagen y preprocesamiento ─────────────────────────────────────
TESS_CONFIG = '--psm 6 -c preserve_interword_spaces=1'
TESS_LANG = 'spa'
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def load_image(path: Path, dpi: int = 300) -> np.ndarray:
    if path.suffix.lower() in IMAGE_EXTS:
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f'No se pudo leer la imagen: {path}')
        return img
    doc = fitz.open(str(path))
    page = doc.load_page(0)
    zoom = dpi / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    doc.close()
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if pix.n == 3 else img


def deskew(img: np.ndarray) -> np.ndarray:
    """Los PDFs escaneados vienen bien orientados; deskew ingenuo mete ruido, así que se omite."""
    return img


MIN_OCR_WIDTH = 2000
DARK_VALUE_THRESHOLD = 170
COLOR_SATURATION_THRESHOLD = 40


def remove_colored_marks(img: np.ndarray) -> np.ndarray:
    """Blanquea píxeles claros O saturados (marcas manuales azul/violeta),
    preservando texto negro impreso. Sobrevive a X tachadas encima del texto."""
    if img.ndim != 3 or img.shape[2] < 3:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 2] > DARK_VALUE_THRESHOLD) | (hsv[:, :, 1] > COLOR_SATURATION_THRESHOLD)
    out = img.copy()
    out[mask] = (255, 255, 255)
    return out


def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    img = remove_colored_marks(img)
    h, w = img.shape[:2]
    if w < MIN_OCR_WIDTH:
        scale = MIN_OCR_WIDTH / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    line_len = max(40, img.shape[1] // 40)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_len))
    lines = cv2.add(cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel_h),
                    cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel_v))
    return cv2.bitwise_not(cv2.subtract(th, lines))


# ─── OCR primario (Tesseract) ───────────────────────────────────────────────
def run_ocr(img: np.ndarray):
    proc = preprocess_for_ocr(img)
    data = pytesseract.image_to_data(proc, lang=TESS_LANG, config=TESS_CONFIG, output_type=Output.DICT)
    items = []
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        if not text:
            continue
        try:
            conf = int(float(data['conf'][i]))
        except (ValueError, TypeError):
            conf = -1
        if conf < 30:
            continue
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        items.append({
            'text': text, 'conf': conf,
            'x': x, 'y': y, 'x2': x + w, 'y2': y + h,
            'cx': x + w / 2, 'cy': y + h / 2,
        })
    return items


# ─── Consultas espaciales sobre tokens OCR ──────────────────────────────────
def find_anchors(items, keywords, y_range=None):
    matches = []
    for it in items:
        t = it['text'].upper()
        if not all(k.upper() in t for k in keywords):
            continue
        if y_range and not (y_range[0] <= it['cy'] <= y_range[1]):
            continue
        matches.append(it)
    return matches


def find_anchor(items, keywords, y_range=None):
    m = find_anchors(items, keywords, y_range)
    return m[0] if m else None


def value_below(items, anchor, max_dy=180, max_dx=80):
    if not anchor:
        return None
    best, best_dy = None, float('inf')
    for it in items:
        if it is anchor or it['y'] < anchor['y2']:
            continue
        dy = it['y'] - anchor['y2']
        dx = abs(it['cx'] - anchor['cx'])
        if dy > max_dy or dx > max_dx:
            continue
        if is_position_tag(it['text']):
            continue
        if dy < best_dy:
            best_dy, best = dy, it
    return best['text'] if best else None


def value_right(items, anchor, max_dx=500, max_dy=20):
    if not anchor:
        return None
    best, best_dx = None, float('inf')
    for it in items:
        if it is anchor or it['x'] < anchor['x2']:
            continue
        dx = it['x'] - anchor['x2']
        dy = abs(it['cy'] - anchor['cy'])
        if dx > max_dx or dy > max_dy:
            continue
        if dx < best_dx:
            best_dx, best = dx, it
    return best['text'] if best else None


# ─── Extracción de header, no_doc y firmas ──────────────────────────────────
def extract_header(items, page_h):
    top = (0, page_h * 0.15)
    despach = sorted([it for it in items
                      if 'espach' in it['text'].lower() and it['cy'] < top[1]],
                     key=lambda x: x['cx'])
    return {
        'mov': value_below(items, find_anchor(items, ['MOV'], y_range=top)),
        'almacen': value_below(items, find_anchor(items, ['ALMACEN'], y_range=top)),
        'depend_destinataria': value_below(items, find_anchor(items, ['DEPEND'], y_range=top)),
        'lote': value_below(items, find_anchor(items, ['LOTE'], y_range=top)),
        'año': value_below(items, find_anchor(items, ['AÑO'], y_range=top))
               or value_below(items, find_anchor(items, ['ANO'], y_range=top)),
        'mes': value_below(items, find_anchor(items, ['MES'], y_range=top)),
        'total_despachada': value_below(items, despach[0]) if despach else None,
        'total_no_despachada': value_below(items, despach[1]) if len(despach) > 1 else None,
    }


NO_DOC_FILENAME_RE = re.compile(r'^(\d{6,10})')
SCANNED_DOC_RE = re.compile(r'(?<!\d)(\d{10})(?!\d)')


def no_doc_from_filename(path: Path) -> str | None:
    m = NO_DOC_FILENAME_RE.match(path.name)
    return m.group(1) if m else None


def _looks_like_doc_number(v):
    if not v:
        return False
    digits = ''.join(c for c in str(v) if c.isdigit())
    return len(digits) >= 6


def extract_doc_number(items, filename_hint: str | None = None):
    """Lee el documento del OCR general; el nombre es solamente el fallback."""
    anchor = find_anchor(items, ['DOC.'])
    val = value_right(items, anchor, max_dx=500)
    if _looks_like_doc_number(val):
        return ''.join(c for c in str(val) if c.isdigit())
    safis = find_anchor(items, ['SAFIS'])
    val = value_right(items, safis, max_dx=500) if safis else None
    if _looks_like_doc_number(val):
        return ''.join(c for c in str(val) if c.isdigit())
    return filename_hint


def extract_doc_number_from_image(img: np.ndarray) -> str | None:
    """OCR numérico dirigido a la franja inferior donde se imprime No. Doc. SAFISS."""
    h, w = img.shape[:2]
    crop = img[int(h * 0.65):h, int(w * 0.15):int(w * 0.90)]
    clean = remove_colored_marks(crop)
    gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(
        binary,
        config='--psm 11 -c tessedit_char_whitelist=0123456789',
    )
    candidates = SCANNED_DOC_RE.findall(text)
    if not candidates:
        return None
    return max(set(candidates), key=candidates.count)


def resolve_doc_number(filename_doc: str | None, local_doc: str | None,
                       gemini_doc: str | None = None) -> tuple[str | None, str]:
    """Resuelve el documento por consenso y corrige confusiones OCR de un dígito.

    El consenso entre Gemini y el OCR local prevalece cuando su lectura sigue
    siendo cercana al nombre. Una lectura sin consenso o completamente ajena
    se descarta para evitar confundir sellos, NIT y códigos con el documento.
    """
    if gemini_doc and gemini_doc == local_doc:
        if filename_doc:
            abbreviated_name = (
                gemini_doc.endswith(filename_doc)
                or filename_doc.endswith(gemini_doc)
            )
            distance = (
                sum(a != b for a, b in zip(filename_doc, gemini_doc))
                if len(filename_doc) == len(gemini_doc) else math.inf
            )
            if not abbreviated_name and distance > 2:
                return filename_doc, 'nombre archivo (descarta lectura sospechosa)'
        return gemini_doc, 'gemini + OCR local'
    # SAFISS usa el número completo (p. ej. 4931159608); el nombre de archivo y
    # a veces Gemini muestran sólo el sufijo (159608). Si la lectura local
    # completa contiene como sufijo tanto al filename como a Gemini, es la
    # versión canónica y debe prevalecer para que el cruce con SAFISS funcione.
    if (local_doc and filename_doc
            and len(local_doc) > len(filename_doc)
            and local_doc.endswith(filename_doc)
            and (gemini_doc is None or local_doc.endswith(gemini_doc))):
        source = 'OCR local (número completo)'
        if gemini_doc:
            source = 'OCR local (número completo, Gemini abreviado)'
        return local_doc, source
    if (gemini_doc and filename_doc and not local_doc
            and len(gemini_doc) > len(filename_doc)
            and gemini_doc.endswith(filename_doc)):
        return gemini_doc, 'gemini (número completo)'
    # Gemini lee la versión completa (sufijo = filename); el OCR local
    # sí devolvió algo pero no coincide con el filename. Esto suele indicar una
    # confusión OCR de 1–2 dígitos en la lectura local, y Gemini es la fuente
    # más confiable en ese escenario.
    if (gemini_doc and filename_doc and local_doc
            and len(gemini_doc) > len(filename_doc)
            and gemini_doc.endswith(filename_doc)
            and not (len(local_doc) >= len(filename_doc)
                     and local_doc.endswith(filename_doc))):
        return gemini_doc, 'gemini (número completo, OCR local incoherente)'
    if gemini_doc and gemini_doc == filename_doc:
        return gemini_doc, 'gemini + nombre archivo'
    if local_doc and local_doc == filename_doc:
        return local_doc, 'OCR local + nombre archivo'
    if gemini_doc and filename_doc:
        return filename_doc, 'nombre archivo (Gemini sin consenso)'
    if gemini_doc:
        return gemini_doc, 'gemini'
    if not local_doc:
        return filename_doc, 'nombre_archivo'

    # Si la lectura difiere del archivo por un solo dígito, normalmente es una
    # confusión OCR. Esta heurística sólo se aplica sin Gemini; con Gemini,
    # su lectura ya se devolvió arriba y no debe ser anulada por el nombre.
    if filename_doc and len(filename_doc) == len(local_doc):
        distance = sum(a != b for a, b in zip(filename_doc, local_doc))
        if distance == 1:
            return filename_doc, 'nombre archivo (corrige OCR)'
    return local_doc, 'OCR local'


SIG_NOISE = ('NOMBRE', 'FIRMA', 'FECHA', 'INFORMACION', 'DOC', 'DOCUMENTOS',
             'ELABORADO', 'SAFIS', 'GENERADOS')


def line_below(items, anchor, max_dy=140, x_left=-40, x_right=440, line_tol=25):
    if not anchor:
        return None
    x_lo, x_hi = anchor['x'] + x_left, anchor['x'] + x_right
    candidates = []
    for it in items:
        if it['y'] <= anchor['y2']:
            continue
        if it['cx'] < x_lo or it['cx'] > x_hi:
            continue
        dy = it['y'] - anchor['y2']
        if dy > max_dy:
            continue
        t = it['text'].upper().strip(' :.')
        if any(n in t for n in SIG_NOISE) or t in ('POR', ''):
            continue
        candidates.append(it)
    if not candidates:
        return None
    candidates.sort(key=lambda x: x['y'])
    y0 = candidates[0]['y']
    line = sorted([c for c in candidates if abs(c['y'] - y0) < line_tol], key=lambda x: x['x'])
    return ' '.join(c['text'] for c in line).strip(' :') or None


def line_right(items, anchor, max_dx=600, line_tol=25):
    if not anchor:
        return None
    same_line = []
    for it in items:
        if it is anchor or it['x'] < anchor['x2']:
            continue
        if abs(it['cy'] - anchor['cy']) > line_tol:
            continue
        if it['x'] - anchor['x2'] > max_dx:
            continue
        t = it['text'].strip(' :.')
        if not t or t in (':', '-'):
            continue
        same_line.append(it)
    if not same_line:
        return None
    same_line.sort(key=lambda x: x['x'])
    return ' '.join(c['text'].strip(':') for c in same_line).strip() or None


def extract_signatures(items):
    result = {}
    for key, labels in [('solicito', ['SOLICITO']), ('autorizo', ['AUTORIZO']),
                        ('recibio', ['RECIBIO']), ('preparado_por', ['PREPARADO']),
                        ('comprobado_por', ['COMPROBADO'])]:
        anchor = find_anchor(items, labels)
        result[key] = line_below(items, anchor) if anchor else None
    elab_anchor = find_anchor(items, ['ELABORADO'])
    result['elaborado'] = line_right(items, elab_anchor) if elab_anchor else None
    return result


# ─── Detección de columnas y extracción de filas ────────────────────────────
DEFAULT_COLS = {
    'idem':            (0.00, 0.10),
    'codigo':          (0.10, 0.25),
    'nombre_producto': (0.25, 0.60),
    'presentacion':    (0.60, 0.72),
    'cant_solicitada': (0.72, 0.86),
    'cant_despachada': (0.86, 1.00),
}


def detect_columns(items, page_w):
    """Infer column x-boundaries from the item-table header row."""
    nombres = [it for it in items
               if 'NOMBRE' in it['text'].upper() and 'PRODUCTO' not in it['text'].upper()]
    if not nombres:
        nombres = [it for it in items if it['text'].upper().startswith('NOMBRE')]
    nombre = None
    for n in sorted(nombres, key=lambda x: x['cy']):
        near_producto = any('PRODUCTO' in it['text'].upper() and abs(it['cy'] - n['cy']) < 40
                            for it in items)
        if near_producto:
            nombre = n
            break
    if not nombre:
        return DEFAULT_COLS

    y_anchor = nombre['cy']
    def _near(needles, exclude=()):
        matches = [it for it in items
                   if any(n in it['text'].upper() for n in needles)
                   and not any(e in it['text'].upper() for e in exclude)
                   and abs(it['cy'] - y_anchor) < 80]
        return matches[0] if matches else None

    codigo = _near(['CODIGO', 'ÓDIGO'], exclude=['CODIGOS'])
    presen = _near(['PRESEN'])
    solic  = _near(['SOLICITAD'])
    despa  = _near(['DESPACHAD'])

    cantidades = sorted([it for it in items
                         if 'CANTIDAD' in it['text'].upper()
                         and abs(it['cy'] - y_anchor) < 80],
                        key=lambda x: x['cx'])
    if not solic and cantidades:
        solic = cantidades[0]
    if not despa and len(cantidades) >= 2:
        despa = cantidades[-1]
    elif not despa and solic:
        despa_x = solic['cx'] + (page_w - solic['cx']) * 0.5
        despa = {'cx': despa_x}

    if not codigo:
        return DEFAULT_COLS

    centers = {'codigo': codigo['cx'], 'nombre_producto': nombre['cx']}
    if presen:
        centers['presentacion'] = presen['cx']
    if solic:
        centers['cant_solicitada'] = solic['cx']
    if despa:
        centers['cant_despachada'] = despa['cx']

    ordered = sorted(centers.items(), key=lambda kv: kv[1])
    fractions = [(k, v / page_w) for k, v in ordered]
    cols = {'idem': (0.0, max(0.02, fractions[0][1] - 0.05))}
    prev_hi = cols['idem'][1]
    for i, (k, cx) in enumerate(fractions):
        if i + 1 < len(fractions):
            hi = (cx + fractions[i + 1][1]) / 2
        else:
            hi = 1.0
        cols[k] = (prev_hi, hi)
        prev_hi = hi
    if 'cant_despachada' in cols:
        lo, hi = cols['cant_despachada']
        cols['cant_despachada'] = (lo, min(hi, lo + (hi - lo) * 0.75))
    return cols


def extract_items(items, page_w, page_h):
    doc_marker = find_anchor(items, ['DOC.'], y_range=(page_h * 0.5, page_h))
    y_end = doc_marker['y'] - 10 if doc_marker else page_h * 0.85
    header_end = max((i['cy'] for i in items if is_position_tag(i['text'])), default=page_h * 0.05)

    cols = detect_columns(items, page_w)
    codigo_lo, codigo_hi = cols.get('codigo', (0.06, 0.25))
    codes = [it for it in items
             if ITEM_CODE_RE.match(it['text'])
             and codigo_lo <= it['cx'] / page_w < codigo_hi + 0.03
             and header_end + 30 < it['cy'] < y_end]
    if not codes:
        return []
    y_start = min(c['cy'] for c in codes) - 25
    region = sorted([it for it in items if y_start <= it['cy'] <= y_end], key=lambda x: x['cy'])
    if not region:
        return []

    rows, current = [], [region[0]]
    for it in region[1:]:
        if it['cy'] - current[-1]['cy'] < 22:
            current.append(it)
        else:
            rows.append(current)
            current = [it]
    rows.append(current)

    def _in_codigo_col(t):
        return (ITEM_CODE_RE.match(t['text'])
                and codigo_lo <= t['cx'] / page_w < codigo_hi + 0.03)

    split_rows = []
    for row in rows:
        codes_in = sorted([t for t in row if _in_codigo_col(t)], key=lambda x: x['cy'])
        if len(codes_in) <= 1:
            split_rows.append(row)
            continue
        boundaries = [-float('inf')]
        for i in range(len(codes_in) - 1):
            boundaries.append((codes_in[i]['cy'] + codes_in[i + 1]['cy']) / 2)
        boundaries.append(float('inf'))
        for i in range(len(codes_in)):
            y_top, y_bot = boundaries[i], boundaries[i + 1]
            sub = [t for t in row if y_top <= t['cy'] < y_bot]
            if sub:
                split_rows.append(sub)
    rows = split_rows

    numeric_cols = {'cant_solicitada', 'cant_despachada'}
    parsed = []
    for row in rows:
        data = {k: '' for k in cols}
        buckets: dict[str, list[dict]] = {k: [] for k in cols}
        for it in sorted(row, key=lambda x: x['cx']):
            rx = it['cx'] / page_w
            for col, (lo, hi) in cols.items():
                if lo <= rx < hi:
                    buckets[col].append(it)
                    data[col] = (data[col] + ' ' + it['text']).strip() if data[col] else it['text']
                    break
        combined = ' '.join(data.values()).upper()
        if 'NOMBRE DEL PRODUCTO' in combined or ('PRESEN' in combined and 'CANTIDAD' in combined):
            continue
        if 'ULTIMA' in combined or 'LINEA' in combined:
            break
        codigo_raw = data.get('codigo', '')
        codigo_clean = _clean_codigo(codigo_raw)
        if not codigo_clean:
            continue
        data['codigo'] = codigo_clean
        for k in numeric_cols:
            if k not in cols:
                continue
            lo, hi = cols[k]
            center = (lo + hi) / 2 * page_w
            data[k] = _pick_numeric_token(buckets[k], center)
        row_y1 = min(r['y'] for r in row)
        row_y2 = max(r['y2'] for r in row)
        real_cod_tokens = [t for t in buckets.get('codigo', [])
                           if ITEM_CODE_RE.match(t['text'])
                           and _clean_codigo(t['text']) == codigo_clean]
        if row_y2 - row_y1 > 80 and real_cod_tokens:
            cod_t = real_cod_tokens[0]
            cod_h = cod_t['y2'] - cod_t['y']
            half = max(20, int(cod_h * 0.9))
            row_y1 = int(cod_t['cy'] - half)
            row_y2 = int(cod_t['cy'] + half)
        data['_y1'] = int(row_y1)
        data['_y2'] = int(row_y2)
        if 'cant_despachada' in cols:
            lo, hi = cols['cant_despachada']
            data['_despa_x1'] = int(lo * page_w)
            data['_despa_x2'] = int(hi * page_w)
        if 'codigo' in cols:
            lo, hi = cols['codigo']
            data['_cod_x1'] = int(lo * page_w)
            data['_cod_x2'] = int(hi * page_w)
        data['_cod_verified'] = codigo_clean in _load_valid_codigos()
        parsed.append(data)
    return parsed


# ─── Filtros de tokens numéricos (anti X-marks) ─────────────────────────────
DIGIT_RUN_RE = re.compile(r'\d+')
MIN_NUMERIC_CONF = 55
MAX_SINGLE_DIGIT_ASPECT = 0.9


def _pick_numeric_token(tokens: list[dict], center_x: float) -> str:
    """De los tokens en una celda numérica, extrae dígitos de cada uno y elige el más cercano
    al centro horizontal. Filtra:
    - Tokens con confianza baja (marcas manuscritas y X suelen tener conf < 55).
    - Tokens de un solo dígito con aspect ratio ~cuadrado (X leídas como 4/5/etc)."""
    if not tokens:
        return ''
    candidates = []
    for t in tokens:
        if t.get('conf', 0) < MIN_NUMERIC_CONF:
            continue
        w = t['x2'] - t['x']
        h = t['y2'] - t['y']
        for m in DIGIT_RUN_RE.finditer(t['text']):
            digits = m.group(0)
            if len(digits) == 1 and h > 0 and w / h > MAX_SINGLE_DIGIT_ASPECT:
                continue
            candidates.append((digits, t['cx']))
    if not candidates:
        return ''
    return min(candidates, key=lambda c: abs(c[1] - center_x))[0]


# ─── Validación de códigos contra BaseMateriales ────────────────────────────
CODIGO_PREFIXES = ('100', '101', '130', '170', '181', '910')
CODIGOS_MAESTRO_PATH = Path(__file__).parent / 'codigos.xlsx'
_VALID_CODIGOS: set[str] | None = None


def _load_valid_codigos() -> set[str]:
    """Carga la hoja BaseMateriales de codigos.xlsx (9-digit codes) una sola vez."""
    global _VALID_CODIGOS
    if _VALID_CODIGOS is not None:
        return _VALID_CODIGOS
    if not CODIGOS_MAESTRO_PATH.exists():
        _VALID_CODIGOS = set()
        return _VALID_CODIGOS
    try:
        df = pd.read_excel(CODIGOS_MAESTRO_PATH, sheet_name='BaseMateriales')
        df.columns = [c.strip() for c in df.columns]
        col = 'Codigo' if 'Codigo' in df.columns else df.columns[0]
        _VALID_CODIGOS = set()
        for v in df[col].dropna():
            if isinstance(v, (int, float)) and float(v).is_integer():
                _VALID_CODIGOS.add(str(int(v)))
            else:
                s = str(v).strip()
                if s.isdigit():
                    _VALID_CODIGOS.add(s)
    except Exception:
        _VALID_CODIGOS = set()
    return _VALID_CODIGOS


OCR_DIGIT_CONFUSIONS = {
    '0': '9863',
    '1': '7',
    '2': '',
    '3': '85',
    '4': '9',
    '5': '638',
    '6': '805',
    '7': '1',
    '8': '063',
    '9': '048',
}


def _try_recover_codigo(digits: str, valid: set[str]) -> str | None:
    """Prueba substituciones de un solo dígito con confusiones OCR típicas.
    Ej: '010100357' con sub pos 0 (0→9) → '910100357'."""
    if len(digits) != 9:
        return None
    for i, ch in enumerate(digits):
        for alt in OCR_DIGIT_CONFUSIONS.get(ch, ''):
            candidate = digits[:i] + alt + digits[i + 1:]
            if candidate in valid:
                return candidate
    return None


def _clean_codigo(raw: str) -> str | None:
    """Devuelve un código válido (existe en BaseMateriales) o el mejor candidato posible.
    Si el candidato no matchea, intenta corregir un dígito con substituciones OCR típicas."""
    if not raw:
        return None
    valid = _load_valid_codigos()
    tokens = re.findall(r'\d+', raw)
    for tok in tokens:
        if len(tok) == 9 and tok in valid:
            return tok
    compact = ''.join(tokens)
    for i in range(len(compact) - 8):
        sub = compact[i:i + 9]
        if sub in valid:
            return sub
    for tok in tokens:
        if len(tok) == 9:
            recovered = _try_recover_codigo(tok, valid)
            if recovered:
                return recovered
    for i in range(len(compact) - 8):
        sub = compact[i:i + 9]
        recovered = _try_recover_codigo(sub, valid)
        if recovered:
            return recovered
    for tok in tokens:
        if len(tok) == 9 and tok[:3] in CODIGO_PREFIXES:
            return tok
    for tok in tokens:
        if len(tok) == 9:
            return tok
    for i in range(len(compact) - 8):
        sub = compact[i:i + 9]
        if sub[:3] in CODIGO_PREFIXES:
            return sub
    m = re.search(r'\d{6,10}', compact)
    return m.group(0) if m else None


# ─── Pipeline por DPI ────────────────────────────────────────────────────────
def _extract_warehouse_receipt(items, page_w: int, page_h: int) -> dict[str, dict]:
    """Extrae actas de ingreso de mercadería, cuyo formato no es requisición."""
    has_acta = any('ACTA' in item['text'].upper() for item in items)
    received_anchors = [
        item for item in items
        if 'RECIBIDO' in item['text'].upper() and item['cy'] < page_h * 0.60
    ]
    if not has_acta or not received_anchors:
        return {}

    valid_codes = _load_valid_codigos()
    code_tokens = []
    for item in items:
        digits = ''.join(re.findall(r'\d', item['text']))
        if len(digits) == 9 and (not valid_codes or digits in valid_codes):
            code_tokens.append((digits, item))
    if not code_tokens:
        return {}

    code_counts: dict[str, int] = {}
    for code, _ in code_tokens:
        code_counts[code] = code_counts.get(code, 0) + 1
    code = max(code_counts, key=code_counts.get)
    first_code_token = min(
        (item for candidate, item in code_tokens if candidate == code),
        key=lambda item: item['cy'],
    )

    anchor = min(received_anchors, key=lambda item: item['cy'])
    quantity_candidates = []
    for item in items:
        normalized = item['text'].strip().replace(',', '.')
        if not re.fullmatch(r'\d+(?:\.\d+)?', normalized):
            continue
        dy = item['y'] - anchor['y2']
        dx = abs(item['cx'] - anchor['cx'])
        if 0 <= dy <= page_h * 0.12 and dx <= page_w * 0.12:
            quantity_candidates.append((dy + dx * 0.20, normalized, item))
    if not quantity_candidates:
        return {}
    _, quantity, quantity_token = min(quantity_candidates, key=lambda candidate: candidate[0])
    if re.fullmatch(r'\d{1,3}\.\d{3}', quantity):
        quantity = quantity.replace('.', '')
    elif quantity.endswith('.0'):
        quantity = quantity[:-2]

    description_tokens = [
        item['text'] for item in sorted(items, key=lambda item: item['x'])
        if abs(item['cy'] - first_code_token['cy']) <= 35
        and item['x'] > first_code_token['x2']
        and item['x'] < page_w * 0.72
        and not re.fullmatch(r'[|_\-]+', item['text'])
    ]
    return {
        code: {
            'codigo': code,
            'nombre_producto': ' '.join(description_tokens).strip(),
            'presentacion': '',
            'cant_solicitada': quantity,
            'cant_despachada': quantity,
            '_cod_verified': code in valid_codes if valid_codes else False,
            '_y1': quantity_token['y'],
            '_y2': quantity_token['y2'],
        }
    }


def _extract_at_dpi(pdf_path: Path, dpi: int):
    """Devuelve (base, items_por_codigo, img) para un DPI dado. Sólo aplica a PDFs."""
    img = deskew(load_image(pdf_path, dpi=dpi))
    items = run_ocr(img)
    if not items:
        return None, {}, img
    page_h = max(it['y2'] for it in items)
    page_w = max(it['x2'] for it in items)
    filename_doc = no_doc_from_filename(pdf_path)
    scanned_doc = extract_doc_number_from_image(img) or extract_doc_number(items)
    resolved_doc, doc_source = resolve_doc_number(filename_doc, scanned_doc)
    base = {
        **extract_header(items, page_h),
        'no_doc': resolved_doc,
        'no_doc_ocr_local': scanned_doc,
        'no_doc_archivo': filename_doc,
        'no_doc_fuente': doc_source,
        'no_doc_coincide_archivo': (
            resolved_doc == filename_doc if resolved_doc and filename_doc else None
        ),
        **extract_signatures(items),
    }
    receipt_rows = _extract_warehouse_receipt(items, page_w, page_h)
    rows = list(receipt_rows.values()) if receipt_rows else extract_items(
        items, page_w, page_h,
    )
    return base, {r['codigo']: r for r in rows if r.get('codigo')}, img


# ─── Fallback local: OCR por celda con CC-filtering ─────────────────────────
DIGIT_ONLY_CONFIGS = (
    '--psm 8 -c tessedit_char_whitelist=0123456789',
    '--psm 7 -c tessedit_char_whitelist=0123456789',
    '--psm 10 -c tessedit_char_whitelist=0123456789',
)


def _ocr_cell(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> str:
    """OCR con whitelist de dígitos. Filtra componentes conectados 'no-dígito':
    conserva sólo CCs más altos que anchos (dígitos impresos) y descarta CCs cuadrados
    o diagonales (X marks a cualquier lado)."""
    h, w = img.shape[:2]
    pad_y = int((y2 - y1) * 0.15)
    y1 = max(0, y1 - pad_y)
    y2 = min(h, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return ''
    crop = img[y1:y2, x1:x2]
    crop = remove_colored_marks(crop)
    ch, cw = crop.shape[:2]
    if ch < 120:
        scale = 120 / ch
        crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    _, th_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, th_inv.shape[1] // 3), 1))
    lines_h = cv2.morphologyEx(th_inv, cv2.MORPH_OPEN, kernel_h)
    no_lines = cv2.subtract(th_inv, lines_h)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(no_lines, connectivity=8)
    digit_ccs = []
    for i in range(1, n_labels):
        cx, cy, cw_i, ch_i, area = stats[i]
        if area < 40 or ch_i < 15:
            continue
        bbox_area = cw_i * ch_i
        fill = area / bbox_area if bbox_area else 0
        aspect = cw_i / ch_i if ch_i else 999
        if aspect > 1.1:
            continue
        if fill < 0.28 and aspect > 0.55:
            continue
        digit_ccs.append((i, cx, cy, cw_i, ch_i, area))
    if not digit_ccs:
        return ''
    digit_ccs.sort(key=lambda c: c[1])
    median_h = int(np.median([c[4] for c in digit_ccs]))
    gap_threshold = max(20, median_h // 2)
    groups, current = [], [digit_ccs[0]]
    for cc in digit_ccs[1:]:
        prev = current[-1]
        gap = cc[1] - (prev[1] + prev[3])
        if gap < gap_threshold:
            current.append(cc)
        else:
            groups.append(current)
            current = [cc]
    groups.append(current)
    best_group = max(groups, key=lambda g: sum(c[5] for c in g))
    kept = np.zeros_like(no_lines)
    for cc in best_group:
        kept[labels == cc[0]] = 255
    clean = cv2.bitwise_not(kept)

    for cfg in DIGIT_ONLY_CONFIGS:
        txt = pytesseract.image_to_string(clean, config=cfg).strip()
        m = re.search(r'\d+', txt)
        if m:
            return m.group(0)
    return ''


# ─── Enhancement opcional: Gemini AI (opt-in) ───────────────────────────────
import sys as _sys
import threading as _threading

GEMINI_MODEL = 'gemini-2.5-flash'
GEMINI_URL = f'https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent'
GEMINI_TIMEOUT = 20

# Contadores en memoria: nos dan un termómetro claro de si Gemini se está
# usando y con qué tasa de error, sin cambiar la API pública del módulo.
GEMINI_STATS = {'ok': 0, 'error': 0, 'skipped_no_key': 0}
_GEMINI_STATS_LOCK = _threading.Lock()


def _gemini_track(kind: str, exc: Exception | None = None, context: str = '') -> None:
    with _GEMINI_STATS_LOCK:
        GEMINI_STATS[kind] = GEMINI_STATS.get(kind, 0) + 1
    if exc is not None:
        message = f'[gemini] {context or "call"} failed: {type(exc).__name__}: {exc}'
        print(message, file=_sys.stderr, flush=True)


def reset_gemini_stats() -> None:
    with _GEMINI_STATS_LOCK:
        for key in GEMINI_STATS:
            GEMINI_STATS[key] = 0


def gemini_ping() -> tuple[bool, str]:
    """Prueba mínima contra el endpoint; útil para diagnosticar la key en vivo."""
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        return False, 'GOOGLE_API_KEY no está configurada'
    payload = {
        'contents': [{'parts': [{'text': 'Respondé con la palabra: OK'}]}],
        'generationConfig': {
            'temperature': 0, 'maxOutputTokens': 8,
            'thinkingConfig': {'thinkingBudget': 0},
        },
    }
    try:
        r = requests.post(f'{GEMINI_URL}?key={api_key}', json=payload, timeout=15)
        r.raise_for_status()
        text = r.json()['candidates'][0]['content']['parts'][0]['text'].strip()
        return True, f'Respuesta: {text!r}'
    except requests.HTTPError as e:
        return False, f'HTTP {e.response.status_code}: {e.response.text[:200]}'
    except Exception as e:
        return False, f'{type(e).__name__}: {e}'
GEMINI_PROMPT = (
    'Esta imagen es una celda de un formulario médico. Contiene un NÚMERO IMPRESO '
    '(dígitos 0-9) y posiblemente una X manuscrita al lado, encima o cruzándolo. '
    'Ignorá completamente la X manuscrita. Respondé SOLO con los dígitos del número '
    'impreso, sin explicaciones ni texto extra. Si no hay número, respondé: VACIO.'
)


GEMINI_MAX_CELL_HEIGHT = 70
GEMINI_DOC_PROMPT = (
    'Leé el campo impreso "No. Doc. SAFISS" de esta requisición. '
    'Devolvé únicamente sus 10 dígitos, sin espacios ni explicaciones. '
    'Ignorá fechas, códigos de producto, cantidades, sellos y números de empleado. '
    'Si no es claramente legible, devolvé VACIO. No inventes ningún dígito.'
)
GEMINI_CODIGO_PROMPT = (
    'Esta imagen es la celda de "código de material" de un formulario médico. '
    'Contiene un NÚMERO IMPRESO de 9 dígitos. Respondé SOLO con los 9 dígitos, '
    'sin espacios, sin guiones, sin texto extra. Si no ves un código, respondé: VACIO.'
)


def _gemini_read(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, prompt: str) -> str:
    """Envía un recorte a Gemini con el prompt dado. Recorta al centro si es multi-fila."""
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        _gemini_track('skipped_no_key')
        return ''
    h_img = img.shape[0]
    if y2 - y1 > GEMINI_MAX_CELL_HEIGHT:
        cy = (y1 + y2) // 2
        half = GEMINI_MAX_CELL_HEIGHT // 2
        y1, y2 = cy - half, cy + half
    pad_y = max(15, (y2 - y1) // 4)
    y1 = max(0, y1 - pad_y)
    y2 = min(h_img, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return ''
    ok, buf = cv2.imencode('.png', img[y1:y2, x1:x2])
    if not ok:
        return ''
    payload = {
        'contents': [{'parts': [
            {'text': prompt},
            {'inline_data': {'mime_type': 'image/png',
                             'data': base64.b64encode(buf.tobytes()).decode('ascii')}},
        ]}],
        'generationConfig': {
            'temperature': 0, 'maxOutputTokens': 32,
            'thinkingConfig': {'thinkingBudget': 0},
        },
    }
    try:
        r = requests.post(f'{GEMINI_URL}?key={api_key}', json=payload, timeout=GEMINI_TIMEOUT)
        r.raise_for_status()
        text = r.json()['candidates'][0]['content']['parts'][0]['text'].strip()
    except Exception as e:
        _gemini_track('error', e, 'cell read')
        return ''
    _gemini_track('ok')
    if 'VACIO' in text.upper():
        return ''
    return text


def _ocr_doc_number_gemini(img: np.ndarray) -> str | None:
    """Lee con Gemini el No. Doc. SAFISS en la franja inferior del formulario."""
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        _gemini_track('skipped_no_key')
        return None
    h, w = img.shape[:2]
    crop = img[int(h * 0.62):h, int(w * 0.10):int(w * 0.95)]
    ok, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return None
    payload = {
        'contents': [{'parts': [
            {'text': GEMINI_DOC_PROMPT},
            {'inline_data': {'mime_type': 'image/jpeg',
                             'data': base64.b64encode(buf.tobytes()).decode('ascii')}},
        ]}],
        'generationConfig': {
            'temperature': 0,
            'maxOutputTokens': 32,
            'thinkingConfig': {'thinkingBudget': 0},
        },
    }
    try:
        response = requests.post(
            f'{GEMINI_URL}?key={api_key}', json=payload, timeout=GEMINI_TIMEOUT,
        )
        response.raise_for_status()
        text = response.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        _gemini_track('error', e, 'doc number')
        return None
    _gemini_track('ok')
    digits = ''.join(re.findall(r'\d', str(text)))
    return digits if len(digits) == 10 else None


def _ocr_codigo_gemini(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> str | None:
    """Pide a Gemini que lea el código impreso. Retorna 9 dígitos si están en BaseMateriales."""
    text = _gemini_read(img, x1, y1, x2, y2, GEMINI_CODIGO_PROMPT)
    digits = ''.join(re.findall(r'\d+', text))
    if len(digits) != 9:
        return None
    # Sin base maestra (Streamlit Cloud sin `codigos.xlsx`), aceptamos cualquier
    # secuencia de 9 dígitos: filtrar exigiendo la base convertía a esta función
    # en un no-op y hacía perder rescates válidos de códigos por parte de Gemini.
    valid = _load_valid_codigos()
    if valid and digits not in valid:
        return None
    return digits


GEMINI_TABLE_PROMPT = (
    'Esta imagen es la tabla de items de un formulario de requisición médica. Cada fila '
    'tiene: CODIGO (9 dígitos impresos), NOMBRE PRODUCTO, PRESENTACION, CANT SOLICITADA '
    '(número impreso) y CANT DESPACHADA (número impreso, a veces con una X manuscrita al '
    'lado o encima — IGNORÁ la X y leé sólo el dígito impreso). '
    'Extraé TODAS las filas visibles como JSON array, sin markdown, sin explicaciones. '
    'Cada objeto debe tener exactamente las claves "codigo", "cant_solicitada" y '
    '"cant_despachada". El código debe contener los 9 dígitos que realmente sean visibles. '
    'Usá "" si un valor no es visible. NO inventes datos.'
)


def _extract_items_gemini(img: np.ndarray, y_start: int, y_end: int) -> list[dict]:
    """Envía la región de items completa a Gemini y parsea el JSON resultante.
    Filtra filas cuyo codigo no esté en BaseMateriales."""
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        _gemini_track('skipped_no_key')
        return []
    h_img, w_img = img.shape[:2]
    y1 = max(0, y_start - 10)
    y2 = min(h_img, y_end + 10)
    if y2 <= y1:
        return []
    crop = img[y1:y2, 0:w_img]
    ok, buf = cv2.imencode('.png', crop)
    if not ok:
        return []
    payload = {
        'contents': [{'parts': [
            {'text': GEMINI_TABLE_PROMPT},
            {'inline_data': {'mime_type': 'image/png',
                             'data': base64.b64encode(buf.tobytes()).decode('ascii')}},
        ]}],
        'generationConfig': {
            'temperature': 0, 'maxOutputTokens': 4096,
            'thinkingConfig': {'thinkingBudget': 0},
            'responseMimeType': 'application/json',
        },
    }
    try:
        r = requests.post(f'{GEMINI_URL}?key={api_key}', json=payload, timeout=60)
        r.raise_for_status()
        text = r.json()['candidates'][0]['content']['parts'][0]['text'].strip()
    except Exception as e:
        _gemini_track('error', e, 'items table')
        return []
    _gemini_track('ok')
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    valid = _load_valid_codigos()
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        cod = str(item.get('codigo', '')).strip()
        if not cod or not cod.isdigit() or len(cod) != 9:
            continue
        # Si tenemos la base maestra `codigos.xlsx`, la usamos como validación
        # dura para descartar lecturas espurias. Si no está disponible (p. ej.
        # Streamlit Cloud, donde el archivo queda fuera del repo), aceptamos
        # cualquier código de 9 dígitos: es preferible tener algún falso
        # positivo a perder silenciosamente todas las filas.
        if valid and cod not in valid:
            continue
        out.append({
            'codigo': cod,
            'cant_solicitada': str(item.get('cant_solicitada', '')).strip(),
            'cant_despachada': str(item.get('cant_despachada', '')).strip(),
        })
    return out


def _ocr_cell_gemini(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> str:
    """Enhancement con Gemini para cant_despachada: lee el dígito impreso ignorando X marks."""
    text = _gemini_read(img, x1, y1, x2, y2, GEMINI_PROMPT)
    digits = DIGIT_RUN_RE.findall(text)
    return digits[0] if digits else ''


# ─── Orquestación: merge doble-pasada + fallbacks ───────────────────────────
def _pick_best_cantidad(a: str, b: str) -> str:
    """Entre dos lecturas de cantidad, prefiere numérica y más corta (evita fusiones tipo '2300')."""
    a, b = str(a or '').strip(), str(b or '').strip()
    a_d, b_d = a.isdigit(), b.isdigit()
    if a_d and b_d:
        return a if a == b else (a if len(a) <= len(b) else b)
    if a_d:
        return a
    if b_d:
        return b
    return a or b


def _needs_refinement(current: str, solic: str) -> bool:
    """Una celda necesita re-OCR si está vacía, mal formada o no cuadra con solicitada.
    Cuando no hay solicitada válida, también se refina (no podemos validar de otro modo)."""
    if not current or not current.isdigit():
        return True
    if len(current) >= 4:
        return True
    if not solic.isdigit():
        return True
    return int(current) != int(solic)


def _fix_unverified_codes(rows_dict: dict, img: np.ndarray) -> dict:
    """Para cada fila con código no verificado en BaseMateriales, pide a Gemini que lo relea.
    Si obtiene un código válido, reindexa la fila con ese código."""
    fixed: dict = {}
    for cod, row in rows_dict.items():
        if row.get('_cod_verified'):
            fixed[cod] = row
            continue
        cx1, cx2 = row.get('_cod_x1'), row.get('_cod_x2')
        y1, y2 = row.get('_y1'), row.get('_y2')
        if None in (cx1, cx2, y1, y2):
            fixed[cod] = row
            continue
        new_cod = _ocr_codigo_gemini(img, cx1, y1, cx2, y2)
        if new_cod and new_cod != cod and new_cod not in fixed:
            row['codigo'] = new_cod
            row['_cod_verified'] = True
            fixed[new_cod] = row
        else:
            fixed[cod] = row
    return fixed


def process_pdf(pdf_path: Path, use_gemini: bool = False, debug_dir: Path | None = None):
    """Pipeline de extracción:
    1. OCR doble pasada (DPI 300 primaria, 400 secundaria) — merge por código.
    2. Si use_gemini: rescatar códigos no válidos con Gemini antes del merge.
    3. Fallback local `_ocr_cell` (CC-filtering) para celdas vacías o sospechosas.
    4. Enhancement opcional con Gemini para cantidades."""
    is_pdf = pdf_path.suffix.lower() not in IMAGE_EXTS

    if is_pdf:
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_300 = ex.submit(_extract_at_dpi, pdf_path, 300)
            fut_400 = ex.submit(_extract_at_dpi, pdf_path, 400)
            try:
                base, rows_300, img_300 = fut_300.result()
            except Exception:
                base, rows_300, img_300 = None, {}, None
            try:
                base_400, rows_400, img_400 = fut_400.result()
            except Exception:
                base_400, rows_400, img_400 = None, {}, None
    else:
        base, rows_300, img_300 = _extract_at_dpi(pdf_path, dpi=300)
        base_400 = None
        rows_400, img_400 = {}, None

    if base is None and base_400 is not None:
        base = base_400
    elif (base is not None and base_400 is not None
          and not base.get('no_doc_ocr_local')
          and base_400.get('no_doc_ocr_local')):
        for key in ('no_doc', 'no_doc_ocr_local', 'no_doc_archivo',
                    'no_doc_fuente', 'no_doc_coincide_archivo'):
            base[key] = base_400.get(key)

    if base is None:
        return [{'archivo': pdf_path.name, 'error': 'OCR sin resultados'}]

    gemini_enabled = use_gemini and bool(os.getenv('GOOGLE_API_KEY'))

    if gemini_enabled and img_300 is not None:
        gemini_doc = _ocr_doc_number_gemini(img_300)
        local_doc = base.get('no_doc_ocr_local')
        base['no_doc_gemini'] = gemini_doc
        base['no_doc_coincide_ocr'] = (
            gemini_doc == local_doc if gemini_doc and local_doc else None
        )
        if gemini_doc:
            filename_doc = base.get('no_doc_archivo')
            resolved_doc, doc_source = resolve_doc_number(
                filename_doc, local_doc, gemini_doc,
            )
            base['no_doc'] = resolved_doc
            base['no_doc_fuente'] = doc_source
            base['no_doc_coincide_archivo'] = (
                resolved_doc == filename_doc if filename_doc else None
            )

    # La doble pasada local puede producir un código recortado o fusionado aunque
    # el resto de la fila sea legible. Esta reparación ya existía, pero no estaba
    # conectada al pipeline. Se ejecuta antes del merge para que una lectura mala
    # no sobreviva como un ítem adicional.
    if gemini_enabled:
        repairs = []
        with ThreadPoolExecutor(max_workers=2) as ex:
            futures = []
            if rows_300 and img_300 is not None:
                futures.append(('300', ex.submit(_fix_unverified_codes, rows_300, img_300)))
            if rows_400 and img_400 is not None:
                futures.append(('400', ex.submit(_fix_unverified_codes, rows_400, img_400)))
            for source, future in futures:
                try:
                    repairs.append((source, future.result()))
                except Exception:
                    continue
        for source, repaired in repairs:
            if source == '300':
                rows_300 = repaired
            else:
                rows_400 = repaired

    if is_pdf:
        all_codes = set(rows_300) | set(rows_400)
        merged = {}
        for cod in all_codes:
            a = rows_300.get(cod, {})
            b = rows_400.get(cod, {})
            row = {**b, **a} if a else dict(b)
            for k in ('cant_solicitada', 'cant_despachada'):
                row[k] = _pick_best_cantidad(a.get(k, ''), b.get(k, ''))
            row['codigo'] = cod
            merged[cod] = row
        rows = list(merged.values())
    else:
        rows = list(rows_300.values())

    for row in rows:
        cod = row.get('codigo')
        if cod in rows_300:
            img_ref, bbox_row = img_300, rows_300[cod]
        elif cod in rows_400 and img_400 is not None:
            img_ref, bbox_row = img_400, rows_400[cod]
        else:
            continue
        y1, y2 = bbox_row.get('_y1'), bbox_row.get('_y2')
        x1, x2 = bbox_row.get('_despa_x1'), bbox_row.get('_despa_x2')
        if None in (y1, y2, x1, x2):
            continue
        current = row.get('cant_despachada', '')
        solic = row.get('cant_solicitada', '')
        if not _needs_refinement(current, solic):
            continue
        local_val = _ocr_cell(img_ref, x1, y1, x2, y2)
        # El OCR especializado de la celda puede corregir lecturas de igual
        # longitud (p. ej. 42 -> 12) cuando la cantidad solicitada no pudo
        # servir de validación. Si la solicitada sí se leyó, conservamos el
        # valor principal para no convertir ceros reales marcados con tinta.
        use_equal_length_local = (
            local_val.isdigit()
            and current.isdigit()
            and len(local_val) == len(current)
            and not solic.isdigit()
        )
        if local_val and (
            not current
            or (local_val.isdigit() and len(local_val) < len(current))
            or use_equal_length_local
        ):
            row['cant_despachada'] = local_val

    if gemini_enabled:
        table_sources = []
        for src_rows, src_img in ((rows_300, img_300), (rows_400, img_400)):
            if not src_rows or src_img is None:
                continue
            y_starts = [r['_y1'] for r in src_rows.values() if r.get('_y1') is not None]
            y_ends = [r['_y2'] for r in src_rows.values() if r.get('_y2') is not None]
            if y_starts and y_ends:
                table_sources.append((src_img, min(y_starts) - 60, max(y_ends) + 60))

        table_results = []
        if table_sources:
            with ThreadPoolExecutor(max_workers=len(table_sources)) as ex:
                table_results = list(ex.map(
                    lambda args: _extract_items_gemini(*args), table_sources,
                ))

        # Se acepta el consenso entre resoluciones. También se acepta una fila
        # presente en una sola resolución cuando esa lectura cubre al menos el
        # 70 % de la tabla local: en ese caso Gemini leyó la tabla completa y la
        # otra llamada fue parcial (algo frecuente en formularios con 15-21
        # filas). El código, además, ya fue validado contra BaseMateriales.
        confirmed: dict[str, dict] = {}
        local_row_count = max(len(rows_300), len(rows_400), 1)
        min_complete_rows = max(3, math.ceil(local_row_count * 0.70))
        complete_results = [r for r in table_results if len(r) >= min_complete_rows]
        claims: dict[str, list[dict]] = {}
        for result in table_results:
            for item in result:
                claims.setdefault(item['codigo'], []).append(item)
        for cod, items in claims.items():
            quantities = {i.get('cant_despachada', '') for i in items
                          if i.get('cant_despachada', '') != ''}
            if len(items) >= 2 and len(quantities) == 1:
                confirmed[cod] = items[0]
                continue
            for result in complete_results:
                item = next((r for r in result if r['codigo'] == cod), None)
                if item is not None and item.get('cant_despachada', '') != '':
                    confirmed[cod] = item
                    break

        existing = {r.get('codigo'): r for r in rows}
        for cod, gemini_row in confirmed.items():
            if cod in existing:
                existing[cod]['cant_solicitada'] = gemini_row.get('cant_solicitada', '')
                existing[cod]['cant_despachada'] = gemini_row['cant_despachada']
            else:
                new_row = dict(gemini_row)
                rows.append(new_row)
                existing[cod] = new_row

        pending = []
        for row in rows:
            cod = row.get('codigo')
            if cod in confirmed:
                continue
            bbox_row = rows_300.get(cod) or rows_400.get(cod)
            if not bbox_row:
                continue
            img_ref = img_300 if cod in rows_300 else img_400
            if img_ref is None:
                continue
            y1, y2 = bbox_row.get('_y1'), bbox_row.get('_y2')
            x1, x2 = bbox_row.get('_despa_x1'), bbox_row.get('_despa_x2')
            if None in (y1, y2, x1, x2):
                continue
            pending.append((row, img_ref, x1, y1, x2, y2))
        if pending:
            with ThreadPoolExecutor(max_workers=min(8, len(pending))) as ex:
                results = list(ex.map(lambda a: _ocr_cell_gemini(*a[1:]), pending))
            for (row, *_), gem in zip(pending, results):
                if gem:
                    row['cant_despachada'] = gem

    for row in rows:
        for k in ('_y1', '_y2', '_despa_x1', '_despa_x2', '_cod_x1', '_cod_x2', '_cod_verified'):
            row.pop(k, None)

    base['archivo'] = pdf_path.name
    base['archivo_hash'] = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    result = [base] if not rows else [{**base, **r} for r in rows]
    # Cada imagen full-page a 300/400 DPI pesa 10-20 MB; sin liberar
    # explícitamente, procesar 30+ archivos en Streamlit Cloud (1 GB de RAM)
    # revienta el container y se pierde toda la corrida.
    del img_300, img_400, rows_300, rows_400
    import gc as _gc
    _gc.collect()
    return result


# ─── Ejecución sobre carpeta / CLI ──────────────────────────────────────────
PREFERRED_COLS = ['archivo', 'no_doc', 'mov', 'almacen', 'depend_destinataria', 'lote', 'año', 'mes',
                  'total_despachada', 'total_no_despachada',
                  'solicito', 'autorizo', 'recibio', 'preparado_por', 'comprobado_por', 'elaborado',
                  'idem', 'codigo', 'nombre_producto', 'presentacion', 'cant_solicitada', 'cant_despachada']


def list_input_files(input_dir: Path) -> list[Path]:
    patterns = ['*.pdf', '*.PDF'] + [f'*{e}' for e in IMAGE_EXTS] + [f'*{e.upper()}' for e in IMAGE_EXTS]
    return sorted({f for p in patterns for f in input_dir.glob(p)})


def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in PREFERRED_COLS if c in df.columns] + [c for c in df.columns if c not in PREFERRED_COLS]
    return df[cols]


def iter_extract_folder(input_dir: Path, use_gemini: bool = False,
                        debug_dir: Path | None = None,
                        skip_names: set[str] | None = None):
    """Genera (indice, total, archivo, filas_dict) por cada PDF procesado.

    `skip_names` recibe nombres de archivo ya procesados (para reanudar tras
    un OOM sin re-correr OCR + IA en los que ya estaban listos). Cuando un
    archivo está en el set, se yieldea con filas vacías y sin invocar OCR.
    """
    skip_names = skip_names or set()
    files = list_input_files(input_dir)
    total = len(files)
    for i, f in enumerate(files):
        if f.name in skip_names:
            yield i + 1, total, f, []
            continue
        try:
            rows = process_pdf(f, use_gemini=use_gemini, debug_dir=debug_dir)
        except Exception as e:
            rows = [{'archivo': f.name, 'error': str(e)}]
        yield i + 1, total, f, rows


def extract_folder(input_dir: Path, use_gemini: bool = False,
                   debug_dir: Path | None = None, on_progress=None) -> pd.DataFrame:
    all_rows = []
    total = 0
    for i, total, f, rows in iter_extract_folder(input_dir, use_gemini=use_gemini, debug_dir=debug_dir):
        if on_progress:
            on_progress(i - 1, total, f.name)
        all_rows.extend(rows)
    if on_progress:
        on_progress(total, total, '')
    if not all_rows:
        return pd.DataFrame()
    return order_columns(pd.DataFrame(all_rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='documentos')
    ap.add_argument('--output', default='output/dataset.csv')
    ap.add_argument('--gemini', action='store_true',
                    help='Potencia el análisis con Gemini AI para celdas difíciles (requiere GOOGLE_API_KEY).')
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    debug_dir = Path('debug') if args.debug else None
    if debug_dir:
        debug_dir.mkdir(exist_ok=True)

    df = extract_folder(input_dir, use_gemini=args.gemini, debug_dir=debug_dir,
                        on_progress=lambda i, n, name: print(f'[{i}/{n}] {name}') if name else None)
    if df.empty:
        print(f'No hay archivos en {input_dir}/ o nada extraido.')
        return
    df.to_csv(output_path, index=False)
    df.to_excel(output_path.with_suffix('.xlsx'), index=False)
    print(f'\n{len(df)} filas -> {output_path} y {output_path.with_suffix(".xlsx")}')


if __name__ == '__main__':
    main()
