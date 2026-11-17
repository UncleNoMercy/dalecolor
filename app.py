"""
DaleColor - Versión web
=======================
Proyecto ABPro - Funciones y Matrices (MTFM01)

Esta página usa las mismas funciones matemáticas de dalecolor.py
(vectores, matrices y sumatorias); solo agrega la parte visual.

Cómo abrirla (en la terminal de VS Code):
    py -m streamlit run app.py
"""

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from dalecolor import (
    MATRICES,
    CONTRASTE_MINIMO_TEXTO,
    CONTRASTE_MINIMO_GRANDE,
    simular_daltonismo,
    simular_un_color,
    colores_principales,
    luminancia,
    contraste,
    analizar_colores,
)

st.set_page_config(page_title="DaleColor", page_icon="🎨", layout="wide",
                   initial_sidebar_state="collapsed")

# Nombres cortos para mostrar, en el mismo orden que MATRICES
TIPOS = {
    "Protanopia (sin rojo)": ("Protanopia", "No percibe el rojo. Confunde rojos, cafés y naranjos."),
    "Deuteranopia (sin verde)": ("Deuteranopia", "No percibe el verde. Confunde el rojo con el verde."),
    "Tritanopia (sin azul)": ("Tritanopia", "No percibe el azul. Confunde el azul con el verde. Es poco frecuente."),
}
PALETA = [(220, 40, 40), (240, 140, 30), (245, 210, 40), (60, 170, 70), (40, 110, 220), (140, 70, 180)]

# ---------------------------------------------------------------------------
# Estilo: tipografía Atkinson Hyperlegible (diseñada para baja visión) y el
# rojo del informe. Los estados usan azul y rojo con íconos, nunca solo rojo/verde.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400&display=swap');
:root { --rojo:#C00000; --rojo-suave:#FBEAEA; --tinta:#222222; --gris:#5E5E5E;
        --linea:#E6E2E2; --azul:#1F5FAD; --azul-suave:#E7EFF9; }
html, body, .stMarkdown, p, li, h1, h2, h3, label, input, table {
  font-family: 'Atkinson Hyperlegible', sans-serif !important; }
/* Fondo claro siempre, aunque el computador use modo oscuro */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: #FFFFFF !important; color: var(--tinta); }
header[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer { display:none !important; }
.block-container { max-width: 1120px; padding-top: 2.4rem; padding-bottom: 4rem; }
p, li { color: var(--tinta); font-size: 1.06rem; line-height: 1.55; }

.hero .marca { color: var(--rojo); font-weight: 700; font-size: 1.15rem; letter-spacing: .01em; margin: 0 0 1.2rem; }
.marca span { display:inline-block; width:.8em; height:.8em; border-radius:50%; background:var(--rojo);
              margin-right:.4em; vertical-align:-.05em; }
.hero h1 { font-size: clamp(2.4rem, 5vw, 3.9rem); line-height: 1.04; font-weight: 700; color: var(--tinta);
           letter-spacing: -.02em; margin: 0 0 1rem; padding: 0; }
.hero p { font-size: 1.2rem; color: var(--gris); max-width: 30em; margin-bottom: .6rem; }
.pies { display:flex; justify-content:space-around; color: var(--gris); font-size:.95rem; margin-top:-.6rem; }
.pie-lamina { text-align:center; color: var(--tinta); font-size: 1rem; margin-top: .5rem; }

h2.seccion { font-size: 1.7rem; font-weight: 700; color: var(--tinta); margin: 2.6rem 0 .3rem; padding: 0;
             border-top: 3px solid var(--rojo); padding-top: .9rem; }
.ayuda { color: var(--gris); margin-bottom: 1rem; max-width: 46em; }

.tipo { border: 1px solid var(--linea); border-radius: 10px; padding: 1rem 1.1rem 1.1rem; height: 100%; }
.tipo h3 { font-size: 1.2rem; margin: 0 0 .25rem; padding: 0; color: var(--tinta); }
.tipo p { font-size: .98rem; color: var(--gris); margin: 0 0 .8rem; }
.franja { display:flex; height: 18px; border-radius: 4px; overflow: hidden; }
.franja + .franja { margin-top: 4px; }
.franja i { flex: 1; }
.leyenda { display:flex; justify-content:space-between; font-size:.82rem; color: var(--gris); margin-top:.35rem; }

.sw { display:inline-block; width:1.15em; height:1.15em; border-radius:4px; border:1px solid rgba(0,0,0,.2);
      vertical-align:-.2em; margin-right:.35em; }
table.dc { border-collapse: collapse; width: 100%; font-size: 1rem; }
table.dc th { text-align:left; font-weight:700; color: var(--tinta); padding: .55rem .6rem;
              border-bottom: 2px solid var(--tinta); }
table.dc td { white-space: nowrap; padding: .55rem .6rem; border-bottom: 1px solid var(--linea); color: var(--tinta); }
table.dc td.num { font-variant-numeric: tabular-nums; font-weight: 700; }
.ok, .no { display:inline-block; padding: .12rem .6rem; border-radius: 999px; font-weight: 700; font-size: .92rem; }
.ok { background: var(--azul-suave); color: var(--azul); }
.no { background: var(--rojo-suave); color: var(--rojo); }

.alerta { display:grid; grid-template-columns: auto 1fr; gap: .2rem 1rem; align-items:center;
          border-left: 4px solid var(--rojo); background: #FCF7F7; padding: .8rem 1rem; margin-bottom: .6rem;
          border-radius: 0 8px 8px 0; }
.alerta .pares { font-size: 1.05rem; white-space: nowrap; }
.alerta .pares .sw { width: 1.6em; height: 1.6em; margin-right: .15em; }
.alerta .flecha { color: var(--gris); margin: 0 .5em; }
.alerta p { margin: 0; }
@media (max-width: 640px) {
  .alerta { grid-template-columns: 1fr; }
  .alerta .pares .sw { width: 1.3em; height: 1.3em; }
  table.dc { font-size: .9rem; }
}
.consejo { background: var(--azul-suave); color: var(--tinta); padding: .8rem 1rem; border-radius: 8px; }

div[data-testid="stFileUploaderDropzone"] { border: 2px dashed #D9A3A3; background: #FFFBFB; }
.stButton > button { border: 2px solid var(--rojo); color: var(--rojo); font-weight: 700; background: #fff; }
.stButton > button:hover { background: var(--rojo); color: #fff; border-color: var(--rojo); }
.firma { margin-top: 3.5rem; color: var(--gris); font-size: .92rem; border-top: 1px solid var(--linea); padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Funciones de apoyo
# ---------------------------------------------------------------------------
def fuente(tamano):
    try:
        return ImageFont.load_default(size=tamano)
    except TypeError:  # versiones antiguas de Pillow
        return ImageFont.load_default()


@st.cache_data(show_spinner=False)
def lamina(numero=7, lado=520, semilla=7):
    """Lámina tipo Ishihara: puntos naranjos forman un número sobre puntos verdes
    de luminancia parecida. Con daltonismo rojo-verde, el número desaparece."""
    rng = np.random.default_rng(semilla)
    mascara = Image.new("L", (lado, lado), 0)
    ImageDraw.Draw(mascara).text((lado / 2, lado / 2 + lado * .02), str(numero), fill=255,
                                 font=fuente(int(lado * .62)), anchor="mm",
                                 stroke_width=int(lado * .03), stroke_fill=255)
    dentro = np.array(mascara) > 0
    img = Image.new("RGB", (lado, lado), (255, 255, 255))
    dibujo = ImageDraw.Draw(img)
    figura = [(222, 118, 84), (232, 140, 96), (205, 104, 78), (240, 160, 110)]
    fondo = [(150, 168, 88), (168, 180, 102), (132, 158, 84), (182, 190, 120)]
    radio_total, puestos = lado / 2 - 4, []
    for _ in range(9000):
        r = rng.uniform(lado * .008, lado * .026)
        x, y = rng.uniform(r, lado - r), rng.uniform(r, lado - r)
        if (x - lado / 2) ** 2 + (y - lado / 2) ** 2 > (radio_total - r) ** 2:
            continue
        if any((x - a) ** 2 + (y - b) ** 2 < (r + c + 1.2) ** 2 for a, b, c in puestos[-400:]):
            continue
        puestos.append((x, y, r))
        paleta = figura if dentro[int(y), int(x)] else fondo
        dibujo.ellipse([x - r, y - r, x + r, y + r], fill=paleta[rng.integers(len(paleta))])
    normal = np.array(img)
    simulada = simular_daltonismo(normal.astype(float), MATRICES["Deuteranopia (sin verde)"]).astype(np.uint8)
    return normal, simulada


def tienda_ejemplo():
    """Una tienda online simple para probar la página sin subir nada."""
    img = Image.new("RGB", (900, 420), (255, 255, 255))
    d, f = ImageDraw.Draw(img), fuente(22)
    d.rectangle([0, 0, 900, 60], fill=(192, 0, 0))
    d.text((20, 18), "MI TIENDA ONLINE", fill=(255, 255, 255), font=f)
    d.rectangle([40, 90, 420, 380], fill=(240, 240, 240))
    d.text((60, 110), "Zapatillas urbanas", fill=(40, 40, 40), font=f)
    d.text((60, 145), "$29.990", fill=(220, 50, 50), font=f)
    d.rectangle([60, 300, 400, 350], fill=(50, 160, 50))
    d.text((130, 312), "AGREGAR AL CARRO", fill=(255, 255, 255), font=f)
    d.rectangle([480, 90, 860, 380], fill=(240, 240, 240))
    d.text((500, 110), "Polera negra", fill=(40, 40, 40), font=f)
    d.text((500, 145), "AGOTADO", fill=(220, 50, 50), font=f)
    d.rectangle([500, 300, 840, 350], fill=(220, 50, 50))
    d.text((610, 312), "SIN STOCK", fill=(255, 255, 255), font=f)
    return img


@st.cache_data(show_spinner=False)
def procesar(arreglo_bytes, alto, ancho):
    """Simula los tres tipos de daltonismo y busca los colores principales."""
    imagen = np.frombuffer(arreglo_bytes, dtype=np.uint8).reshape(alto, ancho, 3).astype(float)
    simulaciones = {n: simular_daltonismo(imagen, M).astype(np.uint8) for n, M in MATRICES.items()}
    return simulaciones, colores_principales(imagen)


def coma(x, dec=2):
    return f"{x:.{dec}f}".replace(".", ",")


def sw(rgb):
    return f'<span class="sw" style="background:rgb{tuple(int(v) for v in rgb)}"></span>'


def estado(ok):
    return '<span class="ok">✓ Cumple</span>' if ok else '<span class="no">✗ No cumple</span>'


def franja(colores):
    return '<div class="franja">' + "".join(f'<i style="background:rgb{tuple(c)}"></i>' for c in colores) + "</div>"


# ---------------------------------------------------------------------------
# Portada
# ---------------------------------------------------------------------------
izq, der = st.columns([1.05, 1], gap="large", vertical_alignment="center")
with izq:
    st.markdown("""
<div class="hero">
  <p class="marca"><span></span>DaleColor</p>
  <h1>Mira tu tienda online con otros ojos.</h1>
  <p>Cerca de 1 de cada 12 hombres tiene daltonismo. Sube una captura de tu tienda y te mostramos
  cómo la ven, qué colores se confunden y si el contraste cumple la norma WCAG 2.1.</p>
</div>""", unsafe_allow_html=True)
with der:
    normal, simulada = lamina()
    # Las dos láminas en una sola imagen, para que queden lado a lado también en el celular
    lado = normal.shape[0]
    par = Image.new("RGB", (lado * 2 + 30, lado), (255, 255, 255))
    par.paste(Image.fromarray(normal), (0, 0))
    par.paste(Image.fromarray(simulada), (lado + 30, 0))
    st.image(par, width="stretch")
    st.markdown('<div class="pies"><span>Visión normal</span><span>Con deuteranopia</span></div>'
                '<p class="pie-lamina">¿Ves el 7 en la primera? En la segunda desaparece.</p>',
                unsafe_allow_html=True)

st.markdown('<h2 class="seccion">Tres formas de ver el color</h2>', unsafe_allow_html=True)
st.markdown('<p class="ayuda">Cada franja muestra los mismos seis colores: arriba con visión normal, '
            'abajo como los percibe cada tipo de daltonismo.</p>', unsafe_allow_html=True)
cols = st.columns(3, gap="medium")
for col, (clave, (nombre, texto)) in zip(cols, TIPOS.items()):
    simulados = [simular_un_color(c, MATRICES[clave]) for c in PALETA]
    col.markdown(f'<div class="tipo"><h3>{nombre}</h3><p>{texto}</p>{franja(PALETA)}{franja(simulados)}'
                 f'<div class="leyenda"><span>Normal arriba</span><span>{nombre} abajo</span></div></div>',
                 unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Subir captura
# ---------------------------------------------------------------------------
st.markdown('<h2 class="seccion">Revisa tu tienda</h2>', unsafe_allow_html=True)
st.markdown('<p class="ayuda">Funciona mejor si recortas la parte importante: un botón, un precio de oferta '
            'o un mensaje de error.</p>', unsafe_allow_html=True)
archivo = st.file_uploader("Captura de tu tienda (PNG o JPG)", type=["png", "jpg", "jpeg"])
if st.button("Usar una tienda de ejemplo"):
    st.session_state["ejemplo"] = True

if archivo is not None:
    original = Image.open(archivo).convert("RGB")
elif st.session_state.get("ejemplo"):
    original = tienda_ejemplo()
else:
    st.markdown('<p class="firma">Proyecto ABPro de Funciones y Matrices, INACAP Santiago Centro.</p>', unsafe_allow_html=True)
    st.stop()

arreglo = np.array(original)
alto, ancho, _ = arreglo.shape
with st.spinner("Multiplicando cada píxel por las matrices de daltonismo..."):
    simulaciones, colores = procesar(arreglo.tobytes(), alto, ancho)

# ---------------------------------------------------------------------------
# Resultados
# ---------------------------------------------------------------------------
st.markdown('<h2 class="seccion">Así ven tu tienda</h2>', unsafe_allow_html=True)
pixeles = f"{ancho * alto:,}".replace(",", ".")
st.markdown(f'<p class="ayuda">Tu captura tiene {pixeles} píxeles. Cada uno es un vector (R, G, B) que '
            f'se multiplicó por una matriz de 3 × 3.</p>', unsafe_allow_html=True)
opciones = list(MATRICES.keys())
elegido = st.segmented_control("Tipo de daltonismo", opciones, default=opciones[1],
                               format_func=lambda k: TIPOS[k][0])
elegido = elegido or opciones[1]
c1, c2 = st.columns(2, gap="medium")
c1.image(original, caption="Visión normal", width="stretch")
c2.image(simulaciones[elegido], caption=f"Con {TIPOS[elegido][0].lower()}", width="stretch")

lista = [c for c, _ in colores]

st.markdown('<h2 class="seccion">Colores que se confunden</h2>', unsafe_allow_html=True)
alertas = analizar_colores(lista) if len(lista) >= 2 else []
alertas.sort(key=lambda a: a[4] / a[3])   # primero las más graves (la diferencia que más baja)
if not alertas:
    st.markdown('<p class="consejo">✓ No encontramos colores que se confundan con daltonismo.</p>',
                unsafe_allow_html=True)
for col1, col2, nombre, d_normal, d_sim in alertas:
    M = MATRICES[nombre]
    s1, s2 = simular_un_color(col1, M), simular_un_color(col2, M)
    baja = 100 * (1 - d_sim / d_normal)
    st.markdown(
        f'<div class="alerta"><div class="pares">{sw(col1)}{sw(col2)}<span class="flecha">se ven como</span>'
        f'{sw(s1)}{sw(s2)}</div><p>Con <b>{TIPOS[nombre][0].lower()}</b>, {col1} y {col2} quedan muy '
        f'parecidos: su diferencia baja un <b>{baja:.0f}&nbsp;%</b>.</p></div>', unsafe_allow_html=True)
if alertas:
    st.markdown('<p class="consejo">No dependas solo del color. Agrega un texto o un ícono, '
                'como ✓ Disponible o ✗ Agotado.</p>', unsafe_allow_html=True)

st.markdown('<h2 class="seccion">Contraste de texto y fondo</h2>', unsafe_allow_html=True)
st.markdown(f'<p class="ayuda">La norma WCAG 2.1 pide un contraste de al menos {coma(CONTRASTE_MINIMO_TEXTO, 1)} '
            f'para texto normal y {CONTRASTE_MINIMO_GRANDE:.0f} para texto grande. Revisa los pares que '
            f'corresponden a un texto sobre su fondo.</p>', unsafe_allow_html=True)
if len(lista) < 2:
    st.markdown('<p class="consejo">La captura tiene muy pocos colores para comparar. '
                'Prueba con otra parte de la página.</p>', unsafe_allow_html=True)
else:
    pares = sorted(((lista[i], lista[j], contraste(lista[i], lista[j]))
                    for i in range(len(lista)) for j in range(i + 1, len(lista))), key=lambda x: x[2])
    filas = "".join(f"<tr><td>{sw(a)}{a}</td><td>{sw(b)}{b}</td><td class='num'>{coma(r)}</td>"
                    f"<td>{estado(r >= CONTRASTE_MINIMO_TEXTO)}</td><td>{estado(r >= CONTRASTE_MINIMO_GRANDE)}</td></tr>"
                    for a, b, r in pares)
    st.markdown('<div style="overflow-x:auto"><table class="dc"><tr><th>Color 1</th><th>Color 2</th>'
                '<th>Contraste</th><th>Texto normal</th><th>Texto grande</th></tr>' + filas + "</table></div>",
                unsafe_allow_html=True)

with st.expander("¿Cómo funciona? La matemática detrás de DaleColor"):
    st.markdown("""
**El color es un vector.** Cada píxel es c = (R, G, B). El rojo puro es (255, 0, 0).

**El daltonismo es una matriz.** Para ver un color como lo ve una persona daltónica, se multiplica su vector
por una matriz de 3 × 3: c' = M · c (Machado et al., 2009).

**El contraste se calcula con una sumatoria.** La luminancia es L = Σ wᵢ · cᵢ = 0,2126 · R + 0,7152 · G + 0,0722 · B,
y el contraste entre dos colores es (L del claro + 0,05) / (L del oscuro + 0,05).
""")
    if lista:
        st.markdown("Luminancia de tus colores principales: " +
                    " &nbsp; ".join(f"{sw(c)}L = {coma(luminancia(c), 3)}" for c in lista), unsafe_allow_html=True)

st.markdown('<p class="firma">Proyecto ABPro de Funciones y Matrices, INACAP Santiago Centro.</p>', unsafe_allow_html=True)
