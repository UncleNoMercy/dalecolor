"""
DaleColor - Núcleo del simulador de daltonismo
====================================================
Proyecto ABPro - Funciones y Matrices (MTFM01)

Qué hace este archivo:
    1. Carga una imagen y la convierte en una matriz de píxeles (vectores RGB).
    2. Aplica la corrección gamma (función por tramos) para pasar a "luz real".
    3. Multiplica cada píxel por la matriz 3x3 de cada tipo de daltonismo.
    4. Devuelve la imagen a colores de pantalla y muestra los resultados.
    5. Busca los colores principales de la imagen y, con sumatorias, calcula
       luminancia, contraste (norma WCAG 2.1) y distancia entre colores para
       avisar qué pares de colores se confunden con cada tipo de daltonismo.

Cómo usarlo (en la terminal de VS Code):
    py dalecolor.py                  -> usa una imagen de prueba generada
    py dalecolor.py mi_captura.png   -> usa su propia imagen
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# MATRICES DE SIMULACIÓN (Machado, Oliveira y Fernandes, 2009)
# Cada matriz transforma un vector de color (R, G, B) en el color que percibe
# una persona con ese tipo de daltonismo (severidad máxima = 1.0).
# ---------------------------------------------------------------------------
MATRICES = {
    "Protanopia (sin rojo)": np.array([
        [0.152286, 1.052583, -0.204868],
        [0.114503, 0.786281, 0.099216],
        [-0.003882, -0.048116, 1.051998],
    ]),
    "Deuteranopia (sin verde)": np.array([
        [0.367322, 0.860646, -0.227968],
        [0.280085, 0.672501, 0.047413],
        [-0.011820, 0.042940, 0.968881],
    ]),
    "Tritanopia (sin azul)": np.array([
        [1.255528, -0.076749, -0.178779],
        [-0.078411, 0.930809, 0.147602],
        [0.004733, 0.691367, 0.303900],
    ]),
}


# ---------------------------------------------------------------------------
# PASO 1: VECTORES Y MATRICES - cargar la imagen
# ---------------------------------------------------------------------------
def cargar_imagen(ruta):
    """Abre una imagen y la devuelve como matriz de alto x ancho x 3 (RGB, 0 a 255)."""
    imagen = Image.open(ruta).convert("RGB")  # quitamos transparencia si la tiene
    return np.array(imagen, dtype=float)


# ---------------------------------------------------------------------------
# PASO 2: FUNCIONES - corrección gamma (función por tramos)
# ---------------------------------------------------------------------------
def srgb_a_lineal(valores):
    """
    Convierte valores de pantalla (0 a 255) a valores de luz real (0 a 1).
        f(c) = c / 12.92                      si c <= 0.04045
        f(c) = ((c + 0.055) / 1.055) ^ 2.4    si c >  0.04045
    """
    c = valores / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lineal_a_srgb(valores):
    """Función inversa: de luz real (0 a 1) vuelve a valores de pantalla (0 a 255)."""
    c = np.clip(valores, 0.0, 1.0)  # la matriz puede dar valores fuera de rango
    c = np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)
    return np.round(c * 255.0)


# ---------------------------------------------------------------------------
# PASO 3: MATRICES - simular el daltonismo
# ---------------------------------------------------------------------------
def simular_daltonismo(imagen, matriz):
    """
    Aplica la matriz de daltonismo a TODOS los píxeles de la imagen.

    La imagen (alto x ancho x 3) se reordena como una matriz P de N filas y
    3 columnas, donde N = alto x ancho y cada fila es un vector (R, G, B).
    Así toda la imagen se procesa con un solo producto matricial:
        P' = P · M^T
    (multiplicar cada fila por M^T es lo mismo que hacer M · c para cada píxel)
    """
    alto, ancho, _ = imagen.shape
    P = srgb_a_lineal(imagen).reshape(-1, 3)      # matriz de N x 3
    P_simulada = P @ matriz.T                     # producto matricial
    resultado = lineal_a_srgb(P_simulada)
    return resultado.reshape(alto, ancho, 3)      # volvemos a la forma de imagen


def simular_un_color(rgb, matriz):
    """Simula un solo color, útil para revisar el cálculo a mano."""
    c = srgb_a_lineal(np.array(rgb, dtype=float))  # vector de 3 componentes
    return tuple(int(v) for v in lineal_a_srgb(matriz @ c))  # M · c


# ---------------------------------------------------------------------------
# PASO 4: SUMATORIAS - luminancia, contraste y distancia
# ---------------------------------------------------------------------------
# Pesos de la luminancia según WCAG 2.1: el ojo es más sensible al verde.
PESOS_LUMINANCIA = np.array([0.2126, 0.7152, 0.0722])

# Criterios de evaluación
CONTRASTE_MINIMO_TEXTO = 4.5   # WCAG 2.1, texto normal (nivel AA)
CONTRASTE_MINIMO_GRANDE = 3.0  # WCAG 2.1, texto grande y elementos gráficos
DISTANCIA_MINIMA = 100         # criterio del grupo: bajo esto, colores "confundibles"


def luminancia(rgb):
    """
    Luminancia relativa (0 = negro, 1 = blanco), como suma ponderada:
        L = Σ (i=1 a 3) w_i · c_i = 0.2126·R + 0.7152·G + 0.0722·B
    donde c_i son las componentes ya linealizadas con la corrección gamma.
    """
    c = srgb_a_lineal(np.array(rgb, dtype=float))
    L = 0.0
    for i in range(3):                 # la sumatoria escrita paso a paso
        L += PESOS_LUMINANCIA[i] * c[i]
    return L


def contraste(rgb1, rgb2):
    """Razón de contraste WCAG: (L_claro + 0.05) / (L_oscuro + 0.05). Va de 1 a 21."""
    L1, L2 = luminancia(rgb1), luminancia(rgb2)
    claro, oscuro = max(L1, L2), min(L1, L2)
    return (claro + 0.05) / (oscuro + 0.05)


def distancia(rgb1, rgb2):
    """
    Distancia (norma) entre dos vectores de color:
        d = raíz( Σ (i=1 a 3) (c1_i - c2_i)^2 )
    Mientras más chica, más se parecen los colores.
    """
    suma = 0.0
    for i in range(3):
        suma += (rgb1[i] - rgb2[i]) ** 2
    return suma ** 0.5


def colores_principales(imagen, cantidad=8, porcentaje_minimo=0.5, distancia_grupo=40):
    """
    Encuentra los colores que más se repiten en la imagen (fondos, botones).

    En una captura real, un mismo color aparece con pequeñas variaciones
    (por ejemplo, el blanco como 255, 254, 253...) por la compresión, las
    sombras y los bordes suavizados. Por eso primero AGRUPAMOS los colores
    parecidos y tomamos el color promedio de cada grupo.

    Pasos:
      1. Se achica la imagen para que el cálculo sea rápido.
      2. Pillow agrupa los píxeles en 16 colores (método "median cut").
      3. Se juntan los grupos cuya distancia entre vectores es menor que
         'distancia_grupo' (son el "mismo" color para una persona).
      4. Se ignoran los grupos que ocupan menos del 'porcentaje_minimo'.
    """
    img = Image.fromarray(np.clip(imagen, 0, 255).astype(np.uint8))
    img.thumbnail((500, 500))                                   # paso 1
    agrupada = img.quantize(colors=16, method=Image.Quantize.MEDIANCUT,
                            dither=Image.Dither.NONE)           # paso 2
    paleta = agrupada.getpalette()
    total = img.width * img.height
    grupos = []                                                 # [color, porcentaje]
    for conteo, indice in sorted(agrupada.getcolors(), reverse=True):
        color = tuple(paleta[3 * indice: 3 * indice + 3])
        porcentaje = 100 * conteo / total
        for g in grupos:                                        # paso 3
            if distancia(g[0], color) < distancia_grupo:
                g[1] += porcentaje
                break
        else:
            grupos.append([color, porcentaje])
    resultado = [(tuple(int(v) for v in c), p) for c, p in grupos if p >= porcentaje_minimo]  # paso 4
    resultado.sort(key=lambda x: x[1], reverse=True)
    return resultado[:cantidad]


def analizar_colores(colores):
    """
    Revisa todos los pares de colores principales y devuelve las alertas:
      - contraste bajo la norma WCAG (en visión normal)
      - colores que se confunden con cada tipo de daltonismo
    """
    alertas = []
    for a in range(len(colores)):
        for b in range(a + 1, len(colores)):
            c1, c2 = colores[a], colores[b]
            d_normal = distancia(c1, c2)
            for nombre, matriz in MATRICES.items():
                s1, s2 = simular_un_color(c1, matriz), simular_un_color(c2, matriz)
                d_sim = distancia(s1, s2)
                # Solo alertamos si con visión normal se distinguían bien,
                # pero con daltonismo quedan bajo la distancia mínima.
                if d_normal >= DISTANCIA_MINIMA and d_sim < DISTANCIA_MINIMA:
                    alertas.append((c1, c2, nombre, d_normal, d_sim))
    return alertas


def imprimir_reporte(colores_con_porcentaje):
    """Muestra en la terminal el análisis completo de los colores principales."""
    colores = [c for c, _ in colores_con_porcentaje]

    print("\n" + "=" * 70)
    print("REPORTE DE ACCESIBILIDAD - DaleColor")
    print("=" * 70)

    print("\nColores principales de la imagen:")
    for i, (c, pct) in enumerate(colores_con_porcentaje, start=1):
        print(f"  {i}. RGB{c}  ocupa {pct:5.1f}%  luminancia = {luminancia(c):.3f}")

    print("\nContraste entre pares de colores (WCAG 2.1):")
    print(f"  {'Color 1':<17}{'Color 2':<17}{'Contraste':>10}   Texto normal  Texto grande")
    for a in range(len(colores)):
        for b in range(a + 1, len(colores)):
            r = contraste(colores[a], colores[b])
            normal = "cumple" if r >= CONTRASTE_MINIMO_TEXTO else "NO cumple"
            grande = "cumple" if r >= CONTRASTE_MINIMO_GRANDE else "NO cumple"
            print(f"  {str(colores[a]):<17}{str(colores[b]):<17}{r:>8.2f}:1   {normal:<13} {grande}")

    alertas = analizar_colores(colores)
    print(f"\nColores que se confunden con daltonismo (distancia < {DISTANCIA_MINIMA}):")
    if not alertas:
        print("  Ninguno. ¡Bien!")
    for c1, c2, nombre, d_normal, d_sim in alertas:
        baja = 100 * (1 - d_sim / d_normal)
        print(f"  ALERTA {nombre}: {c1} y {c2}")
        print(f"         distancia {d_normal:.0f} -> {d_sim:.0f} (baja un {baja:.0f}%)")
    print("=" * 70)


# ---------------------------------------------------------------------------
# IMAGEN DE PRUEBA: botones típicos de una app
# ---------------------------------------------------------------------------
def crear_imagen_prueba(ruta="imagen_prueba.png"):
    """Dibuja una interfaz simple con botones de colores para probar el programa."""
    img = Image.new("RGB", (600, 300), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([40, 40, 280, 120], fill=(220, 50, 50))     # botón rojo: Cancelar
    d.rectangle([320, 40, 560, 120], fill=(50, 160, 50))    # botón verde: Confirmar
    d.rectangle([40, 170, 280, 250], fill=(30, 100, 220))   # botón azul
    d.rectangle([320, 170, 560, 250], fill=(240, 200, 30))  # botón amarillo
    d.text((125, 75), "CANCELAR", fill=(255, 255, 255))
    d.text((405, 75), "CONFIRMAR", fill=(255, 255, 255))
    d.text((130, 205), "AYUDA", fill=(255, 255, 255))
    d.text((410, 205), "AVISO", fill=(0, 0, 0))
    img.save(ruta)
    return ruta


# ---------------------------------------------------------------------------
# PROGRAMA PRINCIPAL
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) > 1:
        ruta = sys.argv[1]
    else:
        ruta = crear_imagen_prueba()
        print(f"No se indicó imagen; se creó '{ruta}' para probar.")

    imagen = cargar_imagen(ruta)
    alto, ancho, _ = imagen.shape
    print(f"Imagen cargada: {ancho} x {alto} píxeles = {alto * ancho:,} vectores RGB")

    # Comprobación con el ejemplo de la Tabla 1 (botón rojo vs. botón verde)
    print("\nEjemplo de la Tabla 1 con protanopia:")
    M = MATRICES["Protanopia (sin rojo)"]
    for nombre, rgb in [("Rojo ", (220, 50, 50)), ("Verde", (50, 160, 50))]:
        print(f"  {nombre} {rgb} -> {simular_un_color(rgb, M)}")

    # Reporte de contraste y colores confundibles (sumatorias)
    imprimir_reporte(colores_principales(imagen))

    # Simulación de los tres tipos y gráfico comparativo 2 x 2
    # El tamaño de la figura se ajusta a la forma de la imagen (ancha o alta),
    # para que los títulos no queden encima de las imágenes.
    ancho_panel = 7
    alto_panel = ancho_panel * alto / ancho
    fig, ejes = plt.subplots(2, 2, figsize=(2 * ancho_panel, 2 * alto_panel + 1.2),
                             layout="constrained")
    ejes = ejes.ravel()
    ejes[0].imshow(imagen.astype(np.uint8))
    ejes[0].set_title("Visión normal")

    for eje, (nombre, matriz) in zip(ejes[1:], MATRICES.items()):
        simulada = simular_daltonismo(imagen, matriz)
        eje.imshow(simulada.astype(np.uint8))
        eje.set_title(nombre)

    for eje in ejes:
        eje.axis("off")

    fig.suptitle("DaleColor: simulación de daltonismo", fontsize=14)
    fig.savefig("resultado_simulacion.png", dpi=120)
    print("\nResultado guardado en 'resultado_simulacion.png'")
    plt.show()


if __name__ == "__main__":
    main()
