# dashboard/app.py

import os
import json

import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import requests  # <-- nieuw: om de FastAPI aan te roepen
from tensorflow.keras.models import load_model

# ==============================
# Basisconfiguratie
# ==============================
st.set_page_config(
    page_title="Surface Crack Anomaly Detection",
    layout="wide"
)

# ==============================
# Paden naar bestanden
# ==============================
MODEL_PATH = "models/saved_models/final_autoencoder.h5"
THRESHOLD_PATH = "models/saved_models/threshold.npy"
METRICS_JSON = "models/saved_models/metrics.json"

# API-config
API_BASE_URL = "http://127.0.0.1:8000"
API_PREDICT_URL = f"{API_BASE_URL}/predict"
API_HEALTH_URL = f"{API_BASE_URL}/health"

# Fallback threshold als threshold.npy ontbreekt
FALLBACK_THRESHOLD = 0.015


# ==============================
# Model + threshold + metrics laden
# ==============================

@st.cache_resource
def get_model():
    """Laad het getrainde autoencoder-model (zonder opnieuw te compileren)."""
    model = load_model(MODEL_PATH, compile=False)
    return model


@st.cache_resource
def get_threshold():
    """Laad de anomaly-threshold, of gebruik een standaardwaarde als het bestand ontbreekt."""
    if os.path.exists(THRESHOLD_PATH):
        return float(np.load(THRESHOLD_PATH))
    return FALLBACK_THRESHOLD


@st.cache_resource
def get_metrics():
    """Laad evaluatiemetrics van de testset (indien beschikbaar)."""
    if os.path.exists(METRICS_JSON):
        with open(METRICS_JSON, "r") as f:
            return json.load(f)
    return None


@st.cache_resource
def api_available() -> bool:
    """Check of de FastAPI-backend bereikbaar is."""
    try:
        resp = requests.get(API_HEALTH_URL, timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


autoencoder = get_model()
threshold = get_threshold()
metrics = get_metrics()
threshold_file_exists = os.path.exists(THRESHOLD_PATH)
backend_online = api_available()


# ==============================
# Hulpfuncties
# ==============================

def preprocess_image(img_bgr, size=(64, 64)):
    """Resize naar target resolutie en normaliseer naar [0,1]."""
    img_resized = cv2.resize(img_bgr, size)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype("float32") / 255.0
    return img_norm  # shape (H, W, 3)


def compute_reconstruction(img_norm):
    """Laat de autoencoder een reconstructie van het beeld maken."""
    x = np.expand_dims(img_norm, axis=0)  # (1, H, W, 3)
    recon = autoencoder.predict(x, verbose=0)[0]
    return recon


def compute_mse(original, recon):
    """Mean Squared Error tussen origineel en reconstructie."""
    return float(np.mean((original - recon) ** 2))


def make_heatmap(original, recon):
    """Absolute reconstructiefout per pixel → genormaliseerde heatmap (H, W) in [0,1]."""
    diff = np.abs(original - recon)  # (H, W, C)
    heat = diff.mean(axis=2)         # gemiddelde over kanalen
    if heat.max() > 0:
        heat = heat / heat.max()
    return heat


def apply_blur_on_heatmap(img_uint8, heatmap, thr=0.6, ksize=(25, 25)):
    """
    Blur alleen de gebieden waar de reconstruction error hoog is.

    img_uint8 : origineel beeld in uint8 (H, W, 3)
    heatmap   : error heatmap in [0,1]
    thr       : drempel voor 'hoog error gebied'
    """
    # Binaire mask op basis van heatmap
    mask = (heatmap >= thr).astype("uint8")
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Blur het hele beeld
    blurred = cv2.GaussianBlur(img_uint8, ksize, 0)

    # Mask naar 3 kanalen
    mask3 = np.stack([mask] * 3, axis=-1)

    # Combineer: waar mask==1 → blurred, anders origineel
    out = (mask3 * blurred + (1 - mask3) * img_uint8).astype("uint8")
    return out


def call_backend_api(file_bytes: bytes):
    """
    Stuur de afbeelding naar de FastAPI-backend.
    Geeft een dict terug met mse, threshold, is_anomaly, label.
    """
    try:
        files = {
            "file": ("upload.jpg", file_bytes, "image/jpeg")
        }
        resp = requests.post(API_PREDICT_URL, files=files, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.sidebar.error(f"API-call mislukt: {e}")
        return None


# ==============================
# Streamlit UI
# ==============================

st.title("🧠 Surface Crack Anomaly Detection Dashboard")
st.write(
    "Deze applicatie gebruikt een **convolutional autoencoder** die is getraind op "
    "**normale (niet-gescheurde)** betonnen oppervlakken. "
    "Afwijkingen, zoals scheuren, leiden tot een hogere reconstructiefout (MSE) en "
    "worden als anomalie gedetecteerd."
)

# Sidebar: modelinformatie
st.sidebar.header("ℹ️ Modelinformatie")

# API-status
if backend_online:
    st.sidebar.success("Backend API: **online** (FastAPI op poort 8000)")
else:
    st.sidebar.warning(
        "Backend API: **offline**. Dashboard gebruikt tijdelijk alleen het lokale model."
    )

if threshold_file_exists:
    st.sidebar.markdown(
        f"**Threshold** (95e percentiel validatie-errors): `{threshold:.6f}`"
    )
else:
    st.sidebar.warning(
        "⚠️ `threshold.npy` niet gevonden.\n\n"
        f"Er wordt nu een **standaard threshold** gebruikt: `{FALLBACK_THRESHOLD:.6f}`.\n"
        "Draai eventueel `python src/train.py` opnieuw om de berekende threshold "
        "op te slaan in `models/saved_models/threshold.npy`."
    )

if metrics is not None:
    st.sidebar.subheader("📈 Testset-metrics")
    st.sidebar.markdown(f"- **Accuracy:** `{metrics['accuracy']:.3f}`")
    st.sidebar.markdown(f"- **Precision:** `{metrics['precision']:.3f}`")
    st.sidebar.markdown(f"- **Recall (anomaly):** `{metrics['recall']:.3f}`")
    st.sidebar.markdown(f"- **F1-score:** `{metrics['f1']:.3f}`")
    st.sidebar.markdown(
        f"- **MSE normal/anomaly:** `{metrics['mse_normal']:.5f}` / "
        f"`{metrics['mse_anomaly']:.5f}`"
    )
else:
    st.sidebar.info(
        "Geen `metrics.json` gevonden. Draai `python src/train.py` om de "
        "evaluatiemetrics op te slaan."
    )

# Optie om blur te tonen
show_blur = st.sidebar.checkbox(
    "🔍 Toon geblurde anomaly (markeer foutregio)",
    value=True,
    help="Accentueer de gebieden waar de reconstructiefout het hoogst is."
)

# Uploadsectie
st.markdown("## 1. Upload een afbeelding")
uploaded_file = st.file_uploader("Selecteer een afbeelding (JPG of PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Bytes (voor API én OpenCV)
    file_bytes_raw = uploaded_file.getvalue()
    file_bytes = np.asarray(bytearray(file_bytes_raw), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("De afbeelding kon niet worden ingelezen.")
    else:
        # Voorbewerking
        img_norm = preprocess_image(img_bgr)          # float32 in [0,1]
        img_disp = (img_norm * 255).astype("uint8")   # voor weergave

        # Reconstructie + fout (LOKAAL)
        recon = compute_reconstruction(img_norm)
        local_mse = compute_mse(img_norm, recon)
        local_is_anomaly = local_mse > threshold

        # Default waarden (vallen terug op lokaal model)
        mse = local_mse
        used_threshold = threshold
        is_anomaly = local_is_anomaly
        label = "ANOMALY (CRACK)" if is_anomaly else "NORMAL"
        source = "lokaal model"

        # PRODUCTION: call backend API als die online is
        if backend_online:
            api_result = call_backend_api(file_bytes_raw)
            if api_result is not None and api_result.get("mse", -1) >= 0:
                mse = api_result.get("mse", mse)
                used_threshold = api_result.get("threshold", used_threshold)
                is_anomaly = bool(api_result.get("is_anomaly", is_anomaly))
                label = api_result.get("label", label)
                source = "FastAPI-backend"

        # Heatmap
        heatmap = make_heatmap(img_norm, recon)

        # Blur (optioneel)
        blurred_img = None
        if is_anomaly and show_blur:
            blurred_img = apply_blur_on_heatmap(img_disp, heatmap, thr=0.6)

        # Visuele analyse
        st.markdown("## 2. Visuele analyse")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("### Origineel (64×64)")
            st.image(img_disp, caption="Ingevoerde afbeelding", use_container_width=True)

        with col2:
            st.markdown("### Reconstructie")
            st.image(recon, caption="Output van de autoencoder", use_container_width=True)

        with col3:
            st.markdown("### Error heatmap")
            fig, ax = plt.subplots()
            ax.imshow(heatmap, cmap="inferno")
            ax.axis("off")
            st.pyplot(fig, use_container_width=True)
            st.caption("Genormaliseerde reconstructiefout per pixel")

        with col4:
            st.markdown("### Geblurde anomaly")
            if blurred_img is not None:
                st.image(
                    blurred_img,
                    caption="Regio’s met hoge reconstructiefout zijn geblurd",
                    use_container_width=True,
                )
            else:
                st.info(
                    "Geen duidelijke anomaly gedetecteerd of de blur-optie is uitgeschakeld."
                )

        # Resultaatblok
        st.markdown("## 3. Resultaat")

        st.markdown(f"- **MSE-score (bron: {source}):** `{mse:.6f}`")
        st.markdown(f"- **Threshold:** `{used_threshold:.6f}`")
        st.markdown(f"- **Classificatie:** `{label}`")

        if is_anomaly:
            st.error(
                "🚨 **ANOMALIE GEDTECTEERD – Waarschijnlijk een crack.**  \n"
                "De reconstructiefout ligt boven de ingestelde threshold. Dit beeld wordt daarom "
                "als *anomalie* geclassificeerd."
            )
        else:
            st.success(
                "✅ **NORMAAL – Geen duidelijke crack gedetecteerd.**  \n"
                "De reconstructiefout ligt onder de threshold. Het patroon lijkt op de "
                "trainingsbeelden van normale betonoppervlakken."
            )

        # Uitlegsectie
        st.markdown("## 4. Toelichting op de methode")
        st.write(
            """
            - De autoencoder is uitsluitend getraind op **normale** betonbeelden (Negative-klasse).
            - Bij een normaal beeld kan het model de input goed reconstrueren → de **MSE is laag**.
            - Bij een scheur of andere afwijking lukt de reconstructie minder goed → de **MSE stijgt**.
            - De threshold is gekozen als de **95e percentiel** van de reconstructie-errors op de
              validatieset. Alles daarboven wordt als **anomalie** beschouwd.
            - De error heatmap laat zien **waar** de reconstructiefout hoog is; deze gebieden
              corresponderen vaak met de scheur.
            - Het dashboard stuurt de afbeelding naar een aparte **FastAPI-backend**. Die berekent
              de MSE, vergelijkt die met de threshold en stuurt het label terug naar deze UI.
            - Als de backend niet bereikbaar is, valt het dashboard automatisch terug op het
              lokale model voor de classificatie.
            """
        )
else:
    st.info("Upload een afbeelding om een voorspelling en visualisatie te krijgen.")