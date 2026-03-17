import streamlit as st
import cv2
import numpy as np

st.set_page_config(layout="wide")
st.title("🕵️ Image Manipulation Detection System")

# -------------------------------
# DETECTION FUNCTION (STRICT + STABLE)
# -------------------------------
def detect_forgery(original, processed):

    img1 = cv2.resize(original, (300, 300))
    img2 = cv2.resize(processed, (300, 300))

    # 🔥 Safety: if images are exactly same → ORIGINAL
    if np.array_equal(img1, img2):
        return False, 0.0

    # -------------------------------
    # 1. PIXEL DIFFERENCE
    # -------------------------------
    diff = cv2.absdiff(img1, img2)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    pixel_change = np.mean(gray_diff)

    # -------------------------------
    # 2. EDGE DIFFERENCE (blur detection)
    # -------------------------------
    edges1 = cv2.Canny(img1, 100, 200)
    edges2 = cv2.Canny(img2, 100, 200)

    edge_diff = np.mean(cv2.absdiff(edges1, edges2))

    # -------------------------------
    # FAKE SCORE (NORMALIZED)
    # -------------------------------
    score = 0

    score += pixel_change * 2.5
    score += edge_diff * 1.5

    # normalize to 0–100
    score = min(score, 100)

    # 🔥 STRICT RULE
    is_fake = score > 1

    return is_fake, score


# -------------------------------
# TRANSFORMATIONS
# -------------------------------
def negative(img): return 255 - img

def log_transform(img):
    img_float = img.astype(np.float32)
    c = 255 / np.log(1 + np.max(img_float))
    return np.uint8(c * np.log(1 + img_float))

def gamma_transform(img, gamma=0.5):
    table = np.array([(i / 255.0) ** gamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)

def histogram_equalization(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def blur(img): return cv2.GaussianBlur(img, (11,11), 0)

def edge_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(cv2.Canny(gray, 100, 200), cv2.COLOR_GRAY2BGR)

def sharpen(img):
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(img, -1, kernel)

def noise(img):
    noise = np.random.randint(0, 20, img.shape, dtype='uint8')
    return cv2.add(img, noise)

def rotate(img):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), 15, 1)
    return cv2.warpAffine(img, M, (w, h))

def flip(img): return cv2.flip(img, 1)

def translate(img):
    h, w = img.shape[:2]
    M = np.float32([[1, 0, int(w*0.1)], [0, 1, int(h*0.1)]])
    return cv2.warpAffine(img, M, (w, h))


# -------------------------------
# UI
# -------------------------------
uploaded = st.file_uploader("📤 Upload Image", type=["jpg","png","jpeg"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    original = cv2.imdecode(file_bytes, 1)

    if "img" not in st.session_state:
        st.session_state.img = original.copy()

    if "result" not in st.session_state:
        st.session_state.result = None
        st.session_state.score = 0

    col1, col2, col3 = st.columns([1,2,2])

    with col1:
        option = st.selectbox("Choose Operation", [
            "No Operation",
            "Negative",
            "Log Transformation",
            "Gamma Transformation",
            "Histogram Equalization",
            "Blur",
            "Edge Detection",
            "Sharpen",
            "Noise",
            "Rotate",
            "Flip",
            "Translate"
        ])

        if st.button("Apply"):
            if option == "No Operation":
                st.session_state.img = original.copy()
            elif option == "Negative":
                st.session_state.img = negative(original)
            elif option == "Log Transformation":
                st.session_state.img = log_transform(original)
            elif option == "Gamma Transformation":
                st.session_state.img = gamma_transform(original)
            elif option == "Histogram Equalization":
                st.session_state.img = histogram_equalization(original)
            elif option == "Blur":
                st.session_state.img = blur(original)
            elif option == "Edge Detection":
                st.session_state.img = edge_detection(original)
            elif option == "Sharpen":
                st.session_state.img = sharpen(original)
            elif option == "Noise":
                st.session_state.img = noise(original)
            elif option == "Rotate":
                st.session_state.img = rotate(original)
            elif option == "Flip":
                st.session_state.img = flip(original)
            elif option == "Translate":
                st.session_state.img = translate(original)

        if st.button("Detect"):
            is_fake, score = detect_forgery(original, st.session_state.img)
            st.session_state.result = is_fake
            st.session_state.score = score

    with col2:
        st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), width='stretch')

    with col3:
        st.image(cv2.cvtColor(st.session_state.img, cv2.COLOR_BGR2RGB), width='stretch')

    # -------------------------------
    # FINAL RESULT
    # -------------------------------
    if st.session_state.result is not None:

        if st.session_state.result:
            st.error(f"FAKE IMAGE DETECTED ({st.session_state.score:.1f}%)")
        else:
            st.success(f"ORIGINAL IMAGE ({st.session_state.score:.1f}%)")
