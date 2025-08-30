import streamlit as st
from PIL import Image
import torch
import numpy as np
from retinaface import RetinaFace
from facenet_pytorch import InceptionResnetV1
import cv2
import base64       
from io import BytesIO
from annotated_text import annotated_text

# -------------------------
# Face Detection Function
# -------------------------
facenet = InceptionResnetV1(pretrained='vggface2').eval()

def detect_faces_and_draw(image: Image.Image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

   
    faces = RetinaFace.detect_faces(img)

    
    

    for key in faces.keys():
        x1, y1, x2, y2 = faces[key]['facial_area']
        face = img[y1:y2, x1:x2]

        face_resized = cv2.resize(face, (160, 160))
        face_tensor = torch.tensor(face_resized).float().permute(2, 0, 1) / 255.0
        face_tensor = (face_tensor - 0.5) / 0.5
        face_tensor = face_tensor.unsqueeze(0)

        with torch.no_grad():
            _ = facenet(face_tensor)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4)

    final_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(final_rgb), len(faces)

# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(layout="wide")

def get_base64_img(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Load and convert logo.png
# image1 = Image.open("logo.png")
# encoded_image1 = get_base64_img(image1)

sidebar_logo = "Vibyn3.png"  # Path to your logo image
main_body_logo = "logo.png"
st.logo(sidebar_logo, icon_image=main_body_logo, size="medium")  

# Load and convert Vibyn1.png
image2 = Image.open("Vibyn7.png")
encoded_image2 = get_base64_img(image2)

# Disable right-click globally (optional)
st.markdown("""
    <script>
        document.addEventListener('contextmenu', event => event.preventDefault());
    </script>
""", unsafe_allow_html=True)





# Display main image with all interaction disabled

def home():
    st.markdown(
        f"""
        <style>
        .no-interaction {{
            pointer-events: none;
            user-select: none;
        }}
        img {{
            -webkit-user-drag: none;
            user-drag: none;
        }}
        </style>
        <div class="no-interaction">
            <img src="data:image/png;base64,{encoded_image2}" style="width: 100%;">
        </div>
        """,
        unsafe_allow_html=True
    )


    st.header("")

    space1, col1, space2 = st.columns([1, 3, 1])

    with col1:
        with st.container():
            st.markdown(
                """
                <style>
                .centered-annotated-text {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            st.markdown('<div class="centered-annotated-text">', unsafe_allow_html=True)

            annotated_text(
                "An advanced AI-powered system designed for real-time face ",
                ("detection", "task", "#cc3c3c"), 
                " and ",
                ("recognition", "task", "#006466"),
                ". Leveraging ",
                ("RetinaFace", "model", "#2f0e07"),
                " for accurate facial detection, and ",
                ("Facenet", "model"),
                " (InceptionResnetV1) for high-precision identity matching. The model is suitable for use in security, attendance, or ",
                ("authentication", "use-case"),
                " systems."
            )

            st.markdown('</div>', unsafe_allow_html=True)




    st.header("")

    st.markdown("---")

# -------------------------
# Members
# -------------------------

def crew():
    st.header("")


    st.title("CREW")
    st.header("")

    spacer1, col1, col2, col3, spacer2 = st.columns([2,2,2,2,1])

    with col1:
        st.image("pri_pic.png", width=200)
        st.subheader("Pritam Roy")
        st.markdown("[[GitHub]](https://github.com/pritamroydev)")

    with col2:
        st.image("anw_pic.png", width=200)
        st.subheader("Anwesha Bhaduri [Leader]")
        st.markdown("[[GitHub]](https://github.com/AnweshaBhadury)")

    with col3:
        st.image("som_pic.png", width=200)
        st.subheader("Somnath Chakaraborty")
        st.markdown("[GitHub]")

    st.header("")

    st.markdown("---")

# -------------------------
# AI interaction
# -------------------------

def aion():
    st.header("")

    st.title("AION")

    st.set_page_config(page_title="#", layout="wide")

    st.write("Identify blurred/ low detailed images as male or female ")
    with st.expander("Show Instructions"):
        st.write("""
        1. Upload an image containing faces.
        2. Click on the "Detect Faces" button to identify and draw rectangles around detected faces.
        3. The output image will be displayed on the right side.
        """)


    st.subheader("")


    if "show_image" not in st.session_state:
        st.session_state["show_image"] = False



    col1, col2 = st.columns(2)

    with col1:
        st.write("Input Image")
        upload_img = st.file_uploader("Drop your image or Click to upload", type=["png", "jpg", "jpeg", "webp"], label_visibility="collapsed")

        if upload_img:
            image = Image.open(upload_img).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if upload_img and st.button("Detect Faces"):
                with st.spinner("Processing..."):
                    result_image, num_faces = detect_faces_and_draw(image)

                    st.session_state["result_image"] = result_image
                    st.session_state["show_image"] = True
                    st.session_state["num_faces"] = num_faces

            if "num_faces" in st.session_state:
                st.success(f"✅ Number of faces detected: {st.session_state['num_faces']}")



    with col2:
        st.write("Output Image")

        if st.session_state["show_image"] and "result_image" in st.session_state:
            st.image(st.session_state["result_image"], caption="Detected Faces", use_container_width=True)
        else:
            image_placeholder = st.empty()
            image_placeholder.markdown("""
                <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
                <div style='border: 2px solid #444; border-radius: 10px; background-color: #1e1e1e;
                            width: 100%; height: 400px; display: flex; flex-direction: column;
                            align-items: center; justify-content: center; color: #666; font-size: 18px;'>
                    <span class="material-icons" style="font-size: 30px; color: #888;">image</span>
                    No image yet
                </div>
            """, unsafe_allow_html=True)


# -------------------------
# Side bar
# -------------------------

st.sidebar.title("Navigation")

choice = st.sidebar.button("← Back", ["All", "Home", "Crew", "AION"])

if st.sidebar.button("Home"):
    section = "Home"
elif st.sidebar.button("Crew"):
    section = "Crew"
elif st.sidebar.button("AION"):
    section = "AION"
elif st.sidebar.button("All"):
    section = "All"
else:
    section = "All"  # default


# Render sections
if section == "Home":
    home()
elif section == "Crew":
    crew()
elif section == "AION":
    aion()
elif section == "All":
    home()
    st.markdown("---")
    crew()
    st.markdown("---")
    aion()





# --- Footer ---

st.markdown("---")

    

st.markdown("""
    <!-- Phosphor Icons CDN -->
    <script src="https://unpkg.com/phosphor-icons"></script>

    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        text-align: center;
        padding: 10px 0 5px 0;
        font-size: 15px;
        color: gray;
        background-color: #0e1117;
        z-index: 9999;
    }
    .social-icons {
        margin-top: 5px;
    }
    .social-icons a {
        color: white;
        margin: 0 15px;
        font-size: 5px;
        text-decoration: none;
        transition: color 0.3s;
    }
    .social-icons a:hover {
        color: white;
    }
    </style>

    <div class="footer">
        Made with ❤️ by Team Vibyn • 2025
        <div class="social-icons">
            <a href="#" target="_blank">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="#f1f1f1" viewBox="0 0 256 256"><path d="M208.31,75.68A59.78,59.78,0,0,0,202.93,28,8,8,0,0,0,196,24a59.75,59.75,0,0,0-48,24H124A59.75,59.75,0,0,0,76,24a8,8,0,0,0-6.93,4,59.78,59.78,0,0,0-5.38,47.68A58.14,58.14,0,0,0,56,104v8a56.06,56.06,0,0,0,48.44,55.47A39.8,39.8,0,0,0,96,192v8H72a24,24,0,0,1-24-24A40,40,0,0,0,8,136a8,8,0,0,0,0,16,24,24,0,0,1,24,24,40,40,0,0,0,40,40H96v16a8,8,0,0,0,16,0V192a24,24,0,0,1,48,0v40a8,8,0,0,0,16,0V192a39.8,39.8,0,0,0-8.44-24.53A56.06,56.06,0,0,0,216,112v-8A58.14,58.14,0,0,0,208.31,75.68ZM200,112a40,40,0,0,1-40,40H112a40,40,0,0,1-40-40v-8a41.74,41.74,0,0,1,6.9-22.48A8,8,0,0,0,80,73.83a43.81,43.81,0,0,1,.79-33.58,43.88,43.88,0,0,1,32.32,20.06A8,8,0,0,0,119.82,64h32.35a8,8,0,0,0,6.74-3.69,43.87,43.87,0,0,1,32.32-20.06A43.81,43.81,0,0,1,192,73.83a8.09,8.09,0,0,0,1,7.65A41.72,41.72,0,0,1,200,104Z"></path></svg>
            </a>
            <a href="#" target="_blank">
               <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="#f1f1f1" viewBox="0 0 256 256"><path d="M216,24H40A16,16,0,0,0,24,40V216a16,16,0,0,0,16,16H216a16,16,0,0,0,16-16V40A16,16,0,0,0,216,24Zm0,192H40V40H216V216ZM96,112v64a8,8,0,0,1-16,0V112a8,8,0,0,1,16,0Zm88,28v36a8,8,0,0,1-16,0V140a20,20,0,0,0-40,0v36a8,8,0,0,1-16,0V112a8,8,0,0,1,15.79-1.78A36,36,0,0,1,184,140ZM100,84A12,12,0,1,1,88,72,12,12,0,0,1,100,84Z"></path></svg>
            </a>
            <a href="#" target="_blank">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="#f1f1f1" viewBox="0 0 256 256"><path d="M214.75,211.71l-62.6-98.38,61.77-67.95a8,8,0,0,0-11.84-10.76L143.24,99.34,102.75,35.71A8,8,0,0,0,96,32H48a8,8,0,0,0-6.75,12.3l62.6,98.37-61.77,68a8,8,0,1,0,11.84,10.76l58.84-64.72,40.49,63.63A8,8,0,0,0,160,224h48a8,8,0,0,0,6.75-12.29ZM164.39,208,62.57,48h29L193.43,208Z"></path></svg>
            </a>
            
        
    </div>
""", unsafe_allow_html=True)
