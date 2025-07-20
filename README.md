# 🧠 AION Face Detection & Embedding Pipeline

This repository contains the **face detection and embedding pipeline** used in the initial phase of the [AION Gender Classification Model](https://github.com/AnweshaBhadury/AION-face-gender-classification-model). It focuses on **clean image input handling**, **robust face detection**, and **high-quality face embeddings**—a modular system that can be extended for tasks like gender classification, face verification, or clustering.

---

## 🚀 Features

- 🔍 **Face Detection** using RetinaFace  
- 🧬 **Face Embedding** using FaceNet (InceptionResnetV1)  
- 📷 Works with **blurred/low-quality images**  
- 🧼 Includes image preprocessing and cleaning  
- 🧪 Clean, testable structure ready for integration with classification models

---

## 🛠️ Tech Stack

- Python  
- Streamlit – for lightweight UI  
- RetinaFace – for accurate face detection  
- Facenet-PyTorch – for generating embeddings  
- OpenCV, PIL, NumPy – for image processing  

---

## 📂 Project Structure
  
```bash
├── trial1.py                → Main Streamlit app  
├── model weights & assets  (stored in same folder)
├── requirements.txt        
```

---

## ▶️ How to Run

1. **Create Virtual Environment**  
    ```bash
    python -m venv  
    .\venv\Scripts\activate  
    ```

2. **Install Dependencies**  
    ```bash
    pip install -r requirements.txt  
    ```

3. **Run the App**  
    ```bash
    streamlit run trial1.py  
    ```

---

## 🔗 Related Project

This is the **pipeline-only version** of the full classification system available here:  
👉 [AION Gender Classification Model by Anwesha Bhadury](https://github.com/AnweshaBhadury/AION-face-gender-classification-model)

---

## 🙏 Credits

- Face Detection – [RetinaFace](https://github.com/serengil/retinaface)  
- Face Embedding – [Facenet PyTorch](https://github.com/timesler/facenet-pytorch)  
- In Collaboration – with [Anwesha Bhadury](https://github.com/AnweshaBhadury) during initial model development  

---

## 📌 Note

This version does **not** include gender classification.  
It focuses solely on the image processing, detection, and embedding pipeline that forms the backbone of facial AI tasks.
