# AI Rapid Prototype: Image Processing (Streamlit)

This project is a rapid prototype for an AI course assignment. It demonstrates:
- Capturing an image from **webcam** (Streamlit's `st.camera_input`), **upload**, or **URL from the Internet**
- Simple **image processing** (grayscale, Gaussian blur, Canny edge)
- **Customizable parameters** via GUI controls
- Display of the **processed output image**
- A **graph** derived from image properties (intensity histogram)

## Demo (Local)

```bash
# 1) Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the app
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

## Features

- **Input Sources**: Webcam, Upload, or Internet URL
- **Processing**: None, Grayscale, Gaussian Blur (kernel size & sigma), Canny Edge (two thresholds)
- **Graph**: Histogram of pixel intensities of the processed image
- **Live Controls**: Adjust parameters in the sidebar

## Notes

- Webcam capture uses `st.camera_input` and requires browser permission.
- URL loading uses `requests`. Ensure the URL is publicly accessible.
- If you're deploying on Streamlit Cloud, keep `opencv-python-headless` in `requirements.txt`.

## Submit as Git URL

Create a new GitHub repository and push:

```bash
git init
git add .
git commit -m "Initial commit: AI rapid prototype (Streamlit)"
git branch -M main
git remote add origin https://github.com/<your-username>/ai-rapid-proto-streamlit.git
git push -u origin main
```

Replace `<your-username>` with your GitHub username. Submit the repository URL.
