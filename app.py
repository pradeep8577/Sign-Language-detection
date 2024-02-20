import streamlit as st
from pipeline import detectPipeline


st.title('Sign Language Letters detection')
st.write('Detects Sign language Alphabets in an image \nPowered by YOLOv8 Nano model')

st.write('')

detect_pipeline = detectPipeline()

st.info('Sign Language Letters detection model loaded successfully!')

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    
    with st.container():
        col1, col2 = st.columns([3, 3])
        
        col1.header('Input Image')
        col1.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        col1.text('')
        col1.text('')

        if st.button('Detect'):
            detections = detect_pipeline.detect_signs(img_path=uploaded_file)
            detections_img = detect_pipeline.drawDetections2Image(img_path=uploaded_file, detections=detections)

            col2.header('Detections')
            col2.image(detections_img, caption='Predictions by model', use_column_width=True)

