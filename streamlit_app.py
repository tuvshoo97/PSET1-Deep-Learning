import streamlit as st
from fastai.vision.all import *
import gdown

st.set_page_config(page_title="Mongolian Four Food Classifier", page_icon=":fork_and_knife:", layout="wide")

st.markdown("# Mongolian Four Food Classifiere")

st.markdown("Mongolians eat these Four traditional foods durin Holidays: Tsuivan, Khuushuur, Buuz, Niislel salad. This app allows you to upload an image of one of these foods and the connected model will classify it for you. Upload an image and try it out!")

st.markdown("This app was created as a demo for the Deep Learning course at LETU Mongolia American University.")

st.markdown("### Upload your image here")

image_file = st.file_uploader("Image Uploader", type=["png","jpg","jpeg"], key="upload_img")

# Model Loading Section
model_path = Path("export.pkl")

if not model_path.exists():
    with st.spinner("Downloading model... this may take a while! \n Don't stop it!"):
        url = 'https://drive.google.com/file/d/1rMgWGS3meqWVLN0PjSHEFbb0jwZT6gCE/view?usp=sharing'
        output = 'export.pkl'
        gdown.download(url, output, quiet=False, use_cookies=False)
    learn_inf = load_learner('export.pkl')
else:
    learn_inf = load_learner('export.pkl')

if image_file is not None:
    with st.spinner("Classifying food...", key="classifying"):
        img = PILImage.create(image_file)
        pred, pred_idx, probs = learn_inf.predict(img)
        
    st.write(f"""## Predicted food: {pred.capitalize()}""")
    st.write(f"""## Probability: {round(max(probs.tolist()), 3) * 100}%""")
    st.image(img, width=300)
    
    st.balloons() # adds background animation
