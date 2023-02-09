import streamlit as st
from fastbook import *
from fastai.vision.all import *
import gdown

st.set_page_config(page_title="Mongolian Four Food Classifier", page_icon=":fork_and_knife:", layout="wide")

st.markdown("""# Mongolian Four Food Classifier

Get ready to impress your Mongolian friends with your knowledge of traditional holiday foods! With this app, you can upload an image of Tsuivan, Khuushuur, Buuz, or Niislel salad and find out which delicious dish it is. Who needs a taste tester when you've got this app? 

This app was created as a fun demo for the Deep Learning course at LETU Mongolia American University, but we won't judge if you use it to win food trivia night. 🍴""")

st.markdown("""### Upload your image here""")

image_file = st.file_uploader("Image Uploader", type=["png","jpg","jpeg"])

# Model Loading Section
model_path = Path("export.pkl")

if not model_path.exists():
    with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
        url = 'https://drive.google.com/uc?id=1cs7nWO-XIQBWqLnTjzuG_lOkEnXwyrpd'
        output = 'export.pkl'
        gdown.download(url, output, quiet=False)
    learn_inf = load_learner('export.pkl')
else:
    learn_inf = load_learner('export.pkl')

if image_file is not None:
    with st.spinner("Classifying food..."):
        img = PILImage.create(image_file)
        pred, pred_idx, probs = learn_inf.predict(img)
        
    st.write(f"""## Predicted food: {pred.capitalize()}""")
    st.write(f"""## Probability: {round(max(probs.tolist()), 3) * 100}%""")
    st.image(img, width=300)
    
    st.balloons() # adds background animation
