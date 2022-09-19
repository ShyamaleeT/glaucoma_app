import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from  PIL import Image
import numpy as np
import cv2
import pandas as pd
from st_aggrid import AgGrid
import plotly.express as px
import io 
import altair as alt
from PIL import Image, ImageOps
import numpy as np
import os
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.applications import ResNet50
from gradcam import GradCAM
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
from matplotlib import pyplot
import imutils
import subprocess


st.set_option('deprecation.showfileUploaderEncoding', False)

if not os.path.isfile('model.h5'):
    subprocess.run(['curl --output model.h5 "https://media.githubusercontent.com/media/ShyamaleeT/glaucocare/main/sep_5.h5"'], shell=True)
    
def preprocess(img, req_size = (224,224)):
    image = Image.fromarray(img.astype('uint8'))
    image = image.resize(req_size)
    face_array = img_to_array(image)
    face_array = np.expand_dims(face_array, 0)
    return face_array

def import_and_predict(image_data, model):
    image = ImageOps.fit(image_data, (224,224),Image.LANCZOS)
    #image = image.convert('RGB')
    image = np.asarray(image)
    #st.image(image, channels='RGB')
    image = (image.astype(np.float32) / 255.0)
    img_reshape = image[np.newaxis,...]
    prediction = model.predict(img_reshape)
    return prediction


with st.sidebar:
    choose = option_menu("Content", ["Glaucoma", "Glaucoma Statistics","Glaucoma Analysis Tool"],
                         icons=['house', 'kanban', 'person lines fill'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "blue", "font-size": "25px", "font-family": "Cooper Black"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#7692c2"},
    }
    )

logo = Image.open('logo.png')
profile = Image.open('a.jpg')
if choose == "Glaucoma":
    col1, col2 = st.columns( [0.8, 0.2])
    
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #004c94;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font"> Glaucoma Diesease </p>', unsafe_allow_html=True)

        #st.subheader("Glaucoma Diagnosis")

    with col2:               # To display brand log
        st.image(logo, width=130 )
    
    st.write('Glaucoma is a common cause of permanent blindness. It has globally affected 76 million individuals aged 40 to 80 by 2020, and the number is expected to rise to around 112 million due to the ageing population by the year 2040. However, many people do not aware of this situation as there are no early symptoms. The most prevalent chronic glaucoma condition arises when the trabecular meshwork of the eye becomes less effective in draining fluid. As this happens, eye pressure rises, leading to damage optic nerve. Thus, glaucoma care is essential for sustaining vision and quality of life, as it is a long-term neurodegenerative condition that can only control.')

    st.image(profile, width=700 )

    st.subheader("Causes for glaucoma")
    st.write('Ocular hypertension (increased pressure within the eye) is the most important risk factor for glaucoma, but only about 50% of people with primary open-angle glaucoma actually have elevated ocular pressure. Ocular hypertension—an intraocular pressure above the traditional threshold of 21 mmHg (2.8 kPa) or even above 24 mmHg (3.2 kPa)—is not necessarily a pathological condition, but it increases the risk of developing glaucoma. One study found a conversion rate of 18% within five years, meaning fewer than one in five people with elevated intraocular pressure will develop glaucomatous visual field loss over that period of time. It is a matter of debate whether every person with an elevated intraocular pressure should receive glaucoma therapy; currently, most ophthalmologists favor treatment of those with additional risk factors. Open-angle glaucoma accounts for 90% of glaucoma cases in the United States. Closed-angle glaucoma accounts for fewer than 10% of glaucoma cases in the United States, but as many as half of glaucoma cases in other nations (particularly East Asian countries).')

    st.subheader("Signs and symptoms")
    st.write("As open-angle glaucoma is usually painless with no symptoms early in the disease process, screening through regular eye exams is important. The only signs are gradually progressive visual field loss and optic nerve changes (increased cup-to-disc ratio on fundoscopic examination.")
    #st.write("About 10% of people with closed angles present with acute angle closure characterized by sudden ocular pain, seeing halos around lights, red eye, very high intraocular pressure (>30 mmHg (4.0 kPa)), nausea and vomiting, suddenly decreased vision, and a fixed, mid-dilated pupil. It is also associated with an oval pupil in some cases. Acute angle closure is an emergency.")

elif choose == "Glaucoma Statistics":
    col1, col2 = st.columns( [0.8, 0.2])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #004c94;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font"> Glaucoma Statistics </p>', unsafe_allow_html=True)

           
    with col2:               # To display brand logo
        st.image(logo,  width=150)

    @st.cache
    def data_upload():
        df= pd.read_csv("Country Map  Estimates of Vision Loss1.csv")
        return df

    df=data_upload()
    #st.dataframe(data = df)

    st.header("The 10 countries with the highest number of persons with vision loss - 2020")
    st.write("As may be expected, these countries also have the largest populations. China and India together account for 49% of the world’s total burden of blindness and vision impairment, while their populations represent 37% of the global population.")
    AgGrid(df)


    @st.cache
    def data_upload1():
        df1= pd.read_csv("Country Map  Estimates of Vision Loss2.csv")
        return df1

    df1=data_upload1()
    #st.dataframe(data = df)

    st.header("The 10 countries with the highest rates of vision loss - 2020")
    st.write("The comparative age-standardised prevalence rate can be helpful in providing a comparison of which country experiences the highest rates of vision impairment, regardless of age structure. India is the only country to appear on both ‘top 10’ lists, as it has the most vision impaired people, as well as the 5th highest overall rate of vision impairment.")
    AgGrid(df1)

    st.markdown(f'<h1 style="color:Red;font-size:25px;"> {"In 2020 in Sri Lanka, there were an estimated 3.9 million people with vision loss. Of these, 89,000 people were blind"}</h1>',unsafe_allow_html=True)
    energy_source = pd.DataFrame({
    "Types": ["Blindness","Mild","Mod-severe","Near","Blindness","Mild","Mod-severe","Near","Blindness","Mild","Mod-severe","Near","Blindness","Mild","Mod-severe","Near"],
    "Age prevalence %":  [0.60893,0.5401989,0.4371604,0.3701105,5.2044229,5.0064095,4.819523,4.6035189,5.4498435,5.6630457,5.5772188,5.1737878,5.7292227,5.5587928,5.3857774,5.2338178],
    "Year": ["1990","1990","1990","1990","2000","2000","2000","2000","2010","2010","2010","2010","2020","2020","2020","2020"]
    })
 
    bar_chart = alt.Chart(energy_source).mark_bar().encode(
        x="Year:O",
        y="Age prevalence %",
        color="Types:N"
    )
    st.altair_chart(bar_chart, use_container_width=True)


elif choose == "Glaucoma Analysis Tool":
    col1, col2 = st.columns([0.8, 0.2])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #004c94;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font"> Fundus Image Analysis For Glaucoma </p>', unsafe_allow_html=True)
           
    with col2:               # To display brand logo
        st.image(logo,  width=150)
    
    st.write("To determine whether glaucomatous symptoms are present in an eye fundus image, please upload the image through the pane that can be found below. Depending on your network connection, it will take about 1~3 minutes to present the result on the screen.")
    
    st.write("This is a simple image classification web app to predict glaucoma through fundus image of eye")
    
    #my_path = os.path.abspath(os.path.dirname(__file__))
    #model_path = os.path.join(my_path, "sep_5.h5")

    model = tf.keras.models.load_model('model.h5', compile=False)
    
    
    label_dict={1:'Glaucoma', 0:'Normal'}

    file = st.file_uploader("Please upload an image(jpg/png/jpeg/bmp) file", type=["jpg", "png", "jpeg", "bmp"])

    if file is not None:
        file_details = {"filename": file.name, "filetype": file.type,"filesize": file.size}
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.subheader("Input image")
            imageI = Image.open(file)
            #st.image(imageI, width=250, channels="BGR",use_column_width=True)
        
            opencv_image = cv2.imdecode(file_bytes, 1)
            st.image(opencv_image, width=100, channels="BGR",use_column_width=True)
            opencv_image_processed = preprocess(opencv_image)

        with col_b:
            # imageI = Image.open(file)
            # st.image(imageI, width=250)

            st.subheader(" Heatmap")
            last_conv_layer= "conv5_block3_out" 

            preds = model.predict(opencv_image_processed)
            i = np.argmax(preds[0])
            cam = GradCAM(model, i, last_conv_layer) 
            heatmap = cam.compute_heatmap(opencv_image_processed)
            heatmap = cv2.resize(heatmap, (opencv_image.shape[1], opencv_image.shape[0]))
            (heatmap, output) = cam.overlay_heatmap(heatmap, opencv_image, alpha=0.5)
            heatmap = imutils.resize(heatmap, width=100)
            st.image(heatmap,channels="BGR",use_column_width=True)


        with col_c:
            st.subheader("Visualized image")
            last_conv_layer= "conv5_block3_out" 

            preds = model.predict(opencv_image_processed)
            i = np.argmax(preds[0])
            cam = GradCAM(model, i, last_conv_layer) 
            heatmap = cam.compute_heatmap(opencv_image_processed)
            heatmap = cv2.resize(heatmap, (opencv_image.shape[1], opencv_image.shape[0]))
            (heatmap, output) = cam.overlay_heatmap(heatmap, opencv_image, alpha=0.5)
            output = imutils.resize(output, width=100)
            st.image(output,channels="BGR",use_column_width=True)

        prediction = import_and_predict(imageI, model)
        pred = ((prediction[0][0]))
        print (pred)
        result=np.argmax(prediction,axis=1)[0]
        print (result)
        accuracy=float(np.max(prediction,axis=1)[0])
        print(accuracy)
        label=label_dict[result]
        print(label)
        # print(pred,result,accuracy)
        # response = {'prediction': {'result': label,'accuracy': accuracy}}
        # print(response)

        Normal_prob = "{:.2%}".format(1-pred)
        Glaucoma_prob = "{:.2%}".format(pred)


        if(pred> 0.5):
            st.markdown(f'<h1 style="color:Red;font-size:35px;">{""" Glaucoma Eye"""}</h1>', unsafe_allow_html=True)
            st.text("The area in the image that is highlighted is thought to be glaucomatous.")
            
        else:
            st.markdown(f'<h1 style="color:Blue;font-size:35px;">{"""Healthy Eye"""}</h1>', unsafe_allow_html=True)
                #st.write("""### You are affected by Glaucoma. Please consult an ophthalmologist as soon as possible.""" )

        st.subheader('Prediction Probability')

        col1, col2 = st.columns(2)
        col1.metric("Glaucoma", Glaucoma_prob)
        col2.metric("Normal", Normal_prob)

        st.caption("**Note:This is a prototype tool for glaucoma diagnosis, using experimental deep learning techniques. It is recommended to consult a medical doctor for a proper diagnosis.")
    

        
