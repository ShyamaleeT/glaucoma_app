import os
import cv2
import keras
import imutils
import subprocess
import numpy as np
import streamlit as st
import tensorflow as tf
from io import StringIO
from keras import losses
from gradcam import GradCAM
from matplotlib import pyplot
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from keras.models import Model
from keras.layers import ELU, ReLU
from keras.models import load_model
from keras.preprocessing import image
from skimage.transform import resize
from streamlit_option_menu import option_menu
from keras.models import Model, load_model
from keras_preprocessing.image import load_img
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.utils import load_img, img_to_array 
from tensorflow.keras.preprocessing.image import img_to_array
from keras.layers import Input, MaxPooling2D, AveragePooling2D, average
from keras.layers import concatenate, Conv2D, Conv2DTranspose, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, UpSampling2D
from keras.layers import Convolution2D, ZeroPadding2D, Embedding, LSTM, concatenate, Lambda, Conv2DTranspose, Cropping2D
act = ReLU
from custom_model import *
from CDR import *

st.set_option('deprecation.showfileUploaderEncoding', False)


if not os.path.isfile('model.h5'):
    subprocess.run(['curl --output model.h5 "https://media.githubusercontent.com/media/ShyamaleeT/glaucocare/main/sep_5.h5"'], shell=True)
if not os.path.isfile('OD_Segmentation.h5'):
    subprocess.run(['curl --output model1.h5 "https://media.githubusercontent.com/media/ShyamaleeT/glaucocare/main/models/OD_Segmentation.h5"'], shell=True)
if not os.path.isfile('OC_Segmentation.h5'):
    subprocess.run(['curl --output model2.h5 "https://media.githubusercontent.com/media/ShyamaleeT/glaucocare/main/models/OC_Segmentation.h5"'], shell=True)
    
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

def read_input(path):
    x = cv2.imread(path)
    x = cv2.resize(x, (256, 256))
    b, g, r = cv2.split(x)
    x = cv2.merge((r, r, r))
    return x.reshape(256, 256, 3)/255.


def read_gt(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    return x/255.

def load_image(image):
    image = read_image(image)
    return img_load

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

    st.header("The 10 countries with the highest number of persons with vision loss - 2022")
    st.write("As may be expected, these countries also have the largest populations. China and India together account for 49% of the world’s total burden of blindness and vision impairment, while their populations represent 37% of the global population.")
    AgGrid(df)

    @st.cache
    def data_upload1():
        df1= pd.read_csv("Country Map  Estimates of Vision Loss2.csv")
        return df1

    df1=data_upload1()
    #st.dataframe(data = df)

    st.header("The 10 countries with the highest rates of vision loss - 2022")
    st.write("The comparative age-standardised prevalence rate can be helpful in providing a comparison of which country experiences the highest rates of vision impairment, regardless of age structure. India is the only country to appear on both ‘top 10’ lists, as it has the most vision impaired people, as well as the 5th highest overall rate of vision impairment.")
    AgGrid(df1)

    st.markdown(f'<h1 style="color:Red;font-size:25px;"> {"In 2022 in Sri Lanka, there were an estimated 3.9 million people with vision loss. Of these, 89,000 people were blind"}</h1>',unsafe_allow_html=True)
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
    st.write("Sample Data: [Fundus Images](https://drive.google.com/drive/folders/1rKa5xtzw4_8Y53Om4e5LIlH6Jhp3hAT8?usp=sharing)")
    st.write("Check out the [User Manual](https://drive.google.com/file/d/13qws360wHPfUNgnrBCLqQRr7RPR1sTjJ/view?usp=sharing)")
    
    #my_path = os.path.abspath(os.path.dirname(__file__))
    #model_path = os.path.join(my_path, "sep_5.h5")

    model = tf.keras.models.load_model('model.h5', compile=False)
    model1 = 'models/OD_Segmentation.h5'
    #model2 = load_model('models/OC_Segmentation.h5', compile=False)
    
    label_dict={1:'Glaucoma', 0:'Normal'}

    file = st.file_uploader("Please upload an image(jpg/png/jpeg/bmp) file", type=["jpg", "png", "jpeg", "bmp"])

    if file is not None:
        file_details = {"filename": file.name, "filetype": file.type,"filesize": file.size}
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Input image")
            imageI = Image.open(file)
            #st.image(imageI, width=250, channels="BGR",use_column_width=True)
        
            opencv_image = cv2.imdecode(file_bytes, 1)
            st.image(opencv_image, width=100, channels="BGR",use_column_width=True)
            opencv_image_processed = preprocess(opencv_image)
            
        with col_b:
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
        
        col1_a, col1_b = st.columns(2)

        with col1_a:

            contour_img = np.array(Image.open(file))
            img = cv2.resize(contour_img, (256, 256))
            b, g, r = cv2.split(img)
            img_r = cv2.merge((b, b, b))/255.
            #img_r1= cv2.resize(img_r, (224,224))
            #st.image(img)
            
            disc_model = get_unet(do=0.25, activation=act)
            disc_model.load_model(model1, compile=False)

            cup_model = get_unet1(do=0.2, activation=act)
            #cup_model.load_weights('models/OC_Segmentation.h5')

            disc_pred = disc_model.predict(np.array([img_r]))
            disc_pred = np.clip(disc_pred, 0, 1)
            pred_disc = (disc_pred[0, :, :, 0]>0.5).astype(int)
            pred_disc = 255 * pred_disc#.*(pred_disc - np.min(pred_disc))/(np.max(pred_disc)-np.min(pred_disc))
            cv2.imwrite('temp_disc.png', pred_disc)

            disc = cv2.imread('temp_disc.png', cv2.IMREAD_GRAYSCALE)
            st.image(pred_disc, width=225)

            masked = cv2.bitwise_and(img, img, mask = disc)
            #st.image(disc)
            #st.image(masked, width=225)
            #plt.show()
            mb, mg, mr = cv2.split(masked)
            masked = cv2.merge((mg, mg, mg)) #Morphological segmentation for defining optic disc from Green channel and optic cup from Red channel

        with col1_b: #cup segmentation
            cup_pred = cup_model.predict(np.array([masked]))
            pred_cup = (cup_pred[0, :, :, 0]>0.5).astype(int)
            pred_cup = cv2.bilateralFilter(cup_pred[0, :, :, 0],10,40,20)
            pred_cup = (pred_cup > 0.5).astype(int)
            pred_cup = resize(pred_cup, (512, 512))
            pred_cup = 255.*(pred_cup - np.min(pred_cup))/(np.max(pred_cup)-np.min(pred_cup))
            cv2.imwrite('temp_cup.png', pred_cup)
            cup = cv2.imread('temp_cup.png', cv2.IMREAD_GRAYSCALE)
            st.image(pred_cup, clamp = True)

        disc = resize(disc, (512, 512))
        cv2.imwrite('temp_disc.png', disc)
        disc = cv2.imread('temp_disc.png', cv2.IMREAD_GRAYSCALE)
        (thresh, disc) = cv2.threshold(disc, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.imwrite('temp_disc.png', disc)
        (thresh, cup) = cv2.threshold(cup, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        cup_img = Image.open('temp_cup.png')
        disc_img = Image.open('temp_disc.png')  
        os.remove('temp_cup.png')
        os.remove('temp_disc.png')

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
    

        
