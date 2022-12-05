import streamlit as st
import os
from tensorflow.keras.models import model_from_json
import cv2
import numpy as np
import urllib
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from Scrape_insta import Scrape_insta
import base64
import plotly.express as px
import pandas as pd

img_big_five = Image.open("images/bigFive.jpg")
file_ = open("images/code_gif.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()
st.write('''
    ## Traitement d'images pour le profilage de la personnalité de l'utilisateur
    ### Prédire la personnalité de votre favori personne
    ''')

st.write("---")
with st.container():
    
    left_column , right_column =st.columns(2)
    with right_column :
        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" alt="cat gif" width=100%>',
            unsafe_allow_html=True,
            
        )
    with left_column :
        st.image(img_big_five)
    st.write("""
            1. L'ouverture (également appelée ouverture à l'expérience) met l'accent sur l'imagination et la perspicacité au maximum des cinq traits de personnalité. Les personnes très ouvertes ont tendance à avoir un large éventail d'intérêts. Ils sont curieux du monde et des autres et désireux d'apprendre de nouvelles choses et de vivre de nouvelles expériences.

            2. la conscience est définie par des niveaux élevés de réflexion, un bon contrôle des impulsions et des comportements axés sur les objectifs. Les personnes très consciencieuses ont tendance à être organisées et soucieuses des détails. Ils planifient à l'avance, réfléchissent à la façon dont leur comportement affecte les autres et sont conscients des délais.

            3. L'extraversion est un trait de personnalité caractérisé par l'excitabilité, la sociabilité, la loquacité, l'affirmation de soi et une grande expressivité émotionnelle. Les personnes très extraverties sont extraverties et ont tendance à gagner de l'énergie dans les situations sociales. Être entouré d'autres les aide à se sentir énergisés et excités.

            4. Amabilité est un trait de personnalité comprend des attributs tels que la confiance,  l'altruisme , la gentillesse, l'affection et d'autres  comportements prosociaux . Les personnes qui sont très agréables ont tendance à être plus coopératives tandis que celles qui ont peu ce trait de personnalité ont tendance à être plus compétitives et parfois même manipulatrices.

            5. Le névrosisme est un trait de personnalité caractérisé par la tristesse, les sautes d'humeur et l'instabilité émotionnelle. Les personnes fortement névrosées ont tendance à éprouver des sautes d'humeur , de l'anxiété, de l'irritabilité et de la tristesse. Ceux qui ont peu de ce trait de personnalité ont tendance à être plus stables et plus résilients émotionnellement .
    """)
    st.write("---")
    st.sidebar.header("nom d'utilisateur instagram")
    insta_username =st.sidebar.text_input("Saisir nom d'utilisateur insatgram")
    if not insta_username :
        st.write('''
    #### Veuillez saisir un nom d'utilisateur Instagram pour commancer la prédiction
    ''')
    else :
        st.subheader("le nom d'utilisateur saisie : "+insta_username)
        def test(insta_username):
            resultat=[]
            urls=list()
            
            s=Scrape_insta(insta_username)
            s.info_profile(insta_username)
            urls=s.post_urls(insta_username)

            with open('model.json', 'r') as json_file:
                model = model_from_json(json_file.read())

            model.load_weights('model_weights.h5')
            for i in range(len(urls)):
                
                url_response = urllib.request.urlopen(urls[i])
                img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
                img = cv2.imdecode(img_array, -1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (208, 208), cv2.INTER_CUBIC)
                img = np.array(img)

                img = np.expand_dims(img, axis = 0)
                y = model.predict(img)
                resultat.extend(y)
            resultat=((np.sum(resultat, axis=0))/ len(urls))*100
            data={'Extraversion' : resultat[0],
                  'Agreeableness' : resultat[1],
                  'Conscientiousness' : resultat[2],
                  'Neurotisicm' : resultat[3],
                  'Openness' : resultat[4]
            }
            data=pd.DataFrame(data,index=[0])
            return resultat, data;
                    

        personality_traits = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neurotisicm', 'Openness']
        resultat, data = test(insta_username)
        st.write(data)
        df = pd.DataFrame({'personality': personality_traits, 'values': resultat})

        fig = plt.figure()

        ax = fig.add_subplot(111, projection="polar")

        #theta = np.arange(len(df) + 1) / float(len(df)) * 2 * np.pi

        values = df['values'].values
        values = np.append(values, values[0])
        theta = df['personality'].values
        theta = np.append(theta, theta[0])

        l1, = ax.plot(theta, values, color="gray", marker="o",
                      label="Name of values")

        ax.tick_params(pad=10)
        ax.fill(theta, values, 'gray', alpha=0.3)
        plt.title('Username : '+insta_username)
        st.pyplot(fig)
##        cols = ['r','b','c','g', 'orange']
##        fig1, ax1 = plt.subplots()
##        ax1.pie(resultat,
##        labels=personality_traits,
##        colors = cols,
##        autopct='%1.1f%%',
##        shadow=True, startangle=90)
##        st.subheader("Résultat graphique")
##        st.pyplot(fig1)
            


