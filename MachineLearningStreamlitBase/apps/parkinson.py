import pickle
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import plotly.graph_objects as pgo
import pandas as pd

model_path = Path.cwd() / 'saved_models' / 'parkinsons_model.sav'
parkinsons_model = pickle.load(open(model_path, 'rb'))


def app():
    global select_patient
    # st.title("Parkinson's Disease Prediction using ML")
    st.markdown(
        """
        <h1 style="font-family: 'Space Mono', monospace;; font-size: 32px; text-align: center; color: orange;">
            Parkinson's Disease Prediction
        </h1>
        """,
        unsafe_allow_html=True
    )

    import random
    parkinsons_data = pd.read_csv(
        r"E:\sohail semseter data\8th Semester\FINAL YEAR PROJECT\PDProgressionSubtypes\data\parkinsons.csv")
    print(parkinsons_data.columns)
    # Random Patient Button
    column_style = '''

         <style>
             .stTextInput label {
             color: #EDEDED;
             font-size: 16px;
             font-weight: bold;
             font-family: 'Space Mono', monospace;
         }
             .stTextInput input {
                 background-color: #181823;
                 border: 1px solid #C1D0B5;
                 border-radius: 5px;
                 padding: 10px;
                 width: 100%;
             }
         </style>
     '''
    # Render the CSS styles

    st.markdown(column_style, unsafe_allow_html=
    True
                )
    st.button("Random Patient 🏥")
    select_patient_index = random.choice(parkinsons_data.index)
    select_patient = parkinsons_data.loc[select_patient_index]
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('**MDVP:Fo(Hz)**', select_patient['MDVP:Fo(Hz)'])
    with col2:
        fhi = st.text_input('**MDVP:Fhi(Hz)**', select_patient['MDVP:Fhi(Hz)'])
    with col3:
        flo = st.text_input('**MDVP:Flo(Hz)**', select_patient['MDVP:Flo(Hz)'])
    with col4:
        Jitter_percent = st.text_input('**MDVP:Jitter(%)**', select_patient['MDVP:Jitter(%)'])
    with col5:
        Jitter_Abs = st.text_input('**MDVP:Jitter(Abs)**', select_patient['MDVP:Jitter(Abs)'])
    with col1:
        RAP = st.text_input('**MDVP:RAP**', select_patient['MDVP:RAP'])
    with col2:
        PPQ = st.text_input('**MDVP:PPQ**', select_patient['MDVP:PPQ'])
    with col3:
        DDP = st.text_input('**Jitter:DDP**', select_patient['Jitter:DDP'])
    with col4:
        Shimmer = st.text_input('**MDVP:Shimmer**', select_patient['MDVP:Shimmer'])
    with col5:
        Shimmer_dB = st.text_input('**MDVP:(dB)**', select_patient['MDVP:Shimmer(dB)'])
    with col1:
        APQ3 = st.text_input('**Shimmer:APQ3**', select_patient['Shimmer:APQ3'])
    with col2:
        APQ5 = st.text_input('**Shimmer:APQ5**', select_patient['Shimmer:APQ5'])
    with col3:
        APQ = st.text_input('**MDVP:APQ**', select_patient['MDVP:APQ'])
    with col4:
        DDA = st.text_input('**Shimmer:DDA**', select_patient['Shimmer:DDA'])
    with col5:
        NHR = st.text_input('**NHR**', select_patient['NHR'])
    with col1:
        HNR = st.text_input('**HNR**', select_patient['HNR'])
    with col2:
        RPDE = st.text_input('**RPDE**', select_patient['RPDE'])
    with col3:
        DFA = st.text_input('**DFA**', select_patient['DFA'])
    with col4:
        spread1 = st.text_input('**spread1**', select_patient['spread1'])
    with col5:
        spread2 = st.text_input('**spread2**', select_patient['spread2'])
    with col1:
        D2 = st.text_input('**D2**', select_patient['D2'])
    with col2:
        PPE = st.text_input('**PPE**', select_patient['PPE'])
    input_data = []
    parkinsons_diagnosis = ''
    image_data = ''
    if st.button("Parkinson's Test Result"):

        input_data = [float(fo), float(fhi), float(flo), float(Jitter_percent), float(Jitter_Abs), float(RAP),
                      float(PPQ),
                      float(DDP), float(Shimmer), float(Shimmer_dB), float(APQ3), float(APQ5), float(APQ), float(DDA),
                      float(NHR), float(HNR), float
                      (RPDE), float(DFA), float(spread1), float(spread2), float(D2), float(PPE)]

        parkinsons_prediction = parkinsons_model.predict([input_data])
        input_data.append(parkinsons_prediction)
        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "#### :red[The person has Parkinson's disease]😞"
            image_data = ('https://img.freepik.com/free-vector/hospital-ward-with-patient-bed-doctor_107791-16654.jpg'
                          '?size '
                     '=626&ext=jpg&ga=GA1.1.123692431.1685036899&semt=ais')



        else:
            parkinsons_diagnosis = "#### :green[The person does not have Parkinson's disease]😃"
            image_data = ('https://img.freepik.com/premium-vector/old-man-hospital-room_82574-2898.jpg?size=626&ext'
                          '=jpg&ga '
                     '=GA1.1.123692431.1685036899&semt=ais')


        st.markdown("---")
        st.title("Prediction Result")
        st.write(parkinsons_diagnosis)
        st.image(image_data)


        fig = pgo.Figure()
        x = np.arange(len(input_data))
        fig.add_trace(pgo.Scatter(x=x, y=input_data, mode='lines+markers', marker=dict(size=10, color='rgb(93, 164, '
                                                                                                      '214)'),
                                  line=dict(color='rgb(44, 160, 101)', width=3)))
        fig.update_layout(title='Input Data', xaxis_title='Input Feature', yaxis_title='Value',
                          title_font=dict(size=24),
                          font=dict(size=18))
        st.plotly_chart(fig)


# Run the app
if __name__ == '__main__':
    app()
