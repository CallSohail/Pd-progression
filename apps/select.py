import streamlit as st
import pandas as pd
# import streamlit.components.v1 as components
import pandas_bokeh


def app():
    # st.title("Identification and prediction of Parkinson disease subtypes and progression using machine learning")
    image_url = "https://img.freepik.com/free-vector/nurses-flat-composition-with-medical-staff-hospital-reception" \
                "-vector-illustration_1284-80772.jpg?size=626&ext=jpg&ga=GA1.1.123692431.1685036899&semt=ais "
    st.image(image_url, caption='Parkinson Disease')
    st.write(
        """
        <style>
        .model-insights {
            font-size: 18px;
            font-family: 'Space Mono', monospace;
            text-align:left;
            
            /* Add more styling properties as needed */
        }
        </style>

        <div class="model-insights"> The goal of the project <span style='color: Cyan; font-weight: bold'>
        Identification & Prediction Of Parkinson Disease Subtypes & Progression Using Machine Learning</span> is to use 
        machine learning techniques to accurately identify and predict the progression of Parkinson's disease 
        subtypes. The project's goal is to develop robust methods for subtyping Parkinson's disease and forecasting 
        its trajectory by analysing clinical, genetic, and imaging data. The ultimate goal is to improve personalised 
        treatment strategies, patient management, and disease understanding. </div>""",
        unsafe_allow_html=True
    )
    # st.write("## Summary")
    # st.markdown("---")
    st.divider()

    def app():
        st.write("## Feature Mapping Table")
        st.write(":blue[_Predict patient PD Subtype we can explain every feature of the model._]")

        feature_mapping = {
            "V04_MCATOT#V04_numerical": ["MOCA Total (Year1)", "The Montreal Cognitive Assessment (MoCA) is a "
                                                               "cognitive screening test used to assess various "
                                                               "cognitive domains, including attention, memory, "
                                                               "language, and visuospatial abilities."],
            "BL_NP1SLPD#BL_numerical": ["NP1SLPD (Baseline)", "Non-Motor Symptom Scale for Parkinson's Disease (NMSS) "
                                                              "is a questionnaire used to assess the non-motor "
                                                              "symptoms experienced by individuals with Parkinson's "
                                                              "disease."],
            "V04_a_trait#V04_numerical": ["a_trait (Year1) ", "Trait Anxiety Inventory (STAI-T) measures trait "
                                                              "anxiety, which refers to a stable tendency to "
                                                              "experience anxiety across various situations."],
            "V04_NP1SLPD#V04_numerical": ["NP1SLPB (Year1) ", "Non-Motor Symptom Scale for Parkinson's Disease (NMSS) "
                                                              "is a questionnaire used to assess the non-motor "
                                                              "symptoms experienced by individuals with Parkinson's "
                                                              "disease."],
            "BL_urinary#BL_numerical": ["urinary (Baseline) ", "The assessment of urinary symptoms is an important "
                                                               "aspect of evaluating Parkinson's disease as urinary "
                                                               "dysfunction is a common non-motor symptom in "
                                                               "Parkinson's disease."],
            "V04_delayed_recall#V04_numerical": ["Delayed recall (Year1) - Delayed recall is a measure of memory "
                                                 "retrieval that assesses an individual's ability to remember "
                                                 "information after a delay."],
            "V04_SDMTOTAL#V04_numerical": ["SDMTOTAL (Year1) ", "Symbol Digit Modalities Test (SDMT) is a "
                                                                "neuropsychological test that assesses cognitive "
                                                                "processing speed and attention."],
            "V04_total#V04_numerical": ["total sdm (Year1) ", "Symbol Digit Modalities Test (SDMT) is a "
                                                              "neuropsychological test that assesses cognitive "
                                                              "processing speed and attention."],
            "V04_NP2DRES#V04_numerical": ["NP2DRES (Year1) ", "Non-Motor Symptom Scale for Parkinson's Disease (NMSS) "
                                                              "is a questionnaire used to assess the non-motor "
                                                              "symptoms experienced by individuals with Parkinson's "
                                                              "disease."],
            "BL_VLTANIM#BL_numerical": ["VLTANIM (Year1) ", "The Visual Learning Test (VLT) assesses an individual's "
                                                            "ability to learn and recall visual information."],
            "V04_DRMAGRAC#V04_numerical": ["DRMAGRAC (Year1) ", "The Delayed Recall Memory Test (DRM) assesses an "
                                                                "individual's ability to recall a list of words after "
                                                                "a delay."],
            "BL_DRMFIGHT#BL_numerical": ["DRMFIGHT (Baseline) ", "The Delayed Recall Memory Test (DRM) assesses an "
                                                                 "individual's ability to recall a list of words after"
                                                                 " a delay."],
            "BL_NP3POSTR#BL_numerical": ["NP3POSTR (Baseline)", "The Movement Disorder Society Unified Parkinson's "
                                                                "Disease Rating Scale (MDS-UPDRS) is a comprehensive "
                                                                "scale used to assess motor and non-motor symptoms in "
                                                                "Parkinson's disease."],
            "V04_HVLTRT2#V04_numerical": ["HVLTRT2 (Year1)", "The Hopkins Verbal Learning Test (HVLT) measures an "
                                                             "individual's ability to learn and recall verbal "
                                                             "information."],
            "V04_NP2HOBB#V04_numerical": ["NP2HOBB (Year1)", "Non-Motor Symptom Scale for Parkinson's Disease (NMSS) "
                                                             "is a questionnaire used to assess the non-motor symptoms "
                                                             "experienced by individuals with Parkinson's disease."],
            "V04_NHY#V04_numerical": ["NHY (Year1)", "The Hoehn and Yahr Scale is a staging scale commonly used to "
                                                     "assess the progression of Parkinson's disease based on motor "
                                                     "symptoms."],
            "V04_ESS4#V04_numerical": ["ESS4 (Year1)", "The Epworth Sleepiness Scale (ESS) is a questionnaire used to "
                                                       "assess daytime sleepiness."],
            "BL_NP2DRES#BL_numerical": ["NP2DRES (Year1) ", "Non-Motor Symptom Scale for Parkinson's Disease (NMSS) is "
                                                            "a questionnaire used to assess the non-motor symptoms "
                                                            "experienced by individuals with Parkinson's disease."],
            "V04_VLTVEG#V04_numerical": ["VLTVEG (Year1)", "The Visual Learning Test (VLT) assesses an individual's "
                                                           "ability to learn and recall visual information."],
            "V04_HVLTRDLY#V04_numerical": ["HVLTRDLY (Year1)", "The Hopkins Verbal Learning Test (HVLT) measures an "
                                                               "individual's ability to learn and recall verbal "
                                                               "information."]
        }
        if st.checkbox(
                "Show Table", key=
                "show_table_checkbox"
        ):
            # Create a list of dictionaries with descriptions for each key-value pair

            data = [{"Feature": key, "Feature Name": values[0], "Feature Description": values[1]}
                    for key, values in feature_mapping.items() if len(values) >= 2]
            df = pd.DataFrame(data)

            # Convert DataFrame to HTML

            df_html = df.to_html(index=
                                 False
                                 )

            # Remove the index column from the HTML string

            df_html = df_html.replace(
                '<th>'
                , '<th style="font-weight:bold; background-color:orange; text-align:left;font-family:monospace; '
                  'font-size:18px"> '

            )
            df_html = df_html.replace(
                '<td>'
                ,
                '<td style="font-family:monospace; font-size:16px">'
            )

            # Display the modified HTML string

            st.write(df_html, unsafe_allow_html=
            True
                     )

    app()
    st.divider()
    # st.write(
    #     "## Our Team"
    # )
    st.markdown(
        """
            <style>
                .team-title {
                    font-size: 35px;
                    font-weight: bold;
                    font-family: 'Space Mono', monospace;
                    text-align: center;
                    # text-decoration: underline;
                }
            </style>
        """,unsafe_allow_html=True)
    st.markdown(
        "<h2 class='team-title'>Our Team</h2>"
        , unsafe_allow_html=True)
    # Define team member information

    team_members = [{"name": "Muhammad Sohail", "image_url": "https://i.postimg.cc/3x8pssvb/Whats-App-Image-2023-05"
                                                             "-30-at-12-00-18-AM.jpg",
                     "description": "Passionate machine learning engineer. Continuous learner in AI, Python, "
                                    "big data. Business-driven mindset. Contributing to AI advancements ",
                     "linkedin_url": "https://www.linkedin.com/in/muhammadsohail951/"},
                    {
                        "name": "Ahmad Nabi Sultan",
                        "image_url": "https://i.postimg.cc/pTJPM68q/Whats-App-Image-2023-05-30-at-12-02-14-AM.jpg",
                        "description": "Seasoned DevOps and Cloud Engineer. Agile. CI/CD. Infrastructure as code. "
                                       "High-quality solutions, Continuously learning, Innovating.",
                        "linkedin_url": "https://www.linkedin.com/in/ahmad-n-sultan"}]

    # Display team members in a horizontal layout

    cols = st.columns(
        len
        (team_members))

    for member, col in zip(team_members, cols):
        with col:
            st.markdown(f"""
                        <style>
                            .circular-image {{
                                width: 250px;
                                height: 250px;
                                border-radius: 50%;
                                overflow: hidden;
                                margin-left: auto;
                                margin-right: auto;
                            }}
                        </style>""",
                        unsafe_allow_html=True)
            st.markdown(f"""
                        <div class="circular-image">
                            <a href="{member['linkedin_url']}" target="_blank" rel="noopener noreferrer">
                            <img src="{member['image_url']}" style="object-fit: cover; width: 100%; height: 
                            100%;"></div>""",
                        unsafe_allow_html=
                        True)
            # st.subheader(member["name"])
            # st.write(member["description"])
            st.markdown(
                f"<div style='text-align: center; font-size: 30px; font-family: monospace; color: orange;'>"f"<h3>"
                f"{member['name']}</h3> "
                f"<p>{member['description']}</p> "f"</div>", unsafe_allow_html=True
            )
            st.divider()
