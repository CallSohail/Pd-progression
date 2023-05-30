import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import hashlib
import matplotlib.pyplot as plt
import sklearn

dict_map_result = {
    'smoker': "Smoking status",
    'cognitiveStatus2': "Cognitive status 2",
    'elEscorialAtDx': "El Escorial category at diagnosis",
    'anatomicalLevel_at_onset': "Anatomical level at onset",
    'site_of_onset': "Site of symptom onset",
    'onset_side': "Onset side",
    'ALSFRS1': "ALSFRS-R part 1 score",
    'FVCPercentAtDx': "FVC% at diagnosis",
    'weightAtDx_kg': "Weight at diagnosis (kg)",
    'rateOfDeclineBMI_per_month': "Rate of BMI decline (per month)",
    'age_at_onset': "Age at symptom onset",
    'firstALSFRS_daysIntoIllness': "Time of first ALSFRS-R measurement (days from symptom onset)"
}

dict_map_result = {
    "V04_MCATOT#V04_numerical": "MOCA Total (Year1)",
    "BL_NP1SLPD#BL_numerical": "NP1SLPD (Baseline)",
    "V04_a_trait#V04_numerical": "a_trait (Year1)",
    "V04_NP1SLPD#V04_numerical": "NP1SLPB (Year1)",
    "BL_urinary#BL_numerical": "urinary (Baseline)",
    "V04_delayed_recall#V04_numerical": "Delayed recall (Year1)",
    "V04_SDMTOTAL#V04_numerical": "SDMTOTAL (Year1)",
    "V04_total#V04_numerical": "total sdm (Year1)",
    "V04_NP2DRES#V04_numerical": "NP2DRES (Year1)",
    "BL_VLTANIM#BL_numerical": "VLTANIM (Year1)",
    "V04_DRMAGRAC#V04_numerical": "DRMAGRAC (Year1)",
    "BL_DRMFIGHT#BL_numerical": "DRMFIGHT (Baseline)",
    "BL_NP3POSTR#BL_numerical": "NP3POSTR (Baseline)",
    "V04_HVLTRT2#V04_numerical": "HVLTRT2 (Year1)",
    "V04_NP2HOBB#V04_numerical": "NP2HOBB (Year1)",
    "V04_NHY#V04_numerical": "NHY (Year1)",
    "V04_ESS4#V04_numerical": "ESS4 (Year1)",
    "BL_NP2DRES#BL_numerical": "NP2DRES (Year1)",
    "V04_VLTVEG#V04_numerical": "VLTVEG (Year1)",
    "V04_HVLTRDLY#V04_numerical": "HVLTRDLY (Year1)",

}


def app():
    @st.cache_resource(ttl=24 * 3600)
    def load_model1():
        with open('saved_models/trainXGB_class_map.pkl', 'rb') as f:
            class_names = list(pickle.load(f))
        return class_names

    class_names = ["PD_h"]  # load_model1()
    # Updated Code By Sohail
    st.markdown(
        """
        <div style="
            text-align: center;
            border: 1px solid red;
            border-radius: 5px;
            padding: 2%;
            background-color: #0b0c11;
            font-family: 'Space Mono', monospace;
            color: orange;
            font-size: 24px;
            font-weight: bold;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
            margin-bottom: 20px;
        ">
            Model Analysis! Not For the User
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("## :blue[SHAP Model Interpretation]")

    @st.cache_resource(ttl=24 * 3600)
    def load_model2():
        with open('saved_models/trainXGB_gpu.aucs', 'rb') as f:
            result_aucs = pickle.load(f)
        return result_aucs

    result_aucs = load_model2()

    if len(result_aucs[class_names[0]]) == 3:
        mean_train_aucs = round(np.mean([result_aucs[i][0] for i in class_names]), 2)
        mean_test_aucs = round(np.mean([result_aucs[i][1] for i in class_names]), 2)
        df_res = pd.DataFrame({'class name': class_names + ['MEAN'],
                               'Discovery Cohort AUC': ["{:.2f}".format(result_aucs[i][0]) for i in class_names] + [
                                   mean_train_aucs],
                               'Replication Cohort AUC': ["{:.2f}".format(result_aucs[i][1]) for i in class_names] + [
                                   mean_test_aucs]})
        replication_avail = True
    else:
        df_res = pd.DataFrame(
            {'class name': class_names, 'Train AUC': ["{:.2f}".format(result_aucs[i][0]) for i in class_names],
             'Test AUC': ["{:.2f}".format(result_aucs[i][1]) for i in class_names]})
        replication_avail = False

    @st.cache_resource(ttl=24 * 3600)
    def get_shapley_value_data(train, replication=True, dict_map_result={}):
        dataset_type = ''
        shap_values = np.concatenate([train[0]['shap_values_train'], train[0]['shap_values_test']], axis=0)
        # shap_values = train[0]['shap_values_train']
        X = pd.concat([train[1]['X_train'], train[1]['X_valid']], axis=0)
        exval = train[2]['explainer_train']
        auc_train = train[3]['AUC_train']
        auc_test = train[3]['AUC_test']
        ids = list(train[3]['ID_train'.format(dataset_type)]) + list(train[3]['ID_test'.format(dataset_type)])
        labels_pred = list(train[3]['y_pred_train'.format(dataset_type)]) + list(
            train[3]['y_pred_test'.format(dataset_type)])
        labels_actual = list(train[3]['y_train'.format(dataset_type)]) + list(train[3]['y_test'.format(dataset_type)])
        shap_values_updated = shap.Explanation(values=np.array(shap_values), base_values=np.array([exval] * len(X)),
                                               data=np.array(X.values), feature_names=X.columns)
        train_samples = len(train[1]['X_train'])
        test_samples = len(train[1]['X_valid'])
        X.columns = ['{}'.format(dict_map_result[col]) if dict_map_result.get(col, None) is not None else col for col in
                     list(X.columns)]
        shap_values_updated = shap_values_updated
        patient_index = [hashlib.md5(str(s).encode()).hexdigest() for e, s in enumerate(ids)]
        return (
            X, shap_values, exval, patient_index, auc_train, auc_test, labels_actual, labels_pred, shap_values_updated,
            train_samples, test_samples)

    st.write("## :orange[_Introduction_]")
    st.write(
        """
        <style>
        .shap-explanation {
            font-size: 16px;
            font-family: 'Space Mono', monospace;;
            # font-weight: bold;
            color: white;
            /* Add more styling properties as needed */
        }
        </style>

        <div class="shap-explanation">
        The Shapley Additive Explanations (SHAP) approach is used to assess feature importance in ensemble 
        learning. It assigns an importance value to each feature, indicating its contribution to the model's success. 
        SHAP provides accurate explanations for individual observations, aligning with human knowledge and 
        expectations. In multiclass classification, one-vs-rest technique is used, training separate models for each 
        class. SHAP helps interpret the model effectively, identifying key variables and validating decisions. It 
        builds trust by ensuring important features align with domain understanding. By employing SHAP, 
        we gain insights, enhance confidence, and make better decisions based on the model's predictions. 
        </div>
        """,
        unsafe_allow_html=True
    )

    feature_set_my = class_names[0]

    @st.cache_resource(ttl=24 * 3600)
    def load_model3():
        with open('saved_models/trainXGB_gpu_{}.data'.format(feature_set_my), 'rb') as f:
            train = pickle.load(f)
        return train

    train = load_model3()
    # data_load_state = st.text('Loading data...')
    cloned_output = get_shapley_value_data(train, replication=replication_avail, dict_map_result=dict_map_result)

    # data_load_state.text("Done Data Loading! (using st.cache)")
    X, shap_values, exval, patient_index, auc_train, auc_test, labels_actual, labels_pred, shap_values_up, len_train, len_test = cloned_output

    @st.cache_resource(ttl=24 * 3600)
    def load_model3():
        shap_values_list = []
        for classname in class_names:
            with open('saved_models/trainXGB_gpu_{}.data'.format(classname), 'rb') as f:
                train_temp = pickle.load(f)
                shap_values_list.append(
                    np.concatenate([train_temp[0]['shap_values_train'], train_temp[0]['shap_values_test']], axis=0))
        shap_values = np.mean(shap_values_list, axis=0)
        return shap_values

    shap_values = load_model3()

    st.write('## :orange[_Summary Plot_]')
    st.write(
        """
        <style>
        .top-features {
            font-size: 16px;
            # font-weight: bold;
            font-family: 'Space Mono', monospace;
            color: white;
            /* Add more styling properties as needed */
        }
        </style>

        <div class="top-features">
        Shows top-20 features that have the most significant impact on the classification model.
        </div>
        """,
        unsafe_allow_html=True
    )
    if True:  # st.checkbox("Show Summary Plot"):
        shap_type = 'trainXGB'
        col1, col2, col2111 = st.columns(3)
        with col1:
            st.write('---')

            @st.cache_resource(ttl=24 * 3600)
            def generate_plot1():
                fig, ax = plt.subplots(figsize=(20, 25))
                shap.plots.beeswarm(
                    shap.Explanation(values=np.array(shap_values), base_values=np.array([exval] * len(X)),
                                     data=np.array(X.values), feature_names=X.columns), show=False, max_display=20,
                    order=shap.Explanation(values=np.array(shap_values), base_values=np.array([exval] * len(X)),
                                           data=np.array(X.values), feature_names=X.columns).mean(0).abs,
                    plot_size=0.47)  # 0.47# , return_objects=True
                return fig

            st.pyplot(generate_plot1())
            st.write('---')
        with col2:
            st.write('---')

            @st.cache_resource(ttl=24 * 3600)
            def generate_plot2():
                fig, ax = plt.subplots(figsize=(20, 25))
                shap.plots.bar(shap.Explanation(values=np.array(shap_values), base_values=np.array([exval] * len(X)),
                                                data=np.array(X.values), feature_names=X.columns).mean(0), show=False,
                               max_display=20, order=shap.Explanation(values=np.array(shap_values),
                                                                      base_values=np.array([exval] * len(X)),
                                                                      data=np.array(X.values),
                                                                      feature_names=X.columns).mean(0).abs)

                return fig

            st.pyplot(generate_plot2())
            st.write('---')
        with col2111:
            st.write('---')

            @st.cache_resource(ttl=24 * 3600)
            def generate_plot3():
                fig, ax = plt.subplots(figsize=(20, 25))
                # temp = shap.Explanation(values=np.array(shap_values), base_values=np.array([exval]*len(X)),
                # data=np.array(X.values), feature_names=X.columns)
                shap.plots.bar(shap.Explanation(values=np.array(shap_values), base_values=np.array([exval] * len(X)),
                                                data=np.array(X.values), feature_names=X.columns).abs.mean(0),
                               show=False, max_display=20, order=shap.Explanation(values=np.array(shap_values),
                                                                                  base_values=np.array(
                                                                                      [exval] * len(X)),
                                                                                  data=np.array(X.values),
                                                                                  feature_names=X.columns).mean(0).abs)
                return fig

            st.pyplot(generate_plot3())
            st.write('---')

    import random
    select_random_samples = np.random.choice(shap_values.shape[0], 800)

    new_X = X.iloc[select_random_samples]
    new_shap_values = shap_values[select_random_samples, :]
    new_labels_pred = np.array(labels_pred, dtype=np.float64)[select_random_samples]

    st.write('## :orange[_Statistics for Individual Classes_]')
    feature_set_my = "PD_h"
    if not feature_set_my == "Select":
        @st.cache_resource(ttl=24 * 3600)
        def load_model9():
            with open('saved_models/trainXGB_gpu_{}.data'.format(feature_set_my), 'rb') as f:
                train = pickle.load(f)
            return train

        train = load_model9()
        cloned_output = get_shapley_value_data(train, replication=replication_avail, dict_map_result=dict_map_result)
        X, shap_values, exval, patient_index, auc_train, auc_test, labels_actual, labels_pred, shap_values_up, len_train, len_test = cloned_output
        col0, col00 = st.columns(2)
        with col0:
            st.write("### :blue[Data Statistics]")
            st.info(':green[Total Features: **{}**]'.format(X.shape[1]))
            st.info(
                ':green[Total Samples: **{}** (Discovey: **{}**, Replication: **{}**)]'.format(X.shape[0], len_train,
                                                                                               len_test))

        with col00:
            st.write("### :blue[ML Model Performance]")
            st.info(':green[AUC Discovery Cohort: **{}**]'.format(round(auc_train, 2)))
            st.info(':green[AUC Replication Cohort: **{}**]'.format(round(auc_test, 2)))

        col01, col02 = st.columns(2)
        with col01:
            st.write("### Discovery Cohort Confusion Matrix")
            Z = sklearn.metrics.confusion_matrix(labels_actual[:len_train], np.array(labels_pred[:len_train]) > 0.5)
            Z_df = pd.DataFrame(Z, columns=['Predicted 0', 'Predicted 1'], index=['Actual 0', 'Actual 1'])
            st.table(Z_df.astype(str))

        with col02:
            st.write("### Replication Cohort Confusion Matrix")
            Z = sklearn.metrics.confusion_matrix(labels_actual[len_train:], np.array(labels_pred[len_train:]) > 0.5)
            Z_df = pd.DataFrame(Z, columns=['Predicted 0', 'Predicted 1'], index=['Actual 0', 'Actual 1'])
            st.table(Z_df.astype(str))

        labels_actual_new = np.array(labels_actual, dtype=np.float64)
        y_pred = (shap_values.sum(1) + exval) > 0
        misclassified = y_pred != labels_actual_new
        st.write('---')

        st.write('### :orange[_Pathways for Misclassified Samples_]')
        if misclassified[len_train:].sum() == 0:
            st.info('No Misclassified Examples!!!')
        elif True:  # st.checkbox("Show Misclassifies Pathways"):
            col6, col7 = st.columns(2)
            with col6:
                st.info('Misclassifications (test): {}/{}]'.format(misclassified[len_train:].sum(), len_test))
                fig, ax = plt.subplots()
                r = shap.decision_plot(exval, shap_values[misclassified], list(X.columns), link='logit',
                                       return_objects=True, new_base_value=0)
                st.pyplot(fig)
            with col7:
                # st.info('Single Example')
                sel_patients = [patient_index[e] for e, i in enumerate(misclassified) if i == 1]
                select_pats = st.selectbox('**Select random misclassified patient**', options=list(sel_patients))
                id_sel_pats = sel_patients.index(select_pats)
                fig, ax = plt.subplots()
                shap.decision_plot(exval, shap_values[misclassified][id_sel_pats],
                                   X.iloc[misclassified, :].iloc[id_sel_pats], link='logit',
                                   feature_order=r.feature_idx, highlight=0, new_base_value=0)
                st.pyplot()
        st.write('## :orange[_Decision Plots_]')
        st.write(
            """
            <style>
            .model-insights {
                font-size: 16px;
                font-family: 'Space Mono', monospace;
                color: white;
                /* Add more styling properties as needed */
            }
            </style>

            <div class="model-insights">
            We selected 800 subsamples to understand the pathways of predictive modeling. SHAP decision plots show how complex models arrive at their predictions (i.e., how models make decisions). 
            Each observation’s prediction is represented by a colored line.
            At the top of the plot, each line strikes the x-axis at its corresponding observation’s predicted value. 
            This value determines the color of the line on a spectrum. 
            Moving from the bottom of the plot to the top, SHAP values for each feature are added to the model’s base value. 
            This shows how each feature contributes to the overall prediction.
            </div>
            """,
            unsafe_allow_html=True
        )

        cols = st.columns(2)
        st.write('### :orange[_Prediction pathways_]')
        if st.checkbox(":green[**View patterns**]"):  # st.checkbox("Show Prediction Pathways (Feature Clustered)"):
            # col3, col4, col5 = st.columns(3)
            # st.write('Typical Prediction Path: Uncertainity (0.2-0.8)')
            r = shap.decision_plot(exval, np.array(new_shap_values), list(new_X.columns), feature_order='hclust',
                                   return_objects=True, show=False)
            T = new_X.iloc[(new_labels_pred >= 0) & (new_labels_pred <= 1)]
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sh = np.array(new_shap_values)[(new_labels_pred >= 0) & (new_labels_pred <= 1), :]
            fig, ax = plt.subplots()
            shap.decision_plot(exval, sh, T, show=False, feature_order=r.feature_idx, link='logit', return_objects=True,
                               new_base_value=0)
            cols[0].pyplot(fig)

            r = shap.decision_plot(exval, np.array(new_shap_values), list(new_X.columns), return_objects=True,
                                   show=False)
            T = new_X.iloc[(new_labels_pred >= 0) & (new_labels_pred <= 1)]
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sh = np.array(new_shap_values)[(new_labels_pred >= 0) & (new_labels_pred <= 1), :]
            fig, ax = plt.subplots()
            shap.decision_plot(exval, sh, T, show=False, feature_order=r.feature_idx, link='logit', return_objects=True,
                               new_base_value=0)
            cols[1].pyplot(fig)
