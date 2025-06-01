import gradio as gr
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import (mean_squared_error, r2_score, 
                            accuracy_score, precision_score, 
                            recall_score, f1_score, 
                            classification_report)

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import resample
from imblearn.combine import SMOTETomek
from scipy import stats
from collections import Counter
from functools import partial



def run_model(csv_file,
            problem_type,
            model_name,
            input_column,
            drop_column,
            corr_threshold,
            imputation,
            scaling_method,
            test_size,
            sampling_method,
            skew_method_right,
            skew_method_left,
            outlier_method,
            outlier_action,
):
    if csv_file is None:
        return "Please upload a CSV File",None
    
    df=pd.read_csv(csv_file.name)
    df=df.sample(frac=0.05)
    print(df.columns)


    df.drop(columns=drop_column,errors="ignore",inplace=True)

    drop_nan=df.isnull().sum()
    drop_nan=drop_nan[drop_nan.values>len(df)*.30]
    
    df.drop(labels=drop_nan.index,inplace=True,axis=1)

    if corr_threshold>0:
        df_numeric = df.select_dtypes(include=['number']).drop(columns=[input_column], errors='ignore')
        if not df_numeric.empty:
            corr_matrix = df_numeric.corr().abs()
            
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
            
            df.drop(to_drop, axis=1, inplace=True)

    
        
    if imputation != "None":
        for col in df.columns:
            if df[col].dtype in [float,int]:
                if imputation=="Mean":
                    df[col].fillna(df[col].mean(),inplace=True)
                elif imputation=="Median":
                    df[col].fillna(df[col].median(),inplace=True)
            else:
                df[col].fillna(df[col].mode()[0],inplace=True)
    
    if outlier_method !="None" and outlier_action != "None":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_columns=[]

        for col in numeric_cols:
            if col == input_column:
                continue
            if outlier_method == "IQR":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR

                mask = (df[col]<lower ) | (df[col]>upper)
            
            elif outlier_method == "Z-Score":
                z_scores=np.abs(stats.zscore(df[col].dropna()))
                mask = z_scores > 3
                mask = pd.Series(mask , index=df[col].dropna().index).reindex_like(df[col])
                mask=mask.fillna(False)
            elif outlier_method == "Percentile":
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                mask = (df[col]<lower) | (df[col]>upper)
            else:
                continue
            
            outlier_ratio = mask.sum()/len(df)

            if outlier_ratio > .30:
                df.drop(columns=col,inplace=True)
            else:
                if outlier_action == "Remove Rows":
                    df = df[~mask]
                
   
    
    if skew_method_right != "None" and skew_method_left != "None":
        for col in df.select_dtypes(include=[np.number]).columns:
            if col==input_column:
                continue
            skew = df[col].skew()
            
            if skew > 0.5:
                if skew_method_right == "Square Root":
                    df[col] = np.sqrt(np.clip(df[col], a_min=0, a_max=None))
                elif skew_method_right == "Cube Root":
                    df[col] = np.cbrt(df[col])
                elif skew_method_right == "Logarithms":
                    df[col] = np.log(df[col].clip(lower=1e-9))
                elif skew_method_right == "Reciprocal":
                    df[col] = 1 / df[col].replace(0, 1e-9)
            elif skew < -0.5:
                if skew_method_left == "Square":
                    df[col] = df[col] ** 2
                elif skew_method_left == "Cube":
                    df[col] = df[col] ** 3
                  

    if input_column in df.columns and pd.api.types.is_object_dtype(df[input_column]):
        le = LabelEncoder()
        df[input_column] = le.fit_transform(df[input_column])

    y = df[input_column]
    X = df.drop(input_column,axis=1)
    if scaling_method !="None":
        if scaling_method=="Standard Scaler":
            scaler=StandardScaler()
        else:
            scaler=MinMaxScaler()
        
        numeric_cols=X.select_dtypes(include=['int64','float64']).columns
        scaled_vals=scaler.fit_transform(X[numeric_cols])
        X[numeric_cols] = pd.DataFrame(scaled_vals, columns=numeric_cols, index=X.index)
    
    category_column = X.select_dtypes(include='object').columns.tolist()
    columns_to_encode = []


    for col in category_column:
        if X[col].nunique() <= 10:
            columns_to_encode.append(col)
        else:
            X.drop(col, axis=1, inplace=True)

    if columns_to_encode:
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        encoded_data = encoder.fit_transform(X[columns_to_encode])
        
        encoded_df = pd.DataFrame(
            encoded_data,
            columns=encoder.get_feature_names_out(columns_to_encode),
            index=X.index
        )

        X.drop(columns_to_encode, axis=1, inplace=True)
        X = pd.concat([X, encoded_df], axis=1)

        


    X_train, X_test, y_train, y_test = train_test_split(
            X,y, test_size=test_size/100, random_state=42)
        
    if sampling_method !="None":
            if sampling_method == "Up Sampling":
                sampler = RandomOverSampler(random_state=42)
                X_train,y_train=sampler.fit_resample(X_train,y_train)
            elif sampling_method == "Down Sampling":
                sampler = RandomUnderSampler(random_state=42)
                X_train,y_train=sampler.fit_resample(X_train,y_train)
            elif sampling_method == "SMOTE":
                sampler = SMOTETomek(random_state=42)
                X_train,y_train=sampler.fit_resample(X_train,y_train)
    
    if problem_type == "Regression":
            if model_name == "Linear Regression":
                model = LinearRegression()
            elif model_name == "Random Forest":
                model = RandomForestRegressor()
            elif model_name == "Decision Tree":
                model = DecisionTreeRegressor()
            elif model_name == "SVR":
                model = SVR()
                
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results_df = pd.DataFrame({
                    "Actual": y_test.values,
                    "Predicted": y_pred
                })
            results_df.to_csv("results.csv",index=False)
                
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            mae = np.mean(np.abs(y_test - y_pred))
                
            if hasattr(model, 'coef_'):
                if len(model.coef_.shape) == 1:
                    coefficients = dict(zip(X.columns, model.coef_))
                else:
                    coefficients = "Multiple coefficients "
            else:
                coefficients = "Feature importances not available for this model"
                
            report = f"""
                Model: {model_name} ({problem_type})
                Target: {target_column}
                
                
                Model Metrics:
                - R² Score: {r2:.4f}
                - Mean Absolute Error: {mae:.4f}
                - Mean Squared Error: {mse:.4f}
                - Root Mean Squared Error: {rmse:.4f}
                
                """
            return report,"results.csv"
                
    else:  # Classification
            if model_name == "Logistic Regression":
                model = LogisticRegression()
            elif model_name == "Random Forest":
                model = RandomForestClassifier()
            elif model_name == "Decision Tree":
                model = DecisionTreeClassifier()
            elif model_name == "SVC":
                model = SVC()
                
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results_df = pd.DataFrame({
                    "Actual": y_test.values,
                    "Predicted": y_pred
                })
            results_df.to_csv("results.csv",index=False)
                
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            class_report = classification_report(y_test, y_pred)
                
            if hasattr(model, 'feature_importances_'):
                importances = dict(zip(X.columns, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                if len(model.coef_.shape) == 1:
                    importances = dict(zip(X.columns, model.coef_[0]))
                else:
                    importances = "Multiple coefficients (check model details)"
            else:
                importances = "Feature importances not available for this model"
                
            report = f"""
                Model: {model_name} ({problem_type})
                Target: {target_column}
                
                
                Model Metrics:
                - Accuracy: {accuracy:.4f}
                - Precision: {precision:.4f}
                - Recall: {recall:.4f}
                - F1 Score: {f1:.4f}
                
                Classification Report:
                {class_report}
                
                """
            return report,"results.csv"


    
with gr.Blocks() as demo:
    with gr.Row():
        file_upload=gr.File(label="Upload CSV File",file_types=['.csv'])

        target_column=gr.Dropdown([],label="Target Column",interactive=True)

    

    with gr.Row():
        problem_type=gr.Dropdown(
            ["Regression","Classification"],
            label='Problem Type',
            value="Regression",interactive=True)
        
        model_name=gr.Dropdown(
            ["Linear Regression","Random Forest","Decision Tree","SVR"],
            label="Model",
            value="Linear Regression",
            interactive=True
        )
        
   
    drop_column=gr.CheckboxGroup([],label="Columns to Drop")

    with gr.Row():
        corr_threshold=gr.Slider(
            0,1,value=0.95,step=0.01,
            label="Correlation Threshold (Remove Features Above)"
        )
        imputation=gr.Dropdown(
            ["Mean","Median","None"],
            label="Imputation",
            value="None",
            interactive=True,
        )
        
    with gr.Row():
        test_size=gr.Slider(
            0,50,value=20,step=0.05,
            label="Test Size"
        )
        sampling_method=gr.Dropdown(
            ["None","Up Sampling","Down Sampling","SMOTE"],
            label="Sampling Method",
            value="None",
            interactive=True,
        )
        scaling_method=gr.Dropdown(
            ["None","Standard Scaler","MinMax Scaler"],
            label="Scaling Method",
            value="None",
            interactive=True
        )
    with gr.Row():
        skew_method_right=gr.Dropdown(
            ["None","Square Root","Cube Root","Logarithms","Reciprocal"],
            label="Right Skewness Method",
            value="None",
            interactive=True
        )
        skew_method_left=gr.Dropdown(
            ["None","Square","Cube"],
            label="Left Skewness Method",
            value="None",
            interactive=True
        )
    with gr.Row():
        outlier_method=gr.Dropdown(
            ["None","IQR","Z-Score","Percentile"],
            label="Outlier Detection Method",
            value="None",
            interactive=True
        )
        outlier_action=gr.Dropdown(
            ["None","Remove Rows"],
            label="Outlier Handling Action",
            value="None",
            interactive=True
        )


    run_button=gr.Button("Run",variant="primary")

    
    output_features=gr.TextArea(label="Processed Data",lines=15)

    download_csv = gr.File(label= "Download Prediction CSV")

    def update_model_options(problem_type):
        if problem_type=="Regression":
            return gr.Dropdown(choices=["Linear Regression","Random Forest","Decision Tree","SVR"],
            value="Linear Regression",
            label="Model",
            interactive=True)
        else:
            return  gr.Dropdown(
                choices=["Logistic Regression","Random Forest","Decision Tree","SVC"],
                value="Logistic Regression",
                label="Model",
                interactive=True)
    problem_type.change(
        update_model_options,
        inputs=problem_type,
        outputs=model_name
    )

    def update_column(csv_file):
        if csv_file is None:
            return [gr.Dropdown(choices=[]),gr.CheckboxGroup(choices=[])]
        df=pd.read_csv(csv_file.name)
        columns=list(df.columns)
        return [
            gr.Dropdown(choices=columns,value=columns[-1] if columns else None,interactive=True),
            gr.CheckboxGroup(choices=columns)
        ]
    file_upload.change(
        update_column,
        inputs=[file_upload],
        outputs=[target_column,drop_column]
    )

    run_button.click(
        run_model,
        inputs=[
            file_upload,
            problem_type,
            model_name,
            target_column,
            drop_column,
            corr_threshold,
            imputation,
            scaling_method,
            test_size,
            sampling_method,
            skew_method_right,
            skew_method_left,
            outlier_method,
            outlier_action,
        ],
        outputs=[output_features,download_csv]
    )

if __name__ == "__main__":
    demo.launch(debug=True)