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
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
from datetime import datetime
import os
from together import Together

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import resample
from imblearn.combine import SMOTETomek
from scipy import stats
from collections import Counter
from functools import partial


def run_all_models(X_train, X_test, y_train, y_test, problem_type):
    results = []
    
    if problem_type == "Regression":
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "SVR": SVR(),
            "KNN": KNeighborsRegressor(),
            "XGBoost": xgb.XGBRegressor(),
            "CatBoost": CatBoostRegressor(verbose=False)
        }
    else:
        models = {
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "SVC": SVC(),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "XGBoost": xgb.XGBClassifier(),
            "CatBoost": CatBoostClassifier(verbose=False)
        }
    
    for model_name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            if problem_type == "Regression":
                metrics = {
                    "R² Score": r2_score(y_test, y_pred),
                    "MAE": np.mean(np.abs(y_test - y_pred)),
                    "MSE": mean_squared_error(y_test, y_pred),
                    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
                }
            else:
                metrics = {
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred, average='weighted'),
                    "Recall": recall_score(y_test, y_pred, average='weighted'),
                    "F1 Score": f1_score(y_test, y_pred, average='weighted')
                }
               
            
            results.append({
                "model": model_name,
                "parameters": model.get_params(),
                "metrics": metrics
            })
            
        except Exception as e:
            results.append({
                "model": model_name,
                "error": str(e)
            })
    
    return results

def format_results_table(results, problem_type):
    if problem_type == "Regression":
        headers = ["Model", "R² Score", "MAE", "MSE", "RMSE"]
    else:
        headers = ["Model", "Accuracy", "Precision", "Recall", "F1 Score",]
    
    table = "| " + " |  ".join(headers) + "  |\n"
    
    for result in results:
        if "error" in result:
            row = [result["model"], "Error: " + result["error"]] + ["N/A"] * (len(headers) - 2)
        else:
            metrics = result["metrics"]
            if problem_type == "Regression":
                row = [
                    result["model"],
                    f"{metrics['R² Score']:.4f}",
                    f"{metrics['MAE']:.4f}",
                    f"{metrics['MSE']:.4f}",
                    f"{metrics['RMSE']:.4f}"
                ]
            else:
                row = [
                    result["model"],
                    f"{metrics['Accuracy']:.4f}",
                    f"{metrics['Precision']:.4f}",
                    f"{metrics['Recall']:.4f}",
                    f"{metrics['F1 Score']:.4f}",
                ]
        table += "|  " + "  |  ".join(row) + "  |\n"
    
    return table

def create_log_file():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return os.path.join(log_dir, f"preprocessing_log_{timestamp}.txt")

def log_step(log_file, step_name, details):
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Step: {step_name}\n")
        f.write(f"{'='*50}\n")
        f.write(f"{details}\n")

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
        return "Please upload a CSV File", None, None
    
    log_file = create_log_file()
    log_step(log_file, "Initial Data Loading", f"Loading data from {csv_file.name}")

    
    df = pd.read_csv(csv_file.name)
    initial_columns = list(df.columns)
    log_step(log_file, "Initial Data Info", f"Initial columns: {initial_columns}\nShape: {df.shape}")

    if drop_column:
        df.drop(columns=drop_column, errors="ignore", inplace=True)
        log_step(log_file, "Manual Column Dropping", f"Dropped columns: {drop_column}")


    drop_nan = df.isnull().sum()
    drop_nan = drop_nan[drop_nan.values > len(df)*.30]
    if not drop_nan.empty:
        df.drop(labels=drop_nan.index, inplace=True, axis=1)
        log_step(log_file, "High Missing Value Columns Dropped", 
                f"Columns dropped due to >30% missing values:\n\n{drop_nan.to_string()}")

    
    if corr_threshold > 0:
        df_numeric = df.select_dtypes(include=['number']).drop(columns=[input_column], errors='ignore')
        if not df_numeric.empty:
            corr_matrix = df_numeric.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
            if to_drop:
                df.drop(to_drop, axis=1, inplace=True)
                log_step(log_file, "Correlation-based Feature Removal", 
                        f"Threshold: {corr_threshold}\nDropped columns: {to_drop}")

    
    if imputation != "None":
        imputed_cols = []
        for col in df.columns:
            if df[col].dtype in [float, int]:
                if imputation == "Mean":
                    df.loc[:, col] = df[col].fillna(df[col].mean())
                    imputed_cols.append((col, "Mean", df[col].mean()))
                elif imputation == "Median":
                    df.loc[:, col] = df[col].fillna(df[col].median())
                    imputed_cols.append((col, "Median", df[col].median()))
            else:
                mode_val = df[col].mode()[0]
                df.loc[:, col] = df[col].fillna(mode_val)
                imputed_cols.append((col, "Mode", mode_val))
        if imputed_cols:
            log_step(log_file, "Imputation", 
                    "Imputed columns:\n" + "\n".join([f"{col}: {method} = {value}" for col, method, value in imputed_cols]))

  
    if outlier_method != "None" and outlier_action != "None":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_info = []
        for col in numeric_cols:
            if col == input_column:
                continue
            if outlier_method == "IQR":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                mask = (df[col] < lower) | (df[col] > upper)
            elif outlier_method == "Z-Score":
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                mask = z_scores > 3
                mask = pd.Series(mask, index=df[col].dropna().index).reindex_like(df[col])
                mask = mask.fillna(False)
            elif outlier_method == "Percentile":
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                mask = (df[col] < lower) | (df[col] > upper)
            else:
                continue
            
            outlier_ratio = mask.sum()/len(df)
            if outlier_ratio > .30:
                df.drop(columns=col, inplace=True)
                outlier_info.append(f"Column {col} dropped due to {outlier_ratio:.2%} outliers")
            else:
                if outlier_action == "Remove Rows":
                    df = df[~mask]
                    outlier_info.append(f"Column {col}: {mask.sum()} rows removed")
        
        if outlier_info:
            log_step(log_file, "Outlier Handling", 
                    f"Method: {outlier_method}\nAction: {outlier_action}\n" + "\n".join(outlier_info))

    
    if skew_method_right != "None" or skew_method_left != "None":
        skew_info = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if col == input_column:
                continue
            skew = df[col].skew()
            if skew > 0.5 and skew_method_right != "None":
                if skew_method_right == "Square Root":
                    df[col] = np.sqrt(np.clip(df[col], a_min=0, a_max=None))
                elif skew_method_right == "Cube Root":
                    df[col] = np.cbrt(df[col])
                elif skew_method_right == "Logarithms":
                    pass
                elif skew_method_right == "Reciprocal":
                    df[col] = 1 / df[col].replace(0, 1e-9)
                skew_info.append(f"Column {col}: Right skew ({skew:.2f}) -> {skew_method_right}")
            elif skew < -0.5 and skew_method_left != "None":
                if skew_method_left == "Square":
                    df[col] = df[col] ** 2
                elif skew_method_left == "Cube":
                    df[col] = df[col] ** 3
                skew_info.append(f"Column {col}: Left skew ({skew:.2f}) -> {skew_method_left}")
        
        if skew_info:
            log_step(log_file, "Skewness Handling", "\n".join(skew_info))

   
    if input_column in df.columns and pd.api.types.is_object_dtype(df[input_column]):
        le = LabelEncoder()
        df[input_column] = le.fit_transform(df[input_column])
        log_step(log_file, "Target Encoding", f"Encoded target column: {input_column}")

    y = df[input_column]
    X = df.drop(input_column, axis=1)

   
    if scaling_method != "None":
        if scaling_method == "Standard Scaler":
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        scaled_vals = scaler.fit_transform(X[numeric_cols])
        X[numeric_cols] = pd.DataFrame(scaled_vals, columns=numeric_cols, index=X.index)
        log_step(log_file, "Feature Scaling", f"Method: {scaling_method}\nScaled columns: {list(numeric_cols)}")

   
    category_column = X.select_dtypes(include='object').columns.tolist()
    columns_to_encode = []
    dropped_columns_info = []
    
    for col in category_column:
        if col not in X.columns:  
            continue
        unique_count = X[col].nunique()
        if unique_count <= 10:
            columns_to_encode.append(col)
        else:
            dropped_columns_info.append((col, unique_count))
            X.drop(col, axis=1, inplace=True)
    
    if dropped_columns_info:
        log_step(log_file, "Categorical Column Dropped", 
                f"Columns dropped due to high cardinality:\n" + 
                "\n".join([f"{col}: {count} unique values" for col, count in dropped_columns_info]))

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
        log_step(log_file, "Categorical Encoding", 
                f"Encoded columns: {columns_to_encode}\nNew features created: {list(encoded_df.columns)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=42)
    log_step(log_file, "Train-Test Split", f"Test size: {test_size}%\nTraining set shape: {X_train.shape}\nTest set shape: {X_test.shape}")

    if sampling_method != "None":
        if sampling_method == "Up Sampling":
            sampler = RandomOverSampler(random_state=42)
        elif sampling_method == "Down Sampling":
            sampler = RandomUnderSampler(random_state=42)
        elif sampling_method == "SMOTE":
            sampler = SMOTETomek(random_state=42)
        
        X_train, y_train = sampler.fit_resample(X_train, y_train)
        log_step(log_file, "Sampling", f"Method: {sampling_method}\nNew training set shape: {X_train.shape}")

    
    if model_name == "Apply All":
        results = run_all_models(X_train, X_test, y_train, y_test, problem_type)
        table = format_results_table(results, problem_type)
        log_step(log_file, "Model Comparison Results", table)
        
        report = f"""
            Model Comparison Results ({problem_type})
            Target: {input_column}
            
            {table}
            """
        return report, "results.csv", log_file

    if problem_type == "Regression":
        if model_name == "Linear Regression":
            model = LinearRegression()
        elif model_name == "Random Forest":
            model = RandomForestRegressor()
        elif model_name == "Decision Tree":
            model = DecisionTreeRegressor()
        elif model_name == "SVR":
            model = SVR()
        elif model_name == "KNN":
            model = KNeighborsRegressor()
        elif model_name == "XGBoost":
            model = xgb.XGBRegressor()
        elif model_name == "CatBoost":
            model = CatBoostRegressor(verbose=False)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results_df = pd.DataFrame({
            "Actual": y_test.values.ravel(),
            "Predicted": y_pred.ravel()
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
        
        log_step(log_file, "Model Results", f"""
            Model: {model_name}
            R² Score: {r2:.4f}
            Mean Absolute Error: {mae:.4f}
            Mean Squared Error: {mse:.4f}
            Root Mean Squared Error: {rmse:.4f}
            """)
        
        report = f"""
            Model: {model_name} ({problem_type})
            Target: {input_column}
            
            
            Model Metrics:
            - R² Score: {r2:.4f}
            - Mean Absolute Error: {mae:.4f}
            - Mean Squared Error: {mse:.4f}
            - Root Mean Squared Error: {rmse:.4f}
            
            """
        return report,"results.csv", log_file

    else:  # Classification
        if model_name == "Logistic Regression":
            model = LogisticRegression()
        elif model_name == "Random Forest":
            model = RandomForestClassifier()
        elif model_name == "Decision Tree":
            model = DecisionTreeClassifier()
        elif model_name == "SVC":
            model = SVC()
        elif model_name == "KNN":
            model = KNeighborsClassifier()
        elif model_name == "Naive Bayes":
            model = GaussianNB()
        elif model_name == "XGBoost":
            model = xgb.XGBClassifier()
        elif model_name == "CatBoost":
            model = CatBoostClassifier(verbose=False)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results_df = pd.DataFrame({
            "Actual": y_test.values.ravel(),
            "Predicted": y_pred.ravel()
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
        
        log_step(log_file, "Model Results", f"""
            Model: {model_name}
            Accuracy: {accuracy:.4f}
            Precision: {precision:.4f}
            Recall: {recall:.4f}
            F1 Score: {f1:.4f}
            
            Classification Report:
            {class_report}
            """)
        
        report = f"""
            Model: {model_name} ({problem_type})
            Target: {input_column}
            
            
            Model Metrics:
            - Accuracy: {accuracy:.4f}
            - Precision: {precision:.4f}
            - Recall: {recall:.4f}
            - F1 Score: {f1:.4f}
            
            Classification Report:
            {class_report}
            
            """
        return report,"results.csv", log_file

def analyze_log_with_ai(log_file, user_query):
    try:
    
        with open(log_file, 'r') as f:
            log_content = f.read()
        
    
        client = Together(api_key='70a19ed73f75eacf938df0612fcbd9c4d3d7fd1089d6a23d9eace82f134d07f5')
        
        prompt = f"""Here is the preprocessing and model training log:
{log_content}

User Query: {user_query}

Please analyze the log and provide insights based on the user's query. Focus on explaining the preprocessing steps, model performance, and any relevant patterns or issues you notice."""

        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error analyzing log: {str(e)}"

with gr.Blocks() as demo:
    with gr.Row():
        file_upload = gr.File(label="Upload CSV File", file_types=['.csv'])
        target_column = gr.Dropdown([], label="Target Column", interactive=True)

    with gr.Row():
        problem_type=gr.Dropdown(
            ["Regression","Classification"],
            label='Problem Type',
            value="Regression",interactive=True)
        
        model_name=gr.Dropdown(
            ["Linear Regression","Random Forest","Decision Tree","SVR","KNN","XGBoost","CatBoost","Apply All"],
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

    run_button = gr.Button("Run", variant="primary")
    
    
    output_features = gr.TextArea(label="Processed Data", lines=15)
    with gr.Row():
        download_csv = gr.File(label="Download Prediction CSV")
        download_log = gr.File(label="Download Processing Log")

    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot(label="AI Analysis", height=400)
        with gr.Column():
            user_query = gr.Textbox(label="Ask about your model", placeholder="Type your question here...")
            analyze_button = gr.Button("Ask", variant="primary")

    def update_model_options(problem_type):
        if problem_type=="Regression":
            return gr.Dropdown(choices=["Linear Regression","Random Forest","Decision Tree","SVR","KNN","XGBoost","CatBoost","Apply All"],
            value="Linear Regression",
            label="Model",
            interactive=True)
        else:
            return  gr.Dropdown(
                choices=["Logistic Regression","Random Forest","Decision Tree","SVC","KNN","Naive Bayes","XGBoost","CatBoost","Apply All"],
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

    def run_and_analyze(csv_file, problem_type, model_name, input_column, drop_column, 
                       corr_threshold, imputation, scaling_method, test_size,
                       sampling_method, skew_method_right, skew_method_left,
                       outlier_method, outlier_action):

        report, results_file, log_file = run_model(
            csv_file, problem_type, model_name, input_column, drop_column,
            corr_threshold, imputation, scaling_method, test_size,
            sampling_method, skew_method_right, skew_method_left,
            outlier_method, outlier_action
        )
        
        initial_analysis = analyze_log_with_ai(log_file, "Provide a summary of the preprocessing steps and model performance")
        
        return report, results_file, log_file, [(None, initial_analysis)]

    def analyze_query(log_file, user_query, chat_history):
        if not log_file:
            return chat_history + [(user_query, "Please run the model first to generate a log file.")]
        
        response = analyze_log_with_ai(log_file, user_query)
        return chat_history + [(user_query, response)]

    run_button.click(
        run_and_analyze,
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
        outputs=[output_features, download_csv, download_log, chatbot]
    )

   
    analyze_button.click(
        analyze_query,
        inputs=[download_log, user_query, chatbot],
        outputs=[chatbot]
    )

if __name__ == "__main__":
    demo.launch(debug=True)