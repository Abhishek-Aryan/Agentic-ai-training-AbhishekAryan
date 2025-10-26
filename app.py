import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score, 
                           classification_report, confusion_matrix, silhouette_score)
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Comprehensive Data Analysis Tool",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .analysis-option {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

class ComprehensiveDataAnalyzer:
    def _init_(self, data):
        self.data = data
        self.numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
        
    def descriptive_statistics(self):
        st.markdown('<div class="section-header">📊 Descriptive Statistics</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Dataset Overview")
            st.write(f"*Shape:* {self.data.shape}")
            st.write(f"*Memory usage:* {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
        with col2:
            st.subheader("Data Types")
            dtype_info = self.data.dtypes.value_counts()
            for dtype, count in dtype_info.items():
                st.write(f"{dtype}:** {count} columns")
                
        with col3:
            st.subheader("Missing Values")
            missing_total = self.data.isnull().sum().sum()
            missing_percent = (missing_total / (self.data.shape[0] * self.data.shape[1])) * 100
            st.write(f"*Total missing:* {missing_total}")
            st.write(f"*Percentage:* {missing_percent:.2f}%")
        
        # Basic statistics
        st.subheader("Basic Statistics")
        st.dataframe(self.data.describe())
        
        # Missing values detail
        if self.data.isnull().sum().sum() > 0:
            st.subheader("Missing Values Detail")
            missing_df = pd.DataFrame({
                'Column': self.data.columns,
                'Missing_Count': self.data.isnull().sum().values,
                'Missing_Percentage': (self.data.isnull().sum().values / len(self.data)) * 100
            })
            st.dataframe(missing_df[missing_df['Missing_Count'] > 0])
    
    def distributions_analysis(self):
        st.markdown('<div class="section-header">📈 Distributions Analysis</div>', unsafe_allow_html=True)
        
        if not self.numeric_columns:
            st.warning("No numeric columns available for distribution analysis.")
            return
            
        # Select columns for distribution analysis
        selected_columns = st.multiselect(
            "Select columns for distribution analysis:",
            self.numeric_columns,
            default=self.numeric_columns[:min(3, len(self.numeric_columns))]
        )
        
        if selected_columns:
            # Histograms
            st.subheader("Histograms")
            fig, axes = plt.subplots(1, len(selected_columns), figsize=(5*len(selected_columns), 4))
            if len(selected_columns) == 1:
                axes = [axes]
                
            for i, col in enumerate(selected_columns):
                axes[i].hist(self.data[col].dropna(), bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Box plots
            st.subheader("Box Plots")
            fig, axes = plt.subplots(1, len(selected_columns), figsize=(5*len(selected_columns), 4))
            if len(selected_columns) == 1:
                axes = [axes]
                
            for i, col in enumerate(selected_columns):
                axes[i].boxplot(self.data[col].dropna())
                axes[i].set_title(f'Box Plot of {col}')
                axes[i].set_ylabel(col)
            plt.tight_layout()
            st.pyplot(fig)
    
    def correlation_analysis(self):
        st.markdown('<div class="section-header">🔗 Correlation Analysis</div>', unsafe_allow_html=True)
        
        if len(self.numeric_columns) < 2:
            st.warning("Need at least 2 numeric columns for correlation analysis.")
            return
            
        # Correlation matrix
        st.subheader("Correlation Matrix")
        corr_matrix = self.data[self.numeric_columns].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Correlation Matrix Heatmap')
        st.pyplot(fig)
        
        # Strong correlations
        st.subheader("Strong Correlations")
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # Threshold for strong correlation
                    strong_corr.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': corr_val
                    })
        
        if strong_corr:
            strong_corr_df = pd.DataFrame(strong_corr)
            st.dataframe(strong_corr_df.sort_values('Correlation', key=abs, ascending=False))
        else:
            st.info("No strong correlations (|r| > 0.7) found.")
    
    def classification_analysis(self):
        st.markdown('<div class="section-header">🎯 Machine Learning - Classification</div>', unsafe_allow_html=True)
        
        if len(self.numeric_columns) < 2 or not self.categorical_columns:
            st.warning("Need at least 2 numeric columns and 1 categorical column for classification.")
            return
            
        # Select target and features
        target_col = st.selectbox("Select target variable:", self.categorical_columns)
        feature_cols = st.multiselect(
            "Select feature variables:",
            self.numeric_columns,
            default=self.numeric_columns[:min(5, len(self.numeric_columns))]
        )
        
        if target_col and len(feature_cols) >= 1:
            # Prepare data
            X = self.data[feature_cols].fillna(self.data[feature_cols].mean())
            y = self.data[target_col]
            
            # Encode target variable
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Model Performance")
                accuracy = accuracy_score(y_test, y_pred)
                st.metric("Accuracy", f"{accuracy:.3f}")
                
                st.subheader("Classification Report")
                class_report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(class_report).transpose())
                
            with col2:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)
            
            # Feature importance
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=importance_df, x='importance', y='feature', ax=ax)
            ax.set_title('Feature Importance')
            st.pyplot(fig)
    
    def regression_analysis(self):
        st.markdown('<div class="section-header">📈 Machine Learning - Regression</div>', unsafe_allow_html=True)
        
        if len(self.numeric_columns) < 2:
            st.warning("Need at least 2 numeric columns for regression analysis.")
            return
            
        # Select target and features
        target_col = st.selectbox("Select target variable:", self.numeric_columns, key="reg_target")
        available_features = [col for col in self.numeric_columns if col != target_col]
        feature_cols = st.multiselect(
            "Select feature variables:",
            available_features,
            default=available_features[:min(5, len(available_features))],
            key="reg_features"
        )
        
        if target_col and len(feature_cols) >= 1:
            # Prepare data
            X = self.data[feature_cols].fillna(self.data[feature_cols].mean())
            y = self.data[target_col]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                mse = mean_squared_error(y_test, y_pred)
                st.metric("Mean Squared Error", f"{mse:.3f}")
                
            with col2:
                rmse = np.sqrt(mse)
                st.metric("Root Mean Squared Error", f"{rmse:.3f}")
                
            with col3:
                r2 = r2_score(y_test, y_pred)
                st.metric("R² Score", f"{r2:.3f}")
            
            # Actual vs Predicted plot
            st.subheader("Actual vs Predicted")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test, y_pred, alpha=0.6)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title('Actual vs Predicted Values')
            st.pyplot(fig)
            
            # Feature importance
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=importance_df, x='importance', y='feature', ax=ax)
            ax.set_title('Feature Importance - Regression')
            st.pyplot(fig)
    
    def clustering_analysis(self):
        st.markdown('<div class="section-header">🔍 Clustering Analysis</div>', unsafe_allow_html=True)
        
        if len(self.numeric_columns) < 2:
            st.warning("Need at least 2 numeric columns for clustering analysis.")
            return
            
        # Select features for clustering
        feature_cols = st.multiselect(
            "Select features for clustering:",
            self.numeric_columns,
            default=self.numeric_columns[:min(4, len(self.numeric_columns))],
            key="cluster_features"
        )
        
        if len(feature_cols) >= 2:
            # Prepare data
            X = self.data[feature_cols].fillna(self.data[feature_cols].mean())
            X_scaled = StandardScaler().fit_transform(X)
            
            # Determine optimal number of clusters
            st.subheader("Optimal Number of Clusters")
            wcss = []  # Within-cluster sum of squares
            silhouette_scores = []
            k_range = range(2, min(11, len(X)//10 + 2))
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                wcss.append(kmeans.inertia_)
                if k > 1:  # Silhouette score requires at least 2 clusters
                    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
            
            # Plot elbow curve and silhouette scores
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(k_range, wcss, 'bo-')
                ax.set_xlabel('Number of Clusters')
                ax.set_ylabel('WCSS')
                ax.set_title('Elbow Method')
                st.pyplot(fig)
            
            with col2:
                if silhouette_scores:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(range(2, 2 + len(silhouette_scores)), silhouette_scores, 'ro-')
                    ax.set_xlabel('Number of Clusters')
                    ax.set_ylabel('Silhouette Score')
                    ax.set_title('Silhouette Analysis')
                    st.pyplot(fig)
            
            # Perform clustering with selected k
            n_clusters = st.slider("Select number of clusters:", 
                                 min_value=2, 
                                 max_value=min(10, len(X)//10), 
                                 value=3)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Add clusters to data
            self.data['Cluster'] = clusters
            
            # Cluster visualization
            st.subheader("Cluster Visualization")
            
            if len(feature_cols) >= 2:
                # Use first two features for 2D plot
                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters, cmap='viridis', alpha=0.6)
                ax.set_xlabel(feature_cols[0])
                ax.set_ylabel(feature_cols[1])
                ax.set_title(f'K-means Clustering (k={n_clusters})')
                plt.colorbar(scatter, ax=ax)
                st.pyplot(fig)
            
            # Cluster statistics
            st.subheader("Cluster Statistics")
            cluster_stats = self.data.groupby('Cluster')[feature_cols].mean()
            st.dataframe(cluster_stats)
            
            # Cluster sizes
            st.subheader("Cluster Sizes")
            cluster_sizes = self.data['Cluster'].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=cluster_sizes.index, y=cluster_sizes.values, ax=ax)
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Number of Points')
            ax.set_title('Cluster Sizes')
            st.pyplot(fig)
    
    def statistical_tests(self):
        st.markdown('<div class="section-header">📋 Statistical Tests</div>', unsafe_allow_html=True)
        
        if not self.numeric_columns:
            st.warning("No numeric columns available for statistical tests.")
            return
            
        st.subheader("Normality Tests")
        
        selected_cols = st.multiselect(
            "Select columns for normality tests:",
            self.numeric_columns,
            default=self.numeric_columns[:min(3, len(self.numeric_columns))],
            key="normality_cols"
        )
        
        if selected_cols:
            normality_results = []
            for col in selected_cols:
                data_clean = self.data[col].dropna()
                if len(data_clean) > 3:  # Need at least 3 observations for normality tests
                    stat_sw, p_sw = stats.shapiro(data_clean)
                    stat_ks, p_ks = stats.kstest(data_clean, 'norm', 
                                               args=(data_clean.mean(), data_clean.std()))
                    
                    normality_results.append({
                        'Column': col,
                        'Shapiro-Wilk p-value': f"{p_sw:.4f}",
                        'Kolmogorov-Smirnov p-value': f"{p_ks:.4f}",
                        'Normal (α=0.05)': 'Yes' if p_sw > 0.05 else 'No'
                    })
            
            if normality_results:
                st.dataframe(pd.DataFrame(normality_results))
        
        # Correlation statistical test
        st.subheader("Statistical Correlation Test")
        
        if len(self.numeric_columns) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                var1 = st.selectbox("Select first variable:", self.numeric_columns, key="corr_var1")
            with col2:
                var2 = st.selectbox("Select second variable:", 
                                  [col for col in self.numeric_columns if col != var1], 
                                  key="corr_var2")
            
            if var1 and var2:
                data_clean = self.data[[var1, var2]].dropna()
                if len(data_clean) > 2:
                    corr_coef, p_value = stats.pearsonr(data_clean[var1], data_clean[var2])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Pearson Correlation", f"{corr_coef:.4f}")
                    with col2:
                        st.metric("P-value", f"{p_value:.4f}")
                    with col3:
                        significance = "Significant" if p_value < 0.05 else "Not Significant"
                        st.metric("Significance (α=0.05)", significance)
    
    def comprehensive_analysis(self):
        st.markdown('<div class="section-header">📑 Comprehensive Analysis</div>', unsafe_allow_html=True)
        st.info("Running all analysis types... This may take a moment for larger datasets.")
        
        # Run all analyses
        self.descriptive_statistics()
        self.distributions_analysis()
        self.correlation_analysis()
        
        if len(self.numeric_columns) >= 2 and self.categorical_columns:
            self.classification_analysis()
        
        if len(self.numeric_columns) >= 2:
            self.regression_analysis()
            self.clustering_analysis()
            self.statistical_tests()

def main():
    st.markdown('<div class="main-header">Comprehensive Data Analysis Tool</div>', unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            st.success(f"Dataset loaded successfully! Shape: {data.shape}")
            
            # Display data preview
            with st.expander("Data Preview"):
                st.dataframe(data.head())
            
            # Initialize analyzer
            analyzer = ComprehensiveDataAnalyzer(data)
            
            # Analysis options
            st.markdown("## Analysis Options")
            
            analysis_option = st.radio(
                "Choose analysis type:",
                [
                    "Descriptive Statistics",
                    "Distributions & Correlations", 
                    "Machine Learning - Classification",
                    "Machine Learning - Regression",
                    "Clustering Analysis",
                    "Statistical Tests",
                    "Comprehensive Analysis (Recommended)"
                ]
            )
            
            # Run selected analysis
            if analysis_option == "Descriptive Statistics":
                analyzer.descriptive_statistics()
                
            elif analysis_option == "Distributions & Correlations":
                analyzer.distributions_analysis()
                analyzer.correlation_analysis()
                
            elif analysis_option == "Machine Learning - Classification":
                analyzer.classification_analysis()
                
            elif analysis_option == "Machine Learning - Regression":
                analyzer.regression_analysis()
                
            elif analysis_option == "Clustering Analysis":
                analyzer.clustering_analysis()
                
            elif analysis_option == "Statistical Tests":
                analyzer.statistical_tests()
                
            elif analysis_option == "Comprehensive Analysis (Recommended)":
                analyzer.comprehensive_analysis()
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    else:
        # Show instructions when no file is uploaded
        st.markdown("""
        ## Welcome to the Comprehensive Data Analysis Tool!
        
        This application provides a complete suite of data analysis capabilities:
        
        ### Available Analyses:
        
        <div class="analysis-option">
        <strong>📊 Descriptive Statistics</strong><br>
        Comprehensive overview including distributions, correlations, and basic insights.
        </div>
        
        <div class="analysis-option">
        <strong>🎯 Machine Learning - Classification</strong><br>
        Predict categorical outcomes with feature importance analysis.
        </div>
        
        <div class="analysis-option">
        <strong>📈 Machine Learning - Regression</strong><br>
        Predict continuous values with performance metrics.
        </div>
        
        <div class="analysis-option">
        <strong>🔍 Clustering Analysis</strong><br>
        Discover natural groupings using K-means clustering algorithm.
        </div>
        
        <div class="analysis-option">
        <strong>📋 Statistical Tests</strong><br>
        Correlation analysis, normality tests, and statistical validations.
        </div>
        
        <div class="analysis-option">
        <strong>📑 Comprehensive Analysis (Recommended)</strong><br>
        All analysis types in one comprehensive report.
        </div>
        
        ### Instructions:
        1. Upload your CSV or Excel file using the uploader above
        2. Select your desired analysis type
        3. Explore the results and insights!
        
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()