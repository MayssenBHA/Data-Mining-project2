import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import umap
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="AI Clustering & Insight Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1f77b4;
        font-weight: 700;
    }
    h2, h3 {
        color: #2c3e50;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Fonction pour prétraiter les données
@st.cache_data
def preprocess_data(df):
    """Prétraite les données en gérant les valeurs manquantes et en encodant les variables catégorielles"""
    df_processed = df.copy()
    
    # Séparation des colonnes numériques et catégorielles
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    
    # Gestion des valeurs manquantes pour les colonnes numériques
    for col in numeric_cols:
        df_processed[col].fillna(df_processed[col].median(), inplace=True)
    
    # Gestion des valeurs manquantes pour les colonnes catégorielles
    for col in categorical_cols:
        df_processed[col].fillna(df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'Unknown', inplace=True)
    
    # Encodage des variables catégorielles
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col + '_encoded'] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le
    
    return df_processed, numeric_cols, categorical_cols, label_encoders

# Fonction pour réduire la dimensionnalité
@st.cache_data
def reduce_dimensions(data, method='PCA', n_components=2, perplexity=30, n_neighbors=15, min_dist=0.1):
    """Réduit la dimensionnalité des données"""
    if method == 'PCA':
        reducer = PCA(n_components=n_components, random_state=42)
        reduced_data = reducer.fit_transform(data)
        variance_explained = reducer.explained_variance_ratio_
        return reduced_data, variance_explained
    
    elif method == 't-SNE':
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42, n_iter=1000)
        reduced_data = reducer.fit_transform(data)
        return reduced_data, None
    
    elif method == 'UMAP':
        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        reduced_data = reducer.fit_transform(data)
        return reduced_data, None
    
    return data, None

# Fonction pour appliquer le clustering
@st.cache_data
def apply_clustering(data, algorithm='K-Means', n_clusters=3, eps=0.5, min_samples=5, max_eps=np.inf):
    """Applique l'algorithme de clustering sélectionné"""
    if algorithm == 'K-Means':
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(data)
        centers = model.cluster_centers_
        return labels, centers
    
    elif algorithm == 'DBSCAN':
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(data)
        # DBSCAN n'a pas de centres prédéfinis, on calcule les centroïdes
        unique_labels = set(labels)
        centers = []
        for label in unique_labels:
            if label != -1:  # Ignorer le bruit
                cluster_points = data[labels == label]
                centers.append(cluster_points.mean(axis=0))
        centers = np.array(centers) if centers else None
        return labels, centers
    
    elif algorithm == 'GMM':
        model = GaussianMixture(n_components=n_clusters, random_state=42)
        labels = model.fit_predict(data)
        centers = model.means_
        return labels, centers
    
    elif algorithm == 'OPTICS':
        model = OPTICS(min_samples=min_samples, max_eps=max_eps)
        labels = model.fit_predict(data)
        unique_labels = set(labels)
        centers = []
        for label in unique_labels:
            if label != -1:
                cluster_points = data[labels == label]
                centers.append(cluster_points.mean(axis=0))
        centers = np.array(centers) if centers else None
        return labels, centers
    
    elif algorithm == 'K-Medoids':
        model = KMedoids(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(data)
        centers = model.cluster_centers_
        return labels, centers
    
    return None, None

# Fonction pour calculer les métriques de clustering
def calculate_metrics(data, labels):
    """Calcule les métriques de qualité du clustering"""
    # Filtrer les points de bruit (label -1) pour les algorithmes basés sur la densité
    mask = labels != -1
    if mask.sum() < 2:
        return None, None, None, 0
    
    filtered_data = data[mask]
    filtered_labels = labels[mask]
    
    # Vérifier qu'il y a au moins 2 clusters
    n_clusters = len(set(filtered_labels))
    if n_clusters < 2:
        return None, None, None, n_clusters
    
    try:
        silhouette = silhouette_score(filtered_data, filtered_labels)
        davies_bouldin = davies_bouldin_score(filtered_data, filtered_labels)
        calinski_harabasz = calinski_harabasz_score(filtered_data, filtered_labels)
        return silhouette, davies_bouldin, calinski_harabasz, n_clusters
    except:
        return None, None, None, n_clusters

# Fonction pour générer l'analyse IA
def generate_ai_insights(df, labels, numeric_cols):
    """Génère des insights automatiques sur les clusters"""
    insights = []
    unique_labels = sorted(set(labels))
    
    for label in unique_labels:
        if label == -1:
            insights.append({
                'cluster': 'Bruit/Outliers',
                'size': sum(labels == label),
                'description': 'Points considérés comme du bruit ou des outliers par l\'algorithme'
            })
            continue
        
        cluster_data = df[labels == label][numeric_cols]
        cluster_size = len(cluster_data)
        
        # Calculer les statistiques du cluster
        stats = {}
        for col in numeric_cols[:5]:  # Limiter aux 5 premières colonnes pour la lisibilité
            mean_val = cluster_data[col].mean()
            overall_mean = df[col].mean()
            diff_pct = ((mean_val - overall_mean) / overall_mean) * 100 if overall_mean != 0 else 0
            stats[col] = {
                'mean': mean_val,
                'diff_pct': diff_pct
            }
        
        # Générer une description
        description = f"Cluster de {cluster_size} éléments. "
        significant_features = []
        for col, stat in stats.items():
            if abs(stat['diff_pct']) > 20:
                direction = "supérieur" if stat['diff_pct'] > 0 else "inférieur"
                significant_features.append(f"{col} {direction} de {abs(stat['diff_pct']):.1f}%")
        
        if significant_features:
            description += "Caractéristiques: " + ", ".join(significant_features[:3])
        else:
            description += "Profil proche de la moyenne générale"
        
        insights.append({
            'cluster': f'Cluster {label}',
            'size': cluster_size,
            'description': description
        })
    
    return insights

# Interface principale
def main():
    st.title("🔬 AI Clustering & Insight Dashboard")
    st.markdown("### Explorez, segmentez et visualisez vos données avec l'IA")
    
    # Sidebar pour le chargement des données
    st.sidebar.header("📊 Chargement des données")
    
    # Option pour charger un fichier
    uploaded_file = st.sidebar.file_uploader(
        "Choisir un fichier CSV",
        type=['csv'],
        help="Téléchargez votre fichier de données au format CSV"
    )
    
    if uploaded_file is not None:
        # Charger les données
        df = pd.read_csv(uploaded_file)
        
        st.sidebar.success(f"✅ Fichier chargé: {uploaded_file.name}")
        st.sidebar.metric("Nombre de lignes", df.shape[0])
        st.sidebar.metric("Nombre de colonnes", df.shape[1])
        
        # Aperçu des données
        with st.expander("📋 Aperçu des données", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Types de données:**")
                st.write(df.dtypes.value_counts())
            with col2:
                st.write("**Valeurs manquantes:**")
                missing = df.isnull().sum()
                if missing.sum() > 0:
                    st.write(missing[missing > 0])
                else:
                    st.write("Aucune valeur manquante")
        
        # Prétraitement
        df_processed, numeric_cols, categorical_cols, label_encoders = preprocess_data(df)
        
        # Sélection des colonnes pour le clustering
        st.sidebar.header("🎯 Configuration du clustering")
        
        available_cols = numeric_cols.copy()
        for col in categorical_cols:
            available_cols.append(col + '_encoded')
        
        selected_cols = st.sidebar.multiselect(
            "Sélectionner les colonnes pour le clustering",
            available_cols,
            default=numeric_cols[:min(5, len(numeric_cols))],
            help="Choisissez les variables à utiliser pour le clustering"
        )
        
        if len(selected_cols) < 2:
            st.warning("⚠️ Veuillez sélectionner au moins 2 colonnes pour le clustering")
            return
        
        # Préparer les données pour le clustering
        X = df_processed[selected_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Configuration de la réduction de dimensionnalité
        st.sidebar.header("📉 Réduction de dimensionnalité")
        reduction_method = st.sidebar.selectbox(
            "Méthode de réduction",
            ['PCA', 't-SNE', 'UMAP'],
            help="Choisissez la méthode pour visualiser les données en 2D/3D"
        )
        
        n_components = st.sidebar.selectbox("Nombre de composantes", [2, 3], index=0)
        
        # Paramètres spécifiques selon la méthode
        if reduction_method == 't-SNE':
            perplexity = st.sidebar.slider("Perplexité", 5, 50, 30, help="Équilibre entre aspects locaux et globaux")
        else:
            perplexity = 30
        
        if reduction_method == 'UMAP':
            n_neighbors = st.sidebar.slider("Nombre de voisins", 2, 100, 15)
            min_dist = st.sidebar.slider("Distance minimale", 0.0, 1.0, 0.1, 0.05)
        else:
            n_neighbors = 15
            min_dist = 0.1
        
        # Configuration du clustering
        st.sidebar.header("🎲 Algorithme de clustering")
        clustering_algorithm = st.sidebar.selectbox(
            "Algorithme",
            ['K-Means', 'DBSCAN', 'GMM', 'OPTICS', 'K-Medoids'],
            help="Sélectionnez l'algorithme de clustering à utiliser"
        )
        
        # Paramètres spécifiques selon l'algorithme
        if clustering_algorithm in ['K-Means', 'GMM', 'K-Medoids']:
            n_clusters = st.sidebar.slider("Nombre de clusters (k)", 2, 10, 3)
        else:
            n_clusters = 3
        
        if clustering_algorithm in ['DBSCAN', 'OPTICS']:
            eps = st.sidebar.slider("Epsilon (ε)", 0.1, 5.0, 0.5, 0.1, help="Distance maximale entre deux points")
            min_samples = st.sidebar.slider("Min samples", 2, 20, 5, help="Nombre minimum de points pour former un cluster")
            if clustering_algorithm == 'OPTICS':
                max_eps = st.sidebar.slider("Max epsilon", 1.0, 10.0, 5.0, 0.5)
            else:
                max_eps = np.inf
        else:
            eps = 0.5
            min_samples = 5
            max_eps = np.inf
        
        # Bouton pour lancer l'analyse
        if st.sidebar.button("🚀 Lancer l'analyse", type="primary", use_container_width=True):
            with st.spinner("🔄 Analyse en cours..."):
                
                # Réduction de dimensionnalité
                X_reduced, variance_explained = reduce_dimensions(
                    X_scaled, reduction_method, n_components, perplexity, n_neighbors, min_dist
                )
                
                # Clustering
                labels, centers = apply_clustering(
                    X_scaled, clustering_algorithm, n_clusters, eps, min_samples, max_eps
                )
                
                # Calculer les métriques
                silhouette, davies_bouldin, calinski, n_found_clusters = calculate_metrics(X_scaled, labels)
                
                # Stocker dans session state
                st.session_state['analysis_done'] = True
                st.session_state['X_reduced'] = X_reduced
                st.session_state['labels'] = labels
                st.session_state['centers'] = centers
                st.session_state['variance_explained'] = variance_explained
                st.session_state['silhouette'] = silhouette
                st.session_state['davies_bouldin'] = davies_bouldin
                st.session_state['calinski'] = calinski
                st.session_state['n_found_clusters'] = n_found_clusters
                st.session_state['reduction_method'] = reduction_method
                st.session_state['clustering_algorithm'] = clustering_algorithm
                st.session_state['df_processed'] = df_processed
                st.session_state['numeric_cols'] = numeric_cols
                
            st.success("✅ Analyse terminée!")
            st.rerun()
        
        # Affichage des résultats
        if st.session_state.get('analysis_done', False):
            X_reduced = st.session_state['X_reduced']
            labels = st.session_state['labels']
            centers = st.session_state['centers']
            variance_explained = st.session_state['variance_explained']
            silhouette = st.session_state['silhouette']
            davies_bouldin = st.session_state['davies_bouldin']
            calinski = st.session_state['calinski']
            n_found_clusters = st.session_state['n_found_clusters']
            reduction_method = st.session_state['reduction_method']
            clustering_algorithm = st.session_state['clustering_algorithm']
            
            # Métriques en haut
            st.header("📊 Métriques de performance")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Nombre de clusters", n_found_clusters)
            
            with col2:
                if silhouette is not None:
                    st.metric("Silhouette Score", f"{silhouette:.3f}", 
                             help="Entre -1 et 1. Plus c'est proche de 1, meilleur c'est")
                else:
                    st.metric("Silhouette Score", "N/A")
            
            with col3:
                if davies_bouldin is not None:
                    st.metric("Davies-Bouldin Index", f"{davies_bouldin:.3f}",
                             help="Plus c'est proche de 0, meilleur c'est")
                else:
                    st.metric("Davies-Bouldin Index", "N/A")
            
            with col4:
                if calinski is not None:
                    st.metric("Calinski-Harabasz", f"{calinski:.0f}",
                             help="Plus c'est élevé, meilleur c'est")
                else:
                    st.metric("Calinski-Harabasz", "N/A")
            
            # Visualisations
            st.header("📈 Visualisations")
            
            tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Scatter Plot", "📊 Distribution", "🎯 Profils", "🤖 Insights IA"])
            
            with tab1:
                # Scatter plot 2D ou 3D
                if n_components == 2:
                    fig = px.scatter(
                        x=X_reduced[:, 0],
                        y=X_reduced[:, 1],
                        color=labels.astype(str),
                        title=f"Visualisation des clusters - {reduction_method}",
                        labels={'x': 'Composante 1', 'y': 'Composante 2', 'color': 'Cluster'},
                        color_discrete_sequence=px.colors.qualitative.Bold,
                        width=800,
                        height=600
                    )
                    
                    # Ajouter les centres si disponibles
                    if centers is not None and reduction_method == 'PCA':
                        centers_reduced, _ = reduce_dimensions(centers, 'PCA', n_components)
                        fig.add_trace(go.Scatter(
                            x=centers_reduced[:, 0],
                            y=centers_reduced[:, 1],
                            mode='markers',
                            marker=dict(size=20, symbol='x', color='black', line=dict(width=2)),
                            name='Centres',
                            showlegend=True
                        ))
                    
                    if variance_explained is not None:
                        fig.add_annotation(
                            text=f"Variance expliquée: {sum(variance_explained)*100:.1f}%",
                            xref="paper", yref="paper",
                            x=0.02, y=0.98,
                            showarrow=False,
                            bgcolor="white",
                            bordercolor="black",
                            borderwidth=1
                        )
                    
                    fig.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(size=12)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                else:  # 3D
                    fig = px.scatter_3d(
                        x=X_reduced[:, 0],
                        y=X_reduced[:, 1],
                        z=X_reduced[:, 2],
                        color=labels.astype(str),
                        title=f"Visualisation 3D des clusters - {reduction_method}",
                        labels={'x': 'Composante 1', 'y': 'Composante 2', 'z': 'Composante 3', 'color': 'Cluster'},
                        color_discrete_sequence=px.colors.qualitative.Bold,
                        width=800,
                        height=700
                    )
                    
                    if variance_explained is not None:
                        fig.add_annotation(
                            text=f"Variance expliquée: {sum(variance_explained)*100:.1f}%",
                            xref="paper", yref="paper",
                            x=0.02, y=0.98,
                            showarrow=False
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # Distribution des clusters
                unique_labels, counts = np.unique(labels, return_counts=True)
                
                fig = px.bar(
                    x=[f"Cluster {l}" if l != -1 else "Bruit" for l in unique_labels],
                    y=counts,
                    title="Distribution des points par cluster",
                    labels={'x': 'Cluster', 'y': 'Nombre de points'},
                    color=counts,
                    color_continuous_scale='Blues'
                )
                fig.update_layout(showlegend=False, plot_bgcolor='white', paper_bgcolor='white')
                st.plotly_chart(fig, use_container_width=True)
                
                # Tableau des tailles
                st.subheader("📋 Taille des clusters")
                cluster_sizes = pd.DataFrame({
                    'Cluster': [f"Cluster {l}" if l != -1 else "Bruit/Outliers" for l in unique_labels],
                    'Nombre de points': counts,
                    'Pourcentage': [f"{(c/len(labels))*100:.1f}%" for c in counts]
                })
                st.dataframe(cluster_sizes, use_container_width=True, hide_index=True)
            
            with tab3:
                # Profils des clusters
                st.subheader("🎯 Profils des clusters")
                
                df_with_clusters = df_processed.copy()
                df_with_clusters['Cluster'] = labels
                
                # Radar chart pour chaque cluster
                unique_labels_sorted = sorted([l for l in set(labels) if l != -1])
                
                if len(numeric_cols) >= 3:
                    selected_features = numeric_cols[:min(6, len(numeric_cols))]
                    
                    for label in unique_labels_sorted[:4]:  # Limiter à 4 clusters pour la lisibilité
                        cluster_data = df_with_clusters[df_with_clusters['Cluster'] == label][selected_features]
                        overall_data = df_processed[selected_features]
                        
                        # Normaliser les valeurs
                        cluster_means = []
                        overall_means = []
                        
                        for col in selected_features:
                            cluster_mean = cluster_data[col].mean()
                            overall_mean = overall_data[col].mean()
                            overall_std = overall_data[col].std()
                            
                            if overall_std > 0:
                                cluster_normalized = (cluster_mean - overall_mean) / overall_std
                            else:
                                cluster_normalized = 0
                            
                            cluster_means.append(cluster_normalized)
                            overall_means.append(0)  # Moyenne générale = 0 après normalisation
                        
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatterpolar(
                            r=cluster_means + [cluster_means[0]],
                            theta=selected_features + [selected_features[0]],
                            fill='toself',
                            name=f'Cluster {label}'
                        ))
                        
                        fig.add_trace(go.Scatterpolar(
                            r=overall_means + [overall_means[0]],
                            theta=selected_features + [selected_features[0]],
                            fill='toself',
                            name='Moyenne générale',
                            line=dict(dash='dash')
                        ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(visible=True, range=[-3, 3])
                            ),
                            showlegend=True,
                            title=f"Profil du Cluster {label} (scores standardisés)",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                # Heatmap des moyennes par cluster
                st.subheader("🔥 Heatmap des caractéristiques")
                
                cluster_profiles = []
                for label in unique_labels_sorted:
                    profile = df_with_clusters[df_with_clusters['Cluster'] == label][numeric_cols[:10]].mean()
                    cluster_profiles.append(profile)
                
                if cluster_profiles:
                    heatmap_data = pd.DataFrame(cluster_profiles)
                    heatmap_data.index = [f'Cluster {i}' for i in unique_labels_sorted]
                    
                    # Normaliser pour la heatmap
                    heatmap_normalized = (heatmap_data - heatmap_data.mean()) / heatmap_data.std()
                    
                    fig = px.imshow(
                        heatmap_normalized.T,
                        labels=dict(x="Cluster", y="Caractéristique", color="Valeur standardisée"),
                        x=heatmap_normalized.index,
                        y=heatmap_normalized.columns,
                        color_continuous_scale='RdBu_r',
                        aspect='auto',
                        title="Profil des clusters (valeurs standardisées)"
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                # Insights IA
                st.subheader("🤖 Analyse automatique par IA")
                
                insights = generate_ai_insights(df_processed, labels, numeric_cols)
                
                for insight in insights:
                    with st.container():
                        st.markdown(f"### {insight['cluster']}")
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.metric("Taille", insight['size'])
                        with col2:
                            st.info(insight['description'])
                        st.divider()
                
                # Résumé global
                st.subheader("📝 Résumé de l'analyse")
                
                quality_text = ""
                if silhouette is not None:
                    if silhouette > 0.5:
                        quality_text = "**Excellente** séparation des clusters"
                    elif silhouette > 0.3:
                        quality_text = "**Bonne** séparation des clusters"
                    elif silhouette > 0:
                        quality_text = "**Faible** séparation des clusters"
                    else:
                        quality_text = "**Mauvaise** séparation des clusters"
                
                # Format metrics safely
                sil_text = f"{silhouette:.3f}" if silhouette is not None else "N/A"
                db_text = f"{davies_bouldin:.3f}" if davies_bouldin is not None else "N/A"
                
                st.markdown(f"""
                **Configuration:**
                - Algorithme: {clustering_algorithm}
                - Méthode de réduction: {reduction_method}
                - Nombre de clusters identifiés: {n_found_clusters}
                
                **Qualité du clustering:**
                - {quality_text}
                - Silhouette Score: {sil_text}
                - Davies-Bouldin Index: {db_text}
                
                **Recommandations:**
                """)
                
                # Ajouter les recommandations sans backslash dans f-string
                if silhouette and silhouette > 0.5:
                    st.markdown("- ✅ Les clusters sont bien définis et distincts.")
                else:
                    st.markdown("- ⚠️ Envisagez de tester différents paramètres ou algorithmes pour améliorer la séparation.")
                
                if davies_bouldin and davies_bouldin < 1.5:
                    st.markdown("- ✅ Le nombre de clusters semble approprié.")
                else:
                    st.markdown("- 💡 Essayez différentes valeurs de k ou d'epsilon pour optimiser le clustering.")
                
                # Télécharger les résultats
                st.subheader("💾 Télécharger les résultats")
                
                results_df = df_processed.copy()
                results_df['Cluster'] = labels
                results_df['Component_1'] = X_reduced[:, 0]
                results_df['Component_2'] = X_reduced[:, 1]
                
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Télécharger les données avec clusters",
                    data=csv,
                    file_name='clustering_results.csv',
                    mime='text/csv',
                    use_container_width=True
                )
    
    else:
        # Message d'accueil
        st.info("👈 Commencez par charger un fichier CSV dans la barre latérale")
        
        st.markdown("""
        ### 🚀 Fonctionnalités principales
        
        - **Algorithmes de clustering:** K-Means, DBSCAN, GMM, OPTICS, K-Medoids
        - **Réduction de dimensionnalité:** PCA, t-SNE, UMAP
        - **Métriques de qualité:** Silhouette, Davies-Bouldin, Calinski-Harabasz
        - **Visualisations interactives:** Scatter plots 2D/3D, Radar charts, Heatmaps
        - **Insights IA:** Analyse automatique et recommandations
        
        ### 📊 Types de données supportés
        
        - Données clients, transactions, produits
        - Données RH, employés
        - Données de ventes, marketing
        - Tout dataset tabulaire au format CSV
        
        ### 🎯 Comment utiliser
        
        1. Chargez votre fichier CSV
        2. Sélectionnez les colonnes à analyser
        3. Choisissez la méthode de réduction de dimensionnalité
        4. Sélectionnez l'algorithme de clustering
        5. Ajustez les hyperparamètres
        6. Lancez l'analyse et explorez les résultats
        """)

if __name__ == "__main__":
    main()
