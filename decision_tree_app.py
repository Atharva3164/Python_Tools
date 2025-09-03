import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_blobs, load_iris, load_wine
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Interactive Decision Tree Visualizer",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def generate_synthetic_data(dataset_type, n_samples=500, noise=0.1, random_state=42):
    """Generate different types of synthetic datasets"""
    np.random.seed(random_state)
    
    if dataset_type == "Moons":
        from sklearn.datasets import make_moons
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif dataset_type == "Circles":
        from sklearn.datasets import make_circles
        X, y = make_circles(n_samples=n_samples, noise=noise, random_state=random_state)
    elif dataset_type == "Blobs":
        X, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=1.0, 
                         random_state=random_state)
    elif dataset_type == "Classification":
        X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0,
                                 n_informative=2, n_clusters_per_class=1,
                                 random_state=random_state)
    else:  # Linear
        X = np.random.randn(n_samples, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    return X, y

def plot_decision_boundary(X, y, model, title="Decision Boundary"):
    """Plot decision boundary with Plotly"""
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    fig = go.Figure()
    
    # Add decision boundary
    fig.add_trace(go.Contour(
        x=np.arange(x_min, x_max, h),
        y=np.arange(y_min, y_max, h),
        z=Z,
        showscale=False,
        opacity=0.2,
        colorscale='RdYlBu',
        line=dict(width=0)
    ))
    
    # Add data points with bright colors like your example
    colors = ['#00ff00', '#00bfff', '#ff6b35', '#ffd700', '#ff1493']  # Bright green, cyan, orange, gold, pink
    unique_classes = np.unique(y)
    
    for i, class_label in enumerate(unique_classes):
        mask = y == class_label
        fig.add_trace(go.Scatter(
            x=X[mask, 0],
            y=X[mask, 1],
            mode='markers',
            name=f'Class {class_label}',
            marker=dict(
                color=colors[i % len(colors)],
                size=6,
                opacity=0.8
            )
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(color='white', size=16)),
        xaxis=dict(
            title="Feature 1",
            gridcolor='rgba(128, 128, 128, 0.3)',
            zerolinecolor='rgba(128, 128, 128, 0.5)',
            color='white'
        ),
        yaxis=dict(
            title="Feature 2", 
            gridcolor='rgba(128, 128, 128, 0.3)',
            zerolinecolor='rgba(128, 128, 128, 0.5)',
            color='white'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30, 30, 30, 1)',
        height=400,
        showlegend=True,
        legend=dict(
            font=dict(color='white'),
            bgcolor='rgba(0,0,0,0.5)'
        )
    )
    
    return fig

def plot_scatter_with_splits(X, y, model, title="Dataset with Decision Splits", show_splits=True):
    """Plot scatter plot with decision tree splits overlay"""
    fig = go.Figure()
    
    # Colors matching your example - bright green and cyan
    colors = ['#00ff41', '#00bfff', '#ff6b35', '#ffd700', '#ff1493']
    unique_classes = np.unique(y)
    
    # Add data points
    for i, class_label in enumerate(unique_classes):
        mask = y == class_label
        fig.add_trace(go.Scatter(
            x=X[mask, 0],
            y=X[mask, 1],
            mode='markers',
            name=f'Class {class_label}',
            marker=dict(
                color=colors[i % len(colors)],
                size=6,
                opacity=0.9,
                line=dict(width=0)
            )
        ))
    
    # Add decision splits if requested
    if show_splits and hasattr(model, 'tree_'):
        tree = model.tree_
        feature = tree.feature
        threshold = tree.threshold
        
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        
        # Get split lines for visualization
        split_lines = []
        
        def add_splits(node_id, x_min_curr, x_max_curr, y_min_curr, y_max_curr, depth=0):
            if depth > 3:  # Limit depth for visualization
                return
                
            if tree.feature[node_id] != -2:  # Not a leaf
                split_feature = tree.feature[node_id]
                split_threshold = tree.threshold[node_id]
                
                if split_feature == 0:  # Split on feature 1 (x-axis)
                    # Vertical line
                    split_lines.append({
                        'type': 'line',
                        'x0': split_threshold, 'x1': split_threshold,
                        'y0': y_min_curr, 'y1': y_max_curr,
                        'line': dict(color='rgba(255, 255, 255, 0.6)', width=2, dash='dash')
                    })
                    
                    # Recurse on children
                    left_child = tree.children_left[node_id]
                    right_child = tree.children_right[node_id]
                    
                    if left_child != -1:
                        add_splits(left_child, x_min_curr, split_threshold, y_min_curr, y_max_curr, depth+1)
                    if right_child != -1:
                        add_splits(right_child, split_threshold, x_max_curr, y_min_curr, y_max_curr, depth+1)
                        
                elif split_feature == 1:  # Split on feature 2 (y-axis)
                    # Horizontal line
                    split_lines.append({
                        'type': 'line',
                        'x0': x_min_curr, 'x1': x_max_curr,
                        'y0': split_threshold, 'y1': split_threshold,
                        'line': dict(color='rgba(255, 255, 255, 0.6)', width=2, dash='dash')
                    })
                    
                    # Recurse on children
                    left_child = tree.children_left[node_id]
                    right_child = tree.children_right[node_id]
                    
                    if left_child != -1:
                        add_splits(left_child, x_min_curr, x_max_curr, y_min_curr, split_threshold, depth+1)
                    if right_child != -1:
                        add_splits(right_child, x_min_curr, x_max_curr, split_threshold, y_max_curr, depth+1)
        
        # Start recursive split visualization
        add_splits(0, x_min, x_max, y_min, y_max)
        
        # Add all split lines to the plot
        fig.update_layout(shapes=split_lines)
    
    fig.update_layout(
        title=dict(text=title, font=dict(color='white', size=16)),
        xaxis=dict(
            gridcolor='rgba(128, 128, 128, 0.3)',
            zerolinecolor='rgba(128, 128, 128, 0.5)',
            color='white',
            showgrid=True
        ),
        yaxis=dict(
            gridcolor='rgba(128, 128, 128, 0.3)',
            zerolinecolor='rgba(128, 128, 128, 0.5)',
            color='white',
            showgrid=True
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30, 30, 30, 1)',
        height=400,
        showlegend=True,
        legend=dict(
            font=dict(color='white'),
            bgcolor='rgba(0,0,0,0.5)'
        ),
        font=dict(color='white')
    )
    
    return fig

def get_split_info(model, feature_names):
    """Extract split information from the trained model"""
    tree = model.tree_
    split_info = []
    
    def traverse_tree(node_id, depth=0, parent_condition="Root"):
        if tree.feature[node_id] != -2:  # Not a leaf
            feature_name = feature_names[tree.feature[node_id]]
            threshold = tree.threshold[node_id]
            samples = tree.n_node_samples[node_id]
            
            split_info.append({
                'Node': node_id,
                'Depth': depth,
                'Condition': parent_condition,
                'Split': f"{feature_name} <= {threshold:.3f}",
                'Samples': samples,
                'Feature': feature_name,
                'Threshold': threshold
            })
            
            # Traverse left child
            left_child = tree.children_left[node_id]
            if left_child != -1:
                left_condition = f"{feature_name} <= {threshold:.3f}"
                traverse_tree(left_child, depth+1, left_condition)
            
            # Traverse right child
            right_child = tree.children_right[node_id]
            if right_child != -1:
                right_condition = f"{feature_name} > {threshold:.3f}"
                traverse_tree(right_child, depth+1, right_condition)
    
    traverse_tree(0)
    return split_info

def plot_tree_structure(model, feature_names=None, max_depth_display=3):
    """Plot tree structure using matplotlib"""
    fig, ax = plt.subplots(figsize=(15, 10))
    
    plot_tree(model, 
             feature_names=feature_names,
             class_names=[f'Class {i}' for i in model.classes_],
             filled=True,
             rounded=True,
             fontsize=10,
             max_depth=max_depth_display,
             ax=ax)
    
    plt.title(f"Decision Tree Structure (showing max depth {max_depth_display})")
    return fig

def calculate_tree_metrics(model, X_train, X_test, y_train, y_test):
    """Calculate various tree metrics"""
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    n_nodes = model.tree_.node_count
    n_leaves = model.get_n_leaves()
    depth = model.get_depth()
    
    return {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'n_nodes': n_nodes,
        'n_leaves': n_leaves,
        'depth': depth,
        'overfitting': train_acc - test_acc
    }

def main():
    # Header
    st.markdown('<h1 class="main-header">🌳 Interactive Decision Tree Visualizer</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Experiment with different hyperparameters and see how they affect decision tree 
    performance in real-time. Visualize decision boundaries, tree structure, and 
    performance metrics.
    """)
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Dataset selection
        st.subheader("Dataset Configuration")
        dataset_type = st.selectbox(
            "Choose Dataset Type:",
            ["Moons", "Circles", "Blobs", "Classification", "Linear", "Iris", "Wine"]
        )
        
        if dataset_type not in ["Iris", "Wine"]:
            n_samples = st.slider("Number of Samples:", 100, 1000, 500, 50)
            noise = st.slider("Noise Level:", 0.0, 0.5, 0.1, 0.05)
        
        random_state = st.number_input("Random State:", 0, 100, 42)
        
        # Hyperparameters
        st.subheader("🔧 Hyperparameters")
        
        # Split Criterion
        criterion = st.selectbox(
            "Criterion:", ["gini", "entropy"],
            help="Function to measure the quality of a split"
        )
        
        # Splitter Strategy
        splitter = st.selectbox(
            "Splitter:", ["best", "random"],
            help="Strategy to choose the split at each node"
        )
        
        # Max Depth with None option
        max_depth_option = st.selectbox(
            "Max Depth:", ["None", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
            index=5,  # Default to 5
            help="Maximum depth of the tree. None means no limit."
        )
        max_depth = None if max_depth_option == "None" else int(max_depth_option)
        
        # Min Samples Split
        min_samples_split = st.slider(
            "Min Samples Split:", 2, 50, 2,
            help="Minimum samples required to split an internal node"
        )
        
        # Min Samples Leaf
        min_samples_leaf = st.slider(
            "Min Samples Leaf:", 1, 20, 1,
            help="Minimum samples required to be at a leaf node"
        )
        
        # Min Weight Fraction Leaf
        min_weight_fraction_leaf = st.slider(
            "Min Weight Fraction Leaf:", 0.0, 0.5, 0.0, 0.01,
            help="Minimum weighted fraction of samples required to be at a leaf node"
        )
        
        # Max Features
        max_features_option = st.selectbox(
            "Max Features:", ["None", "sqrt", "log2", "auto"],
            help="Number of features to consider when looking for the best split"
        )
        if max_features_option == "None":
            max_features = None
        elif max_features_option == "auto":
            max_features = "sqrt"  # auto is deprecated, use sqrt
        else:
            max_features = max_features_option
        
        # Display options
        st.subheader("📊 Display Options")
        plot_type = st.radio(
            "Plot Type:", 
            ["Scatter Plot Only", "With Decision Splits", "With Decision Boundary"],
            help="Choose visualization type"
        )
        
        if plot_type == "With Decision Splits":
            show_splits = st.checkbox("Show Split Lines", True, 
                                    help="Show decision tree split lines on the plot")
        else:
            show_splits = True  # Default value when not using splits
        
        show_tree_structure = st.checkbox("Show Tree Structure", True)
        max_depth_display = st.slider("Max Tree Depth to Display:", 1, 5, 3)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Generate or load data
        if dataset_type == "Iris":
            iris = load_iris()
            X, y = iris.data[:, :2], iris.target  # Use only first 2 features
            feature_names = ['Sepal Length', 'Sepal Width']
        elif dataset_type == "Wine":
            wine = load_wine()
            X, y = wine.data[:, :2], wine.target  # Use only first 2 features
            feature_names = ['Feature 1', 'Feature 2']
        else:
            X, y = generate_synthetic_data(dataset_type, n_samples, noise, random_state)
            feature_names = ['Feature 1', 'Feature 2']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=random_state
        )
        
        # Create and train model
        model = DecisionTreeClassifier(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            random_state=random_state
        )
        
        model.fit(X_train, y_train)
        
        # Visualization plot
        st.subheader("🎯 Data Visualization")
        if plot_type == "Scatter Plot Only":
            scatter_fig = plot_scatter_with_splits(X, y, model, f"{dataset_type} Dataset", show_splits=False)
            st.plotly_chart(scatter_fig, use_container_width=True)
        elif plot_type == "With Decision Splits":
            splits_fig = plot_scatter_with_splits(X, y, model, 
                                                f"{dataset_type} Dataset with Decision Splits", 
                                                show_splits=show_splits)
            st.plotly_chart(splits_fig, use_container_width=True)
        else:
            boundary_fig = plot_decision_boundary(X, y, model, 
                                                f"Decision Boundary - {dataset_type} Dataset")
            st.plotly_chart(boundary_fig, use_container_width=True)
        
        # Tree structure
        if show_tree_structure:
            st.subheader("🌲 Tree Structure")
            tree_fig = plot_tree_structure(model, feature_names, max_depth_display)
            st.pyplot(tree_fig)
    
    with col2:
        # Performance metrics
        st.subheader("📈 Performance Metrics")
        
        metrics = calculate_tree_metrics(model, X_train, X_test, y_train, y_test)
        
        # Display metrics with styling
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Train Accuracy", f"{metrics['train_accuracy']:.3f}")
            st.metric("Nodes", metrics['n_nodes'])
            st.metric("Depth", metrics['depth'])
        
        with col_b:
            st.metric("Test Accuracy", f"{metrics['test_accuracy']:.3f}")
            st.metric("Leaves", metrics['n_leaves'])
            st.metric("Overfitting", f"{metrics['overfitting']:.3f}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Overfitting indicator
        if metrics['overfitting'] > 0.1:
            st.warning("⚠️ High overfitting detected! Consider reducing max_depth or increasing min_samples_split.")
        elif metrics['overfitting'] < 0.02:
            st.success("✅ Good generalization!")
        else:
            st.info("ℹ️ Moderate overfitting - fine-tune parameters.")
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            st.subheader("🎯 Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            })
            
            fig_importance = px.bar(
                importance_df, 
                x='Importance', 
                y='Feature',
                orientation='h',
                title="Feature Importance"
            )
            fig_importance.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(30, 30, 30, 0.8)',
                font=dict(color='white'),
                title=dict(font=dict(color='white'))
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        
        # Decision Splits Information
        if plot_type == "With Decision Splits":
            st.subheader("🔀 Split Details")
            split_info = get_split_info(model, feature_names)
            if split_info:
                splits_df = pd.DataFrame(split_info)
                st.dataframe(
                    splits_df[['Node', 'Depth', 'Split', 'Samples']],
                    use_container_width=True
                )
    
    # Additional Analysis Section
    col3, col4 = st.columns(2)
    
    with col3:
        # Confusion Matrix
        st.subheader("🔍 Confusion Matrix")
        test_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, test_pred)
        
        fig_cm = px.imshow(
            cm, 
            text_auto=True, 
            aspect="auto",
            title="Confusion Matrix (Test Set)",
            labels=dict(x="Predicted", y="Actual")
        )
        fig_cm.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            title=dict(font=dict(color='white'))
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col4:
        # Hyperparameter Effects Visualization
        st.subheader("📊 Parameter Effects")
        
        # Show different effects based on selected parameters
        if splitter == "random":
            st.warning("🎲 **Random Splitter Active**\nTree structure may vary between runs")
        
        if max_depth is None:
            st.info("🌳 **Unlimited Depth**\nTree can grow until other criteria stop it")
        
        if min_weight_fraction_leaf > 0:
            st.info(f"⚖️ **Weight Constraint**\nLeaves need ≥{min_weight_fraction_leaf:.1%} of total sample weight")
        
        # Parameter impact summary
        st.markdown("**Current Setup Impact:**")
        
        impact_metrics = {
            "Overfitting Risk": "High" if (max_depth is None or max_depth > 10) and min_samples_leaf == 1 else "Medium" if max_depth and max_depth > 5 else "Low",
            "Model Complexity": "High" if model.get_depth() > 8 else "Medium" if model.get_depth() > 4 else "Low",
            "Interpretability": "Low" if model.get_n_leaves() > 50 else "Medium" if model.get_n_leaves() > 10 else "High"
        }
        
        for metric, value in impact_metrics.items():
            color = "🔴" if value == "High" else "🟡" if value == "Medium" else "🟢"
            st.write(f"{color} **{metric}:** {value}")
        
        # Additional hyperparameter info
        with st.expander("ℹ️ Current Hyperparameters Summary"):
            st.write(f"**Criterion:** {criterion}")
            st.write(f"**Splitter:** {splitter}")
            st.write(f"**Max Depth:** {'Unlimited' if max_depth is None else max_depth}")
            st.write(f"**Min Samples Split:** {min_samples_split}")
            st.write(f"**Min Samples Leaf:** {min_samples_leaf}")
            st.write(f"**Min Weight Fraction Leaf:** {min_weight_fraction_leaf}")
            st.write(f"**Max Features:** {'All features' if max_features is None else max_features}")
            st.write(f"**Random State:** {random_state}")
            
            # Show actual tree parameters after training
            st.write("---")
            st.write("**Actual Tree Results:**")
            st.write(f"**Final Tree Depth:** {model.get_depth()}")
            st.write(f"**Total Nodes:** {model.tree_.node_count}")
            st.write(f"**Leaf Nodes:** {model.get_n_leaves()}")
            
            if max_depth is None:
                st.info("💡 Max Depth is None - tree will grow until other stopping criteria are met")
            
            if splitter == "random":
                st.info("🎲 Random splitter adds randomness - results may vary between runs even with same random_state")
    
    # Tree Rules Section
    st.subheader("📋 Decision Tree Rules")
    with st.expander("View Tree Rules (Text Format)"):
        tree_rules = export_text(model, feature_names=feature_names)
        st.text(tree_rules)
    
    # Educational Notes
    with st.expander("📚 Educational Notes"):
        st.markdown("""
        ### Understanding All Hyperparameters:
        
        - **Criterion**: 
          - *Gini*: Measures impurity, faster to compute, default choice
          - *Entropy*: Information-theoretic measure, may yield slightly different trees
        
        - **Splitter**:
          - *Best*: Choose the best split among all features (deterministic)
          - *Random*: Choose the best random split (adds randomness, can reduce overfitting)
        
        - **Max Depth**: 
          - Controls tree complexity. Set to None for unlimited depth
          - Deeper trees can capture more patterns but may overfit
        
        - **Min Samples Split**: Higher values prevent splits on small groups, reducing overfitting
        
        - **Min Samples Leaf**: Ensures leaf nodes have enough samples for reliable predictions
        
        - **Min Weight Fraction Leaf**: Fraction of total sample weights required at leaf (useful for imbalanced datasets)
        
        - **Max Features**: 
          - *None*: Consider all features at each split
          - *sqrt*: Consider √(n_features) features
          - *log2*: Consider log₂(n_features) features
          - Limiting features adds randomness and can reduce overfitting
        
        ### Overfitting Indicators:
        - Large gap between train and test accuracy
        - Very deep trees with many leaves
        - High variance in performance across different random states
        
        ### Tips:
        - Start with default parameters and adjust based on performance
        - Use cross-validation for robust evaluation
        - Consider ensemble methods if single trees underperform
        """)

if __name__ == "__main__":
    main()
# streamlit run decision_tree_app.py  -- in terminal  @Atharva3164 - github
