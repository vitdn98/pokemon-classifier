import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Set page config
st.set_page_config(
    page_title="Pokémon Database Explorer",
    page_icon="⚡",
    layout="wide"
)

# Load the Pokemon data
@st.cache_data
def load_pokemon_data():
    df = pd.read_csv('pokemon.csv')
    # Clean up the data
    df['Name'] = df['Name'].str.replace('Mega ', '')  # Clean up Mega names
    df['Type 2'] = df['Type 2'].fillna('None')  # Fill NaN with 'None'
    return df

# Load data
pokemon_df = load_pokemon_data()

# Sidebar for filters
st.sidebar.title("🔍 Filters")

# Search by name
search_name = st.sidebar.text_input("Search by Pokemon name:", "").lower()

# Filter by type
all_types = sorted(pokemon_df['Type 1'].unique().tolist() + pokemon_df['Type 2'].unique().tolist())
all_types = [t for t in all_types if t != 'None']
selected_types = st.sidebar.multiselect("Filter by Type:", all_types)

# Filter by generation
generations = sorted(pokemon_df['Generation'].unique())
selected_generations = st.sidebar.multiselect("Filter by Generation:", generations)

# Filter by legendary status
legendary_filter = st.sidebar.selectbox("Legendary Status:", ["All", "Legendary", "Non-Legendary"])

# Filter by total stats range
min_total, max_total = st.sidebar.slider(
    "Total Stats Range:",
    min_value=int(pokemon_df['Total'].min()),
    max_value=int(pokemon_df['Total'].max()),
    value=(int(pokemon_df['Total'].min()), int(pokemon_df['Total'].max()))
)

# Apply filters
filtered_df = pokemon_df.copy()

if search_name:
    filtered_df = filtered_df[filtered_df['Name'].str.lower().str.contains(search_name)]

if selected_types:
    filtered_df = filtered_df[
        (filtered_df['Type 1'].isin(selected_types)) | 
        (filtered_df['Type 2'].isin(selected_types))
    ]

if selected_generations:
    filtered_df = filtered_df[filtered_df['Generation'].isin(selected_generations)]

if legendary_filter == "Legendary":
    filtered_df = filtered_df[filtered_df['Legendary'] == True]
elif legendary_filter == "Non-Legendary":
    filtered_df = filtered_df[filtered_df['Legendary'] == False]

filtered_df = filtered_df[
    (filtered_df['Total'] >= min_total) & 
    (filtered_df['Total'] <= max_total)
]

# Main content
st.title("⚡ Pokémon Database Explorer")
st.markdown("Explore the complete Pokemon database with advanced filtering and visualization capabilities.")

# Display stats
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Pokemon", len(pokemon_df))
with col2:
    st.metric("Filtered Results", len(filtered_df))
with col3:
    st.metric("Legendary Pokemon", len(pokemon_df[pokemon_df['Legendary'] == True]))
with col4:
    st.metric("Average Total Stats", f"{pokemon_df['Total'].mean():.1f}")

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["📋 Pokemon List", "📊 Statistics", "🎯 Type Predictor", "⚔️ Battle Compare"])

with tab1:
    st.subheader("Pokemon List")
    
    # Sort options
    col1, col2 = st.columns([1, 2])
    with col1:
        sort_by = st.selectbox("Sort by:", ["Name", "Total", "HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed", "Generation"])
    with col2:
        sort_order = st.selectbox("Order:", ["Ascending", "Descending"])
    
    # Apply sorting
    ascending = sort_order == "Ascending"
    filtered_df_sorted = filtered_df.sort_values(sort_by, ascending=ascending)
    
    # Display Pokemon in a table
    if len(filtered_df_sorted) > 0:
        # Create a more detailed display
        for idx, pokemon in filtered_df_sorted.head(50).iterrows():  # Show first 50 results
            with st.expander(f"#{pokemon['#']} {pokemon['Name']} ({pokemon['Type 1']}{'/' + pokemon['Type 2'] if pokemon['Type 2'] != 'None' else ''})"):
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col1:
                    st.write(f"**Generation:** {pokemon['Generation']}")
                    st.write(f"**Total Stats:** {pokemon['Total']}")
                    if bool(pokemon['Legendary']):
                        st.write("🌟 **Legendary**")
                
                with col2:
                    # Create radar chart for stats
                    stats = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
                    values = [int(pokemon[stat]) for stat in stats]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=stats,
                        fill='toself',
                        name=pokemon['Name']
                    ))
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, max(values) * 1.2]
                            )),
                        showlegend=False,
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col3:
                    st.write("**Base Stats:**")
                    st.write(f"HP: {pokemon['HP']}")
                    st.write(f"Attack: {pokemon['Attack']}")
                    st.write(f"Defense: {pokemon['Defense']}")
                    st.write(f"Sp. Atk: {pokemon['Sp. Atk']}")
                    st.write(f"Sp. Def: {pokemon['Sp. Def']}")
                    st.write(f"Speed: {pokemon['Speed']}")
        
        if len(filtered_df_sorted) > 50:
            st.info(f"Showing first 50 results. Total filtered results: {len(filtered_df_sorted)}")
    else:
        st.warning("No Pokemon match your current filters. Try adjusting your search criteria.")

with tab2:
    st.subheader("Database Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Type distribution
        type_counts = pd.concat([pokemon_df['Type 1'], pokemon_df['Type 2']]).value_counts()
        type_counts = type_counts[type_counts.index != 'None']
        
        fig = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            title="Pokemon Type Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Generation distribution
        gen_counts = pokemon_df['Generation'].value_counts().sort_index()
        fig = px.bar(
            x=gen_counts.index,
            y=gen_counts.values,
            title="Pokemon by Generation",
            labels={'x': 'Generation', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Total stats distribution
        fig = px.histogram(
            pokemon_df,
            x='Total',
            nbins=30,
            title="Distribution of Total Stats"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Legendary vs Non-Legendary
        legendary_counts = pokemon_df['Legendary'].value_counts()
        fig = px.pie(
            values=legendary_counts.values,
            names=['Non-Legendary', 'Legendary'],
            title="Legendary vs Non-Legendary"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Advanced statistics
    st.subheader("Advanced Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Average Stats by Type:**")
        avg_stats_by_type = pokemon_df.groupby('Type 1')[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']].mean()
        st.dataframe(avg_stats_by_type.round(1))
    
    with col2:
        st.write("**Top 10 Pokemon by Total Stats:**")
        top_pokemon = pokemon_df.nlargest(10, 'Total')[['Name', 'Type 1', 'Type 2', 'Total']]
        st.dataframe(top_pokemon)

with tab3:
    st.subheader("Pokemon Type Predictor")
    st.write("Predict a Pokemon's type based on its base stats.")
    
    # Use the full dataset for training
    X = pokemon_df[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']]
    y = pokemon_df['Type 1']
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Enter Pokemon Stats:**")
        hp = st.number_input('HP', min_value=1, max_value=255, value=50, key='pred_hp')
        attack = st.number_input('Attack', min_value=1, max_value=255, value=50, key='pred_attack')
        defense = st.number_input('Defense', min_value=1, max_value=255, value=50, key='pred_defense')
        sp_atk = st.number_input('Sp. Atk', min_value=1, max_value=255, value=50, key='pred_sp_atk')
        sp_def = st.number_input('Sp. Def', min_value=1, max_value=255, value=50, key='pred_sp_def')
        speed = st.number_input('Speed', min_value=1, max_value=255, value=50, key='pred_speed')
        
        if st.button('Predict Type'):
            input_stats = np.array([[hp, attack, defense, sp_atk, sp_def, speed]])
            prediction = model.predict(input_stats)
            probabilities = model.predict_proba(input_stats)
            
            st.success(f"**Predicted Type:** {prediction[0]}")
            
            # Show top 3 predictions
            top_3_idx = np.argsort(probabilities[0])[-3:][::-1]
            st.write("**Top 3 Predictions:**")
            for i, idx in enumerate(top_3_idx):
                prob = probabilities[0][idx] * 100
                st.write(f"{i+1}. {model.classes_[idx]}: {prob:.1f}%")
    
    with col2:
        st.write("**Model Information:**")
        st.write(f"Training data: {len(pokemon_df)} Pokemon")
        st.write(f"Number of types: {len(model.classes_)}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed'],
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(
            feature_importance,
            x='Feature',
            y='Importance',
            title="Feature Importance for Type Prediction"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Pokemon Battle Comparison")
    st.write("Compare two Pokemon's stats and battle potential.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Pokemon 1:**")
        pokemon1_name = st.selectbox("Select Pokemon 1:", pokemon_df['Name'].unique(), key='pokemon1')
        pokemon1 = pokemon_df[pokemon_df['Name'] == pokemon1_name].iloc[0]
        
        st.write(f"**Type:** {pokemon1['Type 1']}{'/' + pokemon1['Type 2'] if pokemon1['Type 2'] != 'None' else ''}")
        st.write(f"**Total Stats:** {pokemon1['Total']}")
        if bool(pokemon1['Legendary']):
            st.write("🌟 **Legendary**")
    
    with col2:
        st.write("**Pokemon 2:**")
        pokemon2_name = st.selectbox("Select Pokemon 2:", pokemon_df['Name'].unique(), key='pokemon2')
        pokemon2 = pokemon_df[pokemon_df['Name'] == pokemon2_name].iloc[0]
        
        st.write(f"**Type:** {pokemon2['Type 1']}{'/' + pokemon2['Type 2'] if pokemon2['Type 2'] != 'None' else ''}")
        st.write(f"**Total Stats:** {pokemon2['Total']}")
        if bool(pokemon2['Legendary']):
            st.write("🌟 **Legendary**")
    
    # Comparison chart
    if pokemon1_name != pokemon2_name:
        stats = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[int(pokemon1[stat]) for stat in stats],
            theta=stats,
            fill='toself',
            name=pokemon1_name
        ))
        fig.add_trace(go.Scatterpolar(
            r=[int(pokemon2[stat]) for stat in stats],
            theta=stats,
            fill='toself',
            name=pokemon2_name
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(max([int(pokemon1[stat]) for stat in stats]), 
                               max([int(pokemon2[stat]) for stat in stats])) * 1.2]
                )),
            showlegend=True,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Battle analysis
        st.subheader("Battle Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            p1_total = pokemon1['Total']
            p2_total = pokemon2['Total']
            if p1_total > p2_total:
                st.success(f"**{pokemon1_name}** has higher total stats")
            elif p2_total > p1_total:
                st.success(f"**{pokemon2_name}** has higher total stats")
            else:
                st.info("Both Pokemon have equal total stats")
        
        with col2:
            p1_attack = pokemon1['Attack'] + pokemon1['Sp. Atk']
            p2_attack = pokemon2['Attack'] + pokemon2['Sp. Atk']
            if p1_attack > p2_attack:
                st.success(f"**{pokemon1_name}** has higher attack potential")
            elif p2_attack > p1_attack:
                st.success(f"**{pokemon2_name}** has higher attack potential")
            else:
                st.info("Both Pokemon have equal attack potential")
        
        with col3:
            p1_defense = pokemon1['Defense'] + pokemon1['Sp. Def']
            p2_defense = pokemon2['Defense'] + pokemon2['Sp. Def']
            if p1_defense > p2_defense:
                st.success(f"**{pokemon1_name}** has higher defense potential")
            elif p2_defense > p1_defense:
                st.success(f"**{pokemon2_name}** has higher defense potential")
            else:
                st.info("Both Pokemon have equal defense potential")

# Footer
st.markdown("---")
st.markdown("**Pokemon Database Explorer** - Built with Streamlit and Plotly")
