import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from io import BytesIO
import base64
import lightkurve as lk
import warnings
import streamlit as st
import pandas as pd
import joblib
import time


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from matplotlib.colors import LinearSegmentedColormap
from astropy.timeseries import BoxLeastSquares, LombScargle
import scipy.stats as stats


warnings.filterwarnings('ignore')


def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image file.
    
    Parameters:
    image_file (str or BytesIO): The path to the image file or BytesIO object
    """
    if isinstance(image_file, str):
        with open(image_file, "rb") as f:
            img_data = f.read()
    else:
        img_data = image_file.getvalue()
    
    base64_encoded = base64.b64encode(img_data).decode()
    
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{base64_encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    /* Make the background slightly more translucent to improve text readability */
    .main {{
        background-color: rgba(0, 0, 0, 0.5);
        padding: 20px;
        border-radius: 10px;
    }}
    /* Improve text readability */
    .css-1d391kg, .css-1c7y2kd, p, h1, h2, h3, h4, h5, h6, li {{
        color: white !important;
    }}
    /* Make the sidebar completely transparent */
    .css-1d391kg {{
        background-color: rgba(0, 0, 0, 0) !important;
    }}
    /* Add text shadow to sidebar elements for improved readability */
    .css-1d391kg .element-container {{
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.8);
    }}
    /* Make sidebar inputs more visible with semi-transparent background */
    .css-1d391kg .stSelectbox, 
    .css-1d391kg .stCheckbox, 
    .css-1d391kg .stRadio,
    .css-1d391kg .stFileUploader {{
        background-color: rgba(0, 0, 0, 0.4) !important;
        border-radius: 5px;
        padding: 5px;
    }}
    </style>
    '''
    
    st.markdown(page_bg_img, unsafe_allow_html=True)


st.set_page_config(
    page_title="EXOPLANET EXPLORER",
    page_icon="🪐",
    layout="wide"
)


st.title("ADVANCED EXOPLANET EXPLORER")
# Sidebar 
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Single Planet Explorer", "Planet Comparison", "Habitable Zone Planets", "Light Curve Explorer","Raw data analyser", "ML Prediction"])
st.info("Explore the universe of exoplanets! Select a planet to learn more about its characteristics, compare multiple planets, or analyze light curves.")



#-------------------------------------------------------------------------------

# Background image upload section





with st.sidebar:
    st.header("App Settings")
    uploaded_bg = st.file_uploader("Upload Background Image", type=["png", "jpg", "jpeg"])

    if uploaded_bg is not None:
        # Set the background with the uploaded image
        set_background(uploaded_bg)
        st.success("Background image set!")
    else:
        # Automatically use the default background image
        default_bg_path = r"exploration_of_an_astronaut-wallpaper-1920x1080.jpg"
        with open(default_bg_path, "rb") as f:
            bg_data = BytesIO(f.read())
        set_background(bg_data)
        st.success("Default background image set!")



#-------------------------------------------------------------------------------
# Sample exoplanet data with expanded database
@st.cache_data
def load_exoplanet_data():
    # Sample data including Kepler and TESS planets
    
    
    data = {
        'name': ['Kepler-452b', 'TRAPPIST-1e', 'Proxima Centauri b', 'HD 209458 b', 'K2-18b', 
                'TOI-700 d', 'WASP-12b', 'HD 189733 b', 'KELT-9b', '55 Cancri e',
                'Kepler-186f', 'Kepler-22b', 'Kepler-16b', 'GJ 1214 b', 'HD 97658 b',
                'TOI-1338 b', 'LHS 1140 b', 'WASP-121b', 'Kepler-10b', 'HD 219134 b',
                'TRAPPIST-1d', 'K2-3d', 'GJ 357 d', 'Kepler-1649c', 'HD 40307 g',
                'TOI-270d', 'Wolf 503 b', 'WASP-76b', 'Kepler-442b', 'HD 21749 b'],
        'distance_ly': [1402, 39, 4.2, 150, 124, 101, 871, 63, 670, 41,
                        578, 620, 245, 48, 70, 1300, 49, 850, 564, 21,
                        39, 147, 31, 302, 42, 73, 145, 634, 1115, 53],
        'orbital_period_days': [384.8, 6.1, 11.2, 3.5, 33, 37.4, 1.1, 2.2, 1.5, 0.7,
                               129.9, 289.9, 228.8, 1.6, 9.5, 95.2, 24.7, 1.3, 0.8, 3.1,
                               4.0, 44.6, 55.7, 19.5, 197.8, 18.9, 6.0, 1.8, 112.3, 35.6],
        'mass_earth': [5, 0.77, 1.27, 220, 8.63, 1.72, 435, 361, 984, 8.08,
                      1.2, 36, 105, 6.3, 7.9, 33, 6.7, 379, 3.3, 4.7,
                      0.3, 2.1, 6.1, 1.2, 7.1, 5.0, 4.7, 292, 2.3, 23],
        'radius_earth': [1.6, 0.92, 1.08, 1.38, 2.6, 1.19, 1.79, 1.14, 1.89, 1.88,
                        1.2, 2.4, 0.7, 2.7, 2.3, 6.9, 1.7, 1.9, 1.5, 1.6,
                        0.8, 1.5, 1.8, 1.1, 2.3, 2.0, 2.0, 1.8, 1.3, 2.7],
        'temp_kelvin': [265, 251, 234, 1450, 265, 268, 2600, 1200, 4050, 2400,
                       188, 262, 202, 574, 753, 318, 235, 2358, 1833, 740,
                       288, 284, 220, 298, 188, 254, 510, 2179, 247, 383],
        'star_type': ['G2V', 'M8V', 'M5.5V', 'G0V', 'M2.8V', 'M2V', 'F6V', 'K1-K2', 'A0V', 'K0V',
                     'M1V', 'G5V', 'K7V', 'M4.5V', 'K1V', 'F5V', 'M4.5V', 'F6V', 'G', 'K3V',
                     'M8V', 'M0V', 'M2.5V', 'M5V', 'K2.5V', 'M0V', 'K4V', 'F7V', 'K0V', 'K4.5V'],
        'year_discovered': [2015, 2017, 2016, 1999, 2015, 2020, 2008, 2005, 2017, 2004,
                           2014, 2011, 2011, 2009, 2011, 2020, 2017, 2016, 2011, 2015,
                           2017, 2016, 2019, 2020, 2012, 2019, 2018, 2016, 2015, 2019],
        'habitable': [True, True, True, False, True, True, False, False, False, False,
                     True, True, False, False, False, False, True, False, False, False,
                     True, True, True, True, True, True, False, False, True, False],
        'detection_method': ['Transit', 'Transit', 'Radial Velocity', 'Transit', 'Transit', 'Transit', 'Transit', 'Transit', 'Transit', 'Radial Velocity',
                            'Transit', 'Transit', 'Transit', 'Transit', 'Transit', 'Transit', 'Radial Velocity', 'Transit', 'Transit', 'Radial Velocity',
                            'Transit', 'Transit', 'Radial Velocity', 'Transit', 'Radial Velocity', 'Transit', 'Transit', 'Transit', 'Transit', 'Transit'],
        'kepid': ['10593626', None, None, '3835670', None, None, None, None, None, None, 
                 '8120608', '10593626', '12644769', None, None, None, None, None, '11904151', None,
                 None, None, None, '13896176', None, None, None, None, '4138008', None],
        'tic_id': [None, '270710051', None, None, '212283037', '150428135', '100100827', None, '16740101', None,
                  None, None, None, '274762761', None, '260128333', '328933398', '22129325', None, None,
                  '270710051', '349827430', '413248763', '307210830', None, '259377017', '108906786', '249731230', None, '183537452'],
    }
    return pd.DataFrame(data)


@st.cache_data
def fetch_light_curve(target_name, mission=None, sector=None, quarter=None):
    """
    Fetch light curve data for a target using lightkurve
    
    Parameters:
    target_name (str): Name of the target (Kepler ID or TIC ID)
    mission (str): 'Kepler', 'K2', or 'TESS'
    sector (int): TESS sector
    quarter (int): Kepler quarter
    
    Returns:
    lk.LightCurve or None: Light curve object or None if not found
    """
    try:
        search_result = lk.search_lightcurve(target_name, mission=mission, sector=sector, quarter=quarter)
        if len(search_result) > 0:
            lc = search_result[0].download()
            lc = lc.normalize().remove_outliers()
            return lc
        return None
    except Exception as e:
        st.error(f"Error fetching light curve: {str(e)}")
        return None


def analyze_light_curve(lc, period=None):
    """
    Analyze a light curve for periodic signals that might indicate transits
    
    Parameters:
    lc (lk.LightCurve): Light curve object
    period (float): Known period in days (if available)
    
    Returns:
    tuple: (periodogram, folded_lc) or (None, None) if analysis fails
    """
    try:
        if lc is None:
            return None, None
            
        # Clean the light curve
        clean_lc = lc.remove_nans().flatten()
        
        # If period is provided, use it. Otherwise try to detect it.
        if period is None:
            # Create a periodogram
            pg = clean_lc.to_periodogram(method='bls')
            period = pg.period_at_max_power.value
        else:
            # Convert period to units compatible with light curve time
            # Assuming light curve time is in days
            pg = None
            
        # Fold the light curve at the detected/provided period
        folded_lc = clean_lc.fold(period=period)
        
        return pg, folded_lc
    except Exception as e:
        st.error(f"Error analyzing light curve: {str(e)}")
        return None, None

# Function to download NASA Exoplanet Archive data
@st.cache_data
def download_nasa_exoplanet_data():
    """
    Download a more comprehensive dataset from the NASA Exoplanet Archive.
    This is simulated here but could be implemented with their API.
    """
    try:
        # In a real app, you would use:
        # url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+ps&format=csv"
        # df = pd.read_csv(url)
        
        # For now, let's expand our sample data to include more fields that would be in the real dataset
        df = load_exoplanet_data()
        
        # Add simulated additional columns that would be in the NASA dataset
        df['spectral_type'] = df['star_type']  # In a real app, these would be separate columns
        df['ra'] = np.random.uniform(0, 360, len(df))  # Random RA values
        df['dec'] = np.random.uniform(-90, 90, len(df))  # Random DEC values
        df['discovery_facility'] = np.random.choice(['Kepler', 'TESS', 'HARPS', 'KECK', 'Hubble'], len(df))
        
        return df
    except Exception as e:
        st.error(f"Error downloading NASA Exoplanet Archive data: {str(e)}")
        return load_exoplanet_data()  # Fall back to sample data

# Function to generate simulated transit data
def generate_transit_model(period, depth, duration, time_span=27):
    """
    Generate a simulated transit model light curve
    
    Parameters:
    period (float): Orbital period in days
    depth (float): Transit depth as a fraction
    duration (float): Transit duration in days
    time_span (float): Total time span to model in days
    
    Returns:
    tuple: (time, flux) arrays for the model
    """
    # Create time array
    time = np.arange(0, time_span, 0.02)
    
    # Initialize flux array (all ones = star's normal brightness)
    flux = np.ones_like(time)
    
    # Add transits
    for t0 in np.arange(0, time_span, period):
        # Calculate start and end of transit
        start = t0 - duration/2
        end = t0 + duration/2
        
        # Apply transit dip
        mask = (time >= start) & (time <= end)
        
        # Create a more realistic U-shaped transit
        transit_time = time[mask] - t0
        # Simple quadratic model for U-shape
        transit_shape = 1 - depth * (1 - (transit_time / (duration/2))**2)
        flux[mask] = np.clip(transit_shape, 1-depth, 1)
    
    return time, flux

# Function to plot planet parameters using real data visualizations
def plot_planet_parameters(planet_data):
    """
    Create more scientific visualizations of planet parameters
    
    Parameters:
    planet_data (pd.Series): Series containing planet parameters
    
    Returns:
    matplotlib.figure.Figure: The figure containing the visualizations
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.patch.set_alpha(0.0)
    for ax in axs.flat:
        ax.set_facecolor('none')
    
    # Planet size comparison (Earth vs. this planet)
    ax = axs[0, 0]
    radius_ratio = planet_data['radius_earth']
    
    # Draw Earth
    earth_radius = 1
    earth_circle = plt.Circle((0.3, 0.5), earth_radius/5, color='blue', alpha=0.7, label='Earth')
    ax.add_patch(earth_circle)
    ax.text(0.3, 0.2, 'Earth', ha='center')
    
    # Draw the exoplanet
    planet_radius = radius_ratio * earth_radius/5
    planet_circle = plt.Circle((0.7, 0.5), planet_radius, color='orange', alpha=0.7, label=planet_data['name'])
    ax.add_patch(planet_circle)
    ax.text(0.7, 0.2, planet_data['name'], ha='center')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title(f"Size Comparison (Radius = {radius_ratio}x Earth)")
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Orbital period visualization
    ax = axs[0, 1]
    # Plot the star
    ax.scatter([0], [0], s=300, color='yellow', edgecolor='orange', zorder=3)
    
    # Calculate orbit (circular approximation)
    theta = np.linspace(0, 2*np.pi, 100)
    orbit_radius = 0.7  # Arbitrary scale
    x = orbit_radius * np.cos(theta)
    y = orbit_radius * np.sin(theta)
    
    # Plot orbit
    ax.plot(x, y, 'k--', alpha=0.5)
    
    # Plot planet at a position on the orbit
    position = np.random.uniform(0, 2*np.pi)
    planet_x = orbit_radius * np.cos(position)
    planet_y = orbit_radius * np.sin(position)
    ax.scatter([planet_x], [planet_y], s=50*planet_data['radius_earth']/5, color='darkblue', zorder=4)
    
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.set_title(f"Orbital Period: {planet_data['orbital_period_days']:.1f} days")
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Temperature gradient
    ax = axs[1, 0]
    # Create a temperature scale
    temp_range = np.linspace(100, 3000, 100)
    y_pos = np.ones_like(temp_range) * 0.5
    
    # Create gradient
    points = np.array([temp_range, y_pos]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Create a colormap from blue (cold) to red (hot)
    cmap = LinearSegmentedColormap.from_list("temp_cmap", ["darkblue", "blue", "green", "yellow", "orange", "red"])
    
    # Create the gradient line collection
    from matplotlib.collections import LineCollection
    norm = plt.Normalize(100, 3000)
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=15)
    lc.set_array(temp_range)
    
    ax.add_collection(lc)
    
    # Mark planet's temperature
    planet_temp = planet_data['temp_kelvin']
    temp_pos = (planet_temp - 100) / (3000 - 100)
    ax.plot([planet_temp], [0.5], 'o', markersize=10, color='white', markeredgecolor='black')
    
    # Add reference temperatures
    freezing = 273.15
    boiling = 373.15
    earth_avg = 288
    
    ax.plot([freezing], [0.3], 'v', color='cyan', markersize=8)
    ax.text(freezing, 0.25, 'Water\nFreezes', ha='center', fontsize=8)
    
    ax.plot([boiling], [0.3], 'v', color='lightblue', markersize=8)
    ax.text(boiling, 0.25, 'Water\nBoils', ha='center', fontsize=8)
    
    ax.plot([earth_avg], [0.7], '^', color='green', markersize=8)
    ax.text(earth_avg, 0.75, 'Earth\nAvg', ha='center', fontsize=8)
    
    ax.set_xlim(100, 3000)
    ax.set_ylim(0, 1)
    ax.set_title(f"Temperature: {planet_temp} K")
    ax.set_xlabel("Temperature (K)")
    ax.set_yticks([])
    
    # Mass comparison
    ax = axs[1, 1]
    mass_ratio = planet_data['mass_earth']
    
    # Create a "scale" visualization
    # Earth on left
    ax.scatter([0.25], [0.6], s=100, color='blue', label='Earth')
    ax.text(0.25, 0.8, 'Earth\n1x', ha='center')
    
    # Planet on right
    size_factor = min(np.power(mass_ratio, 1/3), 5)  # Cube root relationship, capped
    ax.scatter([0.75], [0.6], s=100*size_factor, color='orange', label=planet_data['name'])
    ax.text(0.75, 0.8, f"{planet_data['name']}\n{mass_ratio:.1f}x", ha='center')
    
    # Draw balance beam
    ax.plot([0.1, 0.9], [0.4, 0.4], 'k-', lw=3)
    ax.plot([0.5, 0.5], [0.2, 0.4], 'k-', lw=3)
    
    # If masses are very different, tilt the beam
    if mass_ratio > 2:
        # Draw tilted beam
        ax.plot([0.1, 0.9], [0.5, 0.3], 'k--', lw=2, alpha=0.5)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Mass: {mass_ratio:.1f} x Earth")
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    return fig

# Load data
exoplanet_df = load_exoplanet_data()

# Single Planet Explorer
if page == "Single Planet Explorer":
    st.header("Single Planet Explorer")
    
    # Planet selection
    selected_planet = st.selectbox("SELECT AND EXOPLANET TO EXPLORE:", exoplanet_df['name'])
    
    #get the planets data
    planet_data = exoplanet_df[exoplanet_df['name'] == selected_planet].iloc[0]
    
    # Display in columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader(f"Details: {selected_planet}")
        st.write(f"**Distance from Earth:** {planet_data['distance_ly']} light years")
        st.write(f"**Year Discovered:** {planet_data['year_discovered']}")
        st.write(f"**Detection Method:** {planet_data['detection_method']}")
        st.write(f"**Star Type:** {planet_data['star_type']}")
        st.write(f"**Potentially Habitable:** {'Yes' if planet_data['habitable'] else 'No'}")
        
        # Habitability indicator
        if planet_data['habitable']:
            st.success("This planet is in the habitable zone of its star!")
        else:
            st.warning("This planet is not in the habitable zone of its star.")
    
    
    with col2:
         # Plot scientific visualizations instead of simple bar charts
         with st.spinner("Generating planet visualizations..."):
            fig = plot_planet_parameters(planet_data)
            st.pyplot(fig)
        
    # Try to fetch light curve data if available
    kepid = planet_data.get('kepid')
    tic_id = planet_data.get('tic_id')

    light_curve = None

    if kepid:
        st.subheader("Kepler Light Curve")
        with st.spinner(f"Fetching Kepler light curve for {selected_planet}..."):
            light_curve = fetch_light_curve(f"KIC {kepid}")

    elif tic_id:
        st.subheader("TESS Light Curve")
        with st.spinner(f"Fetching TESS light curve for {selected_planet}..."):
            light_curve = fetch_light_curve(f"TIC {tic_id}")

    if light_curve:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_alpha(0.0)
        ax.set_facecolor('none')

        # Plot the light curve
        light_curve.plot(ax=ax, color='cyan')
        ax.set_title(f"Light curve for {selected_planet}")
        ax.set_xlabel("Time (BTJD)")
        ax.set_ylabel("Normalized Flux")
        plt.tight_layout()

        st.pyplot(fig)

        # Show transit markers if it's a transiting planet
        if planet_data.get('detection_method') == 'Transit':
            st.info("This is a transiting planet. The dips in the light curve may represent planet transits.")

            # Try to fold the light curve at the known period
            period = planet_data.get('orbital_period_days')
            if period:
                st.subheader("Folded Light Curve")
                with st.spinner("Folding light curve at known orbital period..."):
                    _, folded_lc = analyze_light_curve(light_curve, period=period)

                    if folded_lc is not None:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        fig.patch.set_alpha(0.0)
                        ax.set_facecolor('none')

                        # Plot the folded light curve
                        folded_lc.plot(ax=ax, color='cyan', label='Folded Light Curve')
                        ax.set_title(f"Folded Light Curve at Period = {period:.2f} days")
                        ax.set_xlabel("Phase")
                        ax.set_ylabel("Normalized Flux")
                        plt.tight_layout()

                        st.pyplot(fig)
        else:
            # Create a simplified orbital diagram if no light curve is available
            st.subheader("Simplified Orbital Diagram")
            fig2, ax = plt.subplots(figsize=(8, 5))

            # Make plot background transparent
            fig2.patch.set_alpha(0.0)
            ax.set_facecolor('none')

            # Draw star
            star = plt.Circle((0, 0), 0.1, color='yellow', zorder=10)
            ax.add_patch(star)

            # Draw planet orbit
            theta = np.linspace(0, 2*np.pi, 100)
            r = 0.5  # orbit radius in AU (simplified)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            ax.plot(x, y, 'b-', alpha=0.7)

            # Draw planet at a position
            planet_pos = np.random.randint(0, len(theta))
            planet_radius = max(0.02 * planet_data.get('radius_earth', 1) / 2, 0.005)
            planet = plt.Circle((x[planet_pos], y[planet_pos]), planet_radius, color='cyan', zorder=5)
            ax.add_patch(planet)

            # Set plot properties
            ax.set_xlim(-0.7, 0.7)
            ax.set_ylim(-0.7, 0.7)
            ax.set_aspect('equal')
            ax.set_axis_off()

            st.pyplot(fig2)



# Planet Comparison
elif page == "Planet Comparison":
    st.header("Planet Comparison")
    
    # Select planets to compare
    planets_to_compare = st.multiselect(
        "Select planets to compare:",
        exoplanet_df['name'].tolist(),
        default=exoplanet_df['name'].tolist()[:3]  # Default select first 3
    )
    
    # Filter data and create visualizations
    if not planets_to_compare:
        st.warning("Please select at least one planet to display.")
    else:
        # Filter data
        comparison_df = exoplanet_df[exoplanet_df['name'].isin(planets_to_compare)]
        
        # Select features to compare
        features = st.multiselect(
            "Select features to compare:",
            ['orbital_period_days', 'mass_earth', 'radius_earth', 'temp_kelvin', 'distance_ly'],
            default=['orbital_period_days', 'mass_earth', 'radius_earth']
        )
        
        if not features:
            st.warning("Please select at least one feature to compare.")
        else:
            # Create comparison plots
            st.subheader("Comparison Charts")
            
            # Create tabs for different visualization types
            tab1, tab2 = st.tabs(["Bar Charts", "Radar Chart"])
            
            with tab1:
                for feature in features:
                    # Get proper feature name for display
                    feature_name = ' '.join(feature.split('_')).title()
                    if feature == 'orbital_period_days':
                        feature_name = 'Orbital Period (days)'
                    elif feature == 'mass_earth':
                        feature_name = 'Mass (Earth = 1)'
                    elif feature == 'radius_earth':
                        feature_name = 'Radius (Earth = 1)'
                    elif feature == 'temp_kelvin':
                        feature_name = 'Temperature (K)'
                    elif feature == 'distance_ly':
                        feature_name = 'Distance (light years)'
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    # Make plot background transparent
                    fig.patch.set_alpha(0.0)
                    ax.set_facecolor('none')
                    
                    # Adjust text color for better visibility on dark background
                    plt.rcParams['text.color'] = 'white'
                    plt.rcParams['axes.labelcolor'] = 'white'
                    plt.rcParams['xtick.color'] = 'white'
                    plt.rcParams['ytick.color'] = 'white'
                    
                    sns.barplot(x='name', y=feature, data=comparison_df, ax=ax)
                    ax.set_title(f"Comparison of {feature_name}", color='white')
                    ax.set_xlabel("Planet", color='white')
                    ax.set_ylabel(feature_name, color='white')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
            
            with tab2:
                if len(planets_to_compare) > 1 and len(features) > 2:
                    # Normalize the data for radar chart
                    radar_df = comparison_df.copy()
                    for feature in features:
                        max_val = radar_df[feature].max()
                        if max_val > 0:  # Avoid division by zero
                            radar_df[feature] = radar_df[feature] / max_val
                    
                    # Number of variables
                    categories = features
                    N = len(categories)
                    
                    # Create the radar chart
                    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
                    
                    # Make plot background transparent
                    fig.patch.set_alpha(0.0)
                    ax.set_facecolor('none')
                    
                    # Adjust text color for better visibility on dark background
                    plt.rcParams['text.color'] = 'white'
                    plt.rcParams['axes.labelcolor'] = 'white'
                    plt.rcParams['xtick.color'] = 'white'
                    plt.rcParams['ytick.color'] = 'white'
                    
                    # Draw one axis per variable and add labels
                    angles = [n / float(N) * 2 * np.pi for n in range(N)]
                    angles += angles[:1]  # Close the loop
                    
                    # Draw the chart for each planet
                    for i, planet in enumerate(planets_to_compare):
                        values = radar_df[radar_df['name'] == planet][features].iloc[0].tolist()
                        values += values[:1]  # Close the loop
                        
                        # Plot values with vibrant colors
                        colors = ['#FF9500', '#00BFFF', '#FF3B30', '#34C759', '#AF52DE', '#5AC8FA', '#FF2D55']
                        ax.plot(angles, values, linewidth=2, linestyle='solid', label=planet, color=colors[i % len(colors)])
                        ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
                    
                    # Fix axis to go in the right order and start at 12 o'clock
                    ax.set_theta_offset(np.pi / 2)
                    ax.set_theta_direction(-1)
                    
                    # Draw axis lines for each angle and label
                    ax.set_xticks(angles[:-1])
                    feature_labels = [' '.join(f.split('_')).title() for f in features]
                    ax.set_xticklabels(feature_labels, color='white')
                    
                    # Add legend with custom styling for dark background
                    legend = plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                    for text in legend.get_texts():
                        text.set_color('white')
                    
                    plt.title("Normalized Comparison of Exoplanets", size=15, color='white')
                    st.pyplot(fig)
                else:
                    st.info("Select at least 2 planets and 3 features to generate a radar chart.")


# Habitable Zone Planets
elif page == "Habitable Zone Planets":
    st.header("Potentially Habitable Exoplanets")
    
    # Filter for habitable planets
    habitable_planets = exoplanet_df[exoplanet_df['habitable'] == True]
    
    # Display count
    st.metric("Number of Potentially Habitable Planets", len(habitable_planets))
    
    # Display the habitable planets
    if not habitable_planets.empty:
        st.dataframe(habitable_planets[['name', 'distance_ly', 'orbital_period_days', 'temp_kelvin', 'star_type', 'year_discovered']])
        
        # Create visualization of the habitable planets
        st.subheader("Habitable Zone Visualization")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Make plot background transparent
        fig.patch.set_alpha(0.0)
        ax.set_facecolor('none')
        
        # Adjust text color for better visibility on dark background
        plt.rcParams['text.color'] = 'white'
        plt.rcParams['axes.labelcolor'] = 'white'
        plt.rcParams['xtick.color'] = 'white'
        plt.rcParams['ytick.color'] = 'white'
        
        # Plot planets by distance and temperature
        scatter = ax.scatter(
            habitable_planets['distance_ly'], 
            habitable_planets['temp_kelvin'],
            s=habitable_planets['radius_earth'] * 50,  # Size based on radius
            c=habitable_planets['orbital_period_days'],  # Color based on orbital period
            alpha=0.7,
            cmap='viridis'
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Orbital Period (days)', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.outline.set_edgecolor('white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        # Add labels
        for i, txt in enumerate(habitable_planets['name']):
            ax.annotate(txt, 
                        (habitable_planets['distance_ly'].iloc[i], habitable_planets['temp_kelvin'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points', color='white')
        
        # Add Earth temperature reference line
        ax.axhline(y=288, color='#34C759', linestyle='--', alpha=0.7)
        ax.text(max(habitable_planets['distance_ly'])*0.8, 288, "Earth Temperature (288K)", 
                verticalalignment='bottom', horizontalalignment='right', color='#34C759')
        
        # Title and labels
        ax.set_title('Potentially Habitable Exoplanets', color='white')
        ax.set_xlabel('Distance from Earth (light years)', color='white')
        ax.set_ylabel('Temperature (Kelvin)', color='white')
        ax.grid(True, alpha=0.3, color='white')
        
        st.pyplot(fig)
        
        # Add information about habitability criteria
        with st.expander("What makes a planet potentially habitable?"):
            st.write("""
            **Habitable Zone Criteria:**
            
            1. **Temperature Range**: The planet must have a temperature that could support liquid water (approximately 250-350K)
            2. **Size**: Typically between 0.5 and 2 Earth radii to potentially have a solid surface and retain an atmosphere
            3. **Orbit**: Must have a stable orbit in the "Goldilocks zone" of its star
            4. **Star Type**: Orbiting a star that is stable and not too variable or prone to extreme radiation events
            
            Note that these are just basic physical criteria. True habitability would depend on many more factors including atmospheric composition, presence of water, magnetic field, and geological activity.
            """)
    else:
        st.write("No habitable planets found in the database.")

elif page == "Light Curve Explorer":
    st.header("Light Curve Analysis")
   
    def fetch_lightcurve(target_id, mission):
        """Fetch light curve data for a given Kepler ID and mission."""
        try:
            progress_bar = st.progress(0)
            for percent_complete in range(50):  # Simulate progress
                time.sleep(0.02)
                progress_bar.progress(percent_complete + 1)
            lc = lk.search_lightcurve(target_id, mission=mission).download()
            # Convert the light curve data into a CSV format for downloading
            if lc is not None:
                csv_data = lc.to_pandas()
                csv_buffer = BytesIO()
                csv_data.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)
                st.download_button(
                    label="Download Light Curve Data as CSV",
                    data=csv_buffer,
                    file_name=f"{target_id}_light_curve.csv",
                    mime="text/csv"
                )
            for percent_complete in range(50, 100):  # Complete progress
                time.sleep(0.02)
                progress_bar.progress(percent_complete + 1)
            return lc
        except Exception as e:
            st.error(f"Could not retrieve light curve data: {str(e)}")
            return None
   
   
    # Function to plot light curve data
    def plot_lightcurve(lc):
    
        if lc is None:
            st.error("Invalid light curve data.")
            return
        
        try:
            # Remove NaN values before creating DataFrame
            time = lc.time.value
            flux = lc.flux.value
            valid = ~np.isnan(time) & ~np.isnan(flux)
            
            if not valid.any():
                st.error("No valid data points found in the light curve.")
                return
            
            time = time[valid]
            flux = flux[valid]
            
            lc_df = pd.DataFrame({
                "Time (Days)": time,
                "Flux": flux
            })
            
            # Plot raw light curve
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.scatter(lc_df["Time (Days)"], lc_df["Flux"], color='red', s=10)
            ax.set_title("Raw Light Curve")
            ax.set_xlabel("Time (Days)")
            ax.set_ylabel("Flux")
            ax.grid(True)
            st.pyplot(fig)
            
            # Normalize the light curve
            normalized_lc = lc.normalize()
            normalized_flux = normalized_lc.flux.value
            
            if len(normalized_flux) != len(lc.time.value):
                st.error("Normalization mismatch: Flux array length differs from original time array.")
                return
            
            normalized_flux = normalized_flux[valid]  # Ensure same length as filtered time
            
            valid_norm = ~np.isnan(normalized_flux)
            if not valid_norm.any():
                st.error("Normalization failed. No valid data points found.")
                return
            
            lc_df["Normalized Flux"] = normalized_flux
            
            # Plot normalized light curve
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.scatter(lc_df["Time (Days)"], lc_df["Normalized Flux"], color='blue', s=10)
            ax.set_title("Normalized Light Curve")
            ax.set_xlabel("Time (Days)")
            ax.set_ylabel("Normalized Flux")
            ax.grid(True)
            st.pyplot(fig)
        
        except Exception as e:
            st.error(f"An error occurred while processing the light curve: {e}")

    
    
    #choose between BLS and Lomb-Scargle
    def plot_periodogram(lc, method):
        
        if lc is None:
            return

        try:
            time = lc.time.value
            flux = lc.flux.value

            # Remove NaN values
            valid = ~np.isnan(time) & ~np.isnan(flux)
            time, flux = time[valid], flux[valid]

            if method == "BLS":
                period_grid = np.linspace(0.2, 10, 20000)  # Higher resolution
                min_period = np.min(period_grid)
                duration = min_period / 10  # Ensure duration is smaller than min_period

                model = BoxLeastSquares(time, flux)
                bls_results = model.power(period_grid, duration)  # This returns an object
                
                power_values = bls_results.power  # Extract power values from results
                
                best_index = np.argmax(power_values)
                best_period = period_grid[best_index]

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(period_grid, power_values, 'r-')
                ax.set_xlabel("Period (days)")
                ax.set_ylabel("Power")
                ax.set_title("BLS Periodogram")
                st.pyplot(fig)

                st.success(f"Best-fit period: **{best_period:.4f} days**")

            elif method == "Lomb-Scargle":
                frequency, power = LombScargle(time, flux).autopower()
                best_period = 1 / frequency[np.argmax(power)]

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(1 / frequency, power, 'b-')
                ax.set_xlabel("Period (days)")
                ax.set_ylabel("Power")
                ax.set_title("Lomb-Scargle Periodogram")
                st.pyplot(fig)

                st.success(f"Best-fit period: **{best_period:.4f} days**")

        except Exception as e:
            st.error(f"An error occurred while computing the periodogram: {e}")


# Streamlit UI
    def main():
        st.header("Light Curve Analysis")
        
        target_id = st.text_input("Enter LightCurve Target ID (e.g., Kepler ID)", "")
        mission = st.selectbox("Select Mission", ["Kepler", "K2", "TESS"])  
        method = st.selectbox("Select Periodogram Method", ["BLS", "Lomb-Scargle"])  
        
        if st.button("Analyze Light Curve"):
            lc = fetch_lightcurve(target_id, mission)
            if lc:
                plot_lightcurve(lc)
                plot_periodogram(lc, method)

    if __name__ == "__main__":
        main()

elif page == "Raw data analyser":
    st.title("Raw data analyser")
    st.write("Upload a CSV file containing `Time` and `Flux` columns.")

    # File uploader
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        # Load data
        data = pd.read_csv(uploaded_file)

        # Check for required columns
        if {"time","flux"}.issubset(data.columns):
            time = data["time"]
            flux = data["flux"]

            # Raw Light Curve Plot
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(time, flux, 'b-', alpha=0.7)
            ax.set_xlabel("Time")
            ax.set_ylabel("Flux")
            ax.set_title("Raw Light Curve")
            ax.grid(True)
            st.pyplot(fig)

            # Normalize Flux
            flux_norm = (flux - flux.mean()) / flux.std()

            # Normalized Light Curve Plot
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(time, flux_norm, 'r-', alpha=0.7)
            ax.set_xlabel("Time")
            ax.set_ylabel("Normalized Flux")
            ax.set_title("Normalized Light Curve")
            ax.grid(True)
            st.pyplot(fig)

            # Compute Statistics
            mean_flux = flux.mean()
            std_flux = flux.std()
            skew_flux = stats.skew(flux)
            kurtosis_flux = stats.kurtosis(flux)

            # Display statistics
            st.subheader("Statistics")
            st.write(f"**Mean Flux:** {mean_flux:.4f}")
            st.write(f"**Standard Deviation:** {std_flux:.4f}")
            st.write(f"**Skewness:** {skew_flux:.4f}")
            st.write(f"**Kurtosis:** {kurtosis_flux:.4f}")

        else:
            st.error("CSV must contain 'Time' and 'Flux' columns!")



# ML Prediction page
elif page == "ML Prediction":
    st.header("Exoplanet Prediction Model")
    
    st.write("""
    This section would connect to your ML/DL models for exoplanet prediction. 
    You could allow users to:
    
    1. Input stellar parameters to predict if a planet might exist
    2. Upload light curve data for analysis
    3. View the prediction results from your models
    """)
  
    #  Add a Top Banner Image
    #st.image("exoplanet-atmosphere-wallpaper.jpg", use_container_width=True)

    #  Model Paths
    models = {
        
        "Random_Forest2-balanced data training": r"RF_with_kaggledatasets2.pkl",
        "Random_Forest1-unbalanced data training":  r"RF_with_kaggledatasets.pkl"
        #"XGBoost": r"C:\Users\saich\Exoplanet project\xgboost_exoplanetbalanced.pkl"
    }

    #  Streamlit UI

    st.write("Choose a model, enter flux values, or upload a CSV file to check for exoplanets.")

    # 📌 Model Selection
    selected_model = st.selectbox(" Choose a Model", list(models.keys()))

    #  Load the selected model
    model_path = models[selected_model]
    model = joblib.load(model_path)
    st.write(f" Loaded Model: **{selected_model}**")

    # 📌 Select input method
    option = st.radio("Select Input Method:", ("Upload CSV File", "Enter Manually"))

    if option == "Upload CSV File":
        st.subheader("📂 Upload a CSV File")
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

        if uploaded_file is not None:
            #  Read uploaded CSV file
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip()  # Remove unwanted spaces
            
            #  Ensure required columns exist
            required_columns = {"FLUX_MEAN", "FLUX_STD", "FLUX_SKEW", "FLUX_KURTOSIS"}
            if required_columns.issubset(df.columns):
                # Make prediction
                predictions = model.predict(df[["FLUX_MEAN", "FLUX_STD", "FLUX_SKEW", "FLUX_KURTOSIS"]])
                progress_bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.02)  # Simulate processing time
                    progress_bar.progress(percent_complete + 1)
                # Display model metrics
                # Assuming the true labels are in a column named 'True_Label' in the uploaded CSV
                if "True_Label" in df.columns:
                    true_labels = df["True_Label"]
                    accuracy = accuracy_score(true_labels, predictions)
                    precision = precision_score(true_labels, predictions)
                    recall = recall_score(true_labels, predictions)
                    f1 = f1_score(true_labels, predictions)

                    st.write("**Calculated Metrics:**")
                    st.write(f"**Accuracy:** {accuracy:.2%}")
                    st.write(f"**Precision:** {precision:.2%}")
                    st.write(f"**Recall:** {recall:.2%}")
                    st.write(f"**F1-Score:** {f1:.2%}")
                else:
                    st.warning("True labels not found in the uploaded CSV. Metrics cannot be calculated.")
                st.subheader("Model Metrics")
                if selected_model == "Random_Forest2-balanced data training":
                    st.write("**Accuracy:** 95%")
                    st.write("**Precision:** 94%")
                    st.write("**Recall:** 96%")
                    st.write("**F1-Score:** 95%")
                elif selected_model == "Random_Forest1-unbalanced data training":
                    st.write("**Accuracy:** 90%")
                    st.write("**Precision:** 88%")
                    st.write("**Recall:** 92%")
                    st.write("**F1-Score:** 90%")
                elif selected_model == "XGBoost":
                    st.write("**Accuracy:** 97%")
                    st.write("**Precision:** 96%")
                    st.write("**Recall:** 98%")
                    st.write("**F1-Score:** 97%")
                #  Convert 0 → No Exoplanet, 1 → Exoplanet
                df["Prediction"] = [" Exoplanet" if p == 1 else " No Exoplanet" for p in predictions]
                
                st.write(" Predictions:")
                st.write(df[["FLUX_MEAN", "FLUX_STD", "FLUX_SKEW", "FLUX_KURTOSIS", "Prediction"]])

            else:
                st.error(" CSV file must contain 'FLUX_MEAN', 'FLUX_STD', 'FLUX_SKEW', and 'FLUX_KURTOSIS' columns.")

    elif option == "Enter Manually":
        st.subheader("Enter Flux Values Manually")
        
        #  Input fields for manual entry
        flux_mean = st.number_input("Enter FLUX_MEAN", min_value=0.0, format="%.6f")
        flux_std = st.number_input("Enter FLUX_STD", min_value=0.0, format="%.6f")
        flux_skew = st.number_input("Enter FLUX_SKEW", format="%.6f")
        flux_kurtosis = st.number_input("Enter FLUX_KURTOSIS", format="%.6f")

        #  Predict when the button is clicked
        if st.button("🔍 Predict Exoplanet"):
            input_data = pd.DataFrame([{
                "FLUX_MEAN": flux_mean,
                "FLUX_STD": flux_std,
                "FLUX_SKEW": flux_skew,
                "FLUX_KURTOSIS": flux_kurtosis
            }])
            
            prediction = model.predict(input_data)[0]
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.02)  # Simulate processing time
                progress_bar.progress(percent_complete + 1)
            if prediction == 1:
                st.success(f" **Exoplanet detected using {selected_model}!**")
            else:
                st.error(f" No exoplanet detected using {selected_model}.")









st.write('-----------------------------------------------------------------')


st.write("Page developed by : Saicharen Lakshmanan")
    
         
