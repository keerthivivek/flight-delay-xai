import streamlit as st
import duckdb
import pandas as pd
import pydeck as pdk
import plotly.express as px
import os
import warnings
import datetime
from pandas.errors import SettingWithCopyWarning
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
st.set_page_config(page_title="Flight Route Visualizer", layout="wide")

FLIGHT_DATA_PATH = "all_data.parquet"
GEO_DATA_PATH = "flight_geo/L_AIRPORT_ID_with_Coordinates.csv"
AIRLINE_DATA_PATH = "flight_geo/L_AIRLINE_ID.csv"


# Open an in-process connection. Using an in-memory DB for speed.
@st.cache_resource
def get_data_conn():
    conn = duckdb.connect(database=':memory:')
    # Register the parquet and csv as views (lazy; keeps metadata)
    if os.path.exists(FLIGHT_DATA_PATH):
        conn.execute(f"CREATE VIEW flights AS SELECT * FROM read_parquet('{FLIGHT_DATA_PATH}')")
    else:
        st.error(f"Missing flight data file at {FLIGHT_DATA_PATH}")
        st.stop()

    if os.path.exists(GEO_DATA_PATH):
        conn.execute(f"CREATE VIEW geo AS SELECT * FROM read_csv_auto('{GEO_DATA_PATH}')")
    else:
        st.error(f"Missing location data file at {GEO_DATA_PATH}")
        st.stop()

    if os.path.exists(AIRLINE_DATA_PATH):
        conn.execute(f"CREATE VIEW airlines AS SELECT * FROM read_csv_auto('{AIRLINE_DATA_PATH}')")
    else:
        # not fatal
        conn.execute("CREATE VIEW airlines AS SELECT NULL AS Code, NULL AS Description LIMIT 0")
    return conn


conn = get_data_conn()

# small helper SQL expressions reused
# make_date and dow (Monday=1..Sunday=7)
DOW_SQL = """((CAST(strftime('%w', make_date(Year, Month, DayofMonth)) AS INTEGER) + 6) % 7) + 1"""

delay_cols = ['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']


@st.cache_data(show_spinner="Loading airline data...")
def load_airline_data():
    df = conn.execute("SELECT Code, Description FROM airlines").fetchdf()
    if "Code" in df.columns:
        df["Code"] = df["Code"].astype(str)
        return dict(zip(df["Code"], df["Description"]))
    return {}


@st.cache_data
def load_geo_data():
    df = conn.execute("SELECT Code, \"Airport Name\" AS airport_name, Latitude, Longitude FROM geo").fetchdf()
    # keep Code types consistent (airport ids are numeric sometimes)
    df["Code"] = df["Code"].astype(str)
    return df


@st.cache_data(show_spinner="Loading min/max dates...")
def get_min_max_dates():
    q = f"""
    SELECT 
      MIN(Year) as min_y, MIN(Month) as min_m, MIN(DayofMonth) as min_d,
      MAX(Year) as max_y, MAX(Month) as max_m, MAX(DayofMonth) as max_d
    FROM flights
    """
    r = conn.execute(q).fetchone()
    try:
        min_date = datetime.date(int(r[0]), int(r[1]), int(r[2]))
        max_date = datetime.date(int(r[3]), int(r[4]), int(r[5]))
    except Exception:
        # fallback
        min_date = datetime.date(2024, 1, 1)
        max_date = datetime.date(2024, 12, 31)
    return min_date, max_date


@st.cache_data(show_spinner="Finding airports for date...")
def get_airports_for_date(selected_date):
    q = f"""
    SELECT DISTINCT CAST(OriginAirportID AS VARCHAR) AS Code
    FROM flights
    WHERE Year = {selected_date.year}
      AND Month = {selected_date.month}
      AND DayofMonth = {selected_date.day}
    """
    origin_ids = [str(x[0]) for x in conn.execute(q).fetchall()]
    if not origin_ids:
        return {}
    df_geo = load_geo_data()
    dff = df_geo[df_geo["Code"].isin(origin_ids)].copy()
    dff["display"] = dff["Code"].astype(str) + " - " + dff["airport_name"].fillna("Unknown")
    return dict(zip(dff["Code"], dff["display"]))


@st.cache_data(show_spinner="Getting destinations for origin...")
def get_destinations_for_origin(origin_id, selected_date):
    q = f"""
    SELECT DISTINCT CAST(DestAirportID AS VARCHAR) AS Code
    FROM flights
    WHERE Year = {selected_date.year}
      AND Month = {selected_date.month}
      AND DayofMonth = {selected_date.day}
      AND CAST(OriginAirportID AS VARCHAR) = '{origin_id}'
    """
    dest_ids = [str(x[0]) for x in conn.execute(q).fetchall()]
    if not dest_ids:
        return {}
    df_geo = load_geo_data()
    dfd = df_geo[df_geo["Code"].isin(dest_ids)].copy()
    dfd["display"] = dfd["Code"].astype(str) + " - " + dfd["airport_name"].fillna("Unknown")
    return dict(zip(dfd["Code"], dfd["display"]))


@st.cache_data(show_spinner="Getting departure times...")
def get_available_dep_times(origin_id, dest_ids, selected_date):
    where_origin = f"AND CAST(OriginAirportID AS VARCHAR) = '{origin_id}'"
    where_date = f"Year = {selected_date.year} AND Month = {selected_date.month} AND DayofMonth = {selected_date.day}"
    dest_clause = ""
    if dest_ids and dest_ids != ["ALL"]:
        dest_list = ",".join([f"'{d}'" for d in dest_ids])
        dest_clause = f"AND CAST(DestAirportID AS VARCHAR) IN ({dest_list})"
    q = f"""
    SELECT DISTINCT CRSDepTime FROM flights
    WHERE {where_date} {where_origin} {dest_clause}
      AND CRSDepTime IS NOT NULL
    """
    rows = conn.execute(q).fetchdf()
    times = sorted([int(x) for x in rows["CRSDepTime"].unique() if pd.notna(x)])
    return times


def format_time(time_int):
    if pd.isna(time_int) or time_int is None:
        return "N/A"
    try:
        t = int(time_int)
        if t < 0 or t > 2400:
            return "N/A"
        s = str(t).zfill(4)
        return f"{s[:2]}:{s[2:]}"
    except Exception:
        return "N/A"


def delay_ratio_to_color(ratio, alpha=200):
    # This color function produces the Green (On-Time) -> Yellow (Mixed) -> Red (Delayed) gradient
    ratio = max(0, min(1, ratio))
    if ratio < 0.5:
        # Green to Yellow
        r = int(510 * ratio)
        g = 255
    else:
        # Yellow to Red
        r = 255
        g = int(510 * (1 - ratio))
    # Return as list [R, G, B, A]
    return [r, g, 0, alpha]


@st.cache_data(show_spinner="Fetching routes...")
def get_routes_for_selection(origin_id, dest_ids, selected_date, delay_filter, time_filter):
    # Build SQL filter clauses
    where_clauses = [
        f"Year = {selected_date.year}",
        f"Month = {selected_date.month}",
        f"DayofMonth = {selected_date.day}",
        f"CAST(OriginAirportID AS VARCHAR) = '{origin_id}'"
    ]
    if dest_ids and dest_ids != ["ALL"]:
        dest_list = ",".join([f"'{d}'" for d in dest_ids])
        where_clauses.append(f"CAST(DestAirportID AS VARCHAR) IN ({dest_list})")
    if time_filter != "ALL":
        where_clauses.append(f"CRSDepTime = {int(time_filter)}")
    # delay filter
    if delay_filter == "Delayed":
        where_clauses.append("(ArrDel15 = 1 OR DepDel15 = 1)")
    elif delay_filter == "On-Time":
        where_clauses.append("(COALESCE(ArrDel15,0) != 1 AND COALESCE(DepDel15,0) != 1)")

    where_sql = " AND ".join(where_clauses)

    # Aggregation query to produce arcs
    q_agg = f"""
    SELECT
      CAST(OriginAirportID AS VARCHAR) as origin,
      CAST(DestAirportID AS VARCHAR) as dest,
      COUNT(*) AS total_flights,
      SUM(CASE WHEN COALESCE(ArrDel15,0)=1 OR COALESCE(DepDel15,0)=1 THEN 1 ELSE 0 END) AS delayed_count
    FROM flights
    WHERE {where_sql}
    GROUP BY origin, dest
    """
    agg = conn.execute(q_agg).fetchdf()
    if agg.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # join geo coordinates
    geo = load_geo_data()
    geo_src = geo.rename(
        columns={"Code": "origin", "airport_name": "orig_name", "Latitude": "src_lat", "Longitude": "src_lon"})
    geo_dst = geo.rename(
        columns={"Code": "dest", "airport_name": "dest_name", "Latitude": "dest_lat", "Longitude": "dest_lon"})

    # join using pandas (small dataframe), it's fine
    merged = agg.merge(geo_src, on="origin", how="left").merge(geo_dst, on="dest", how="left")
    merged["delay_ratio"] = merged["delayed_count"] / merged["total_flights"]
    merged["pct_delayed"] = (merged["delay_ratio"] * 100).round(1).astype(str) + "%"
    merged["color"] = merged["delay_ratio"].apply(delay_ratio_to_color)

    # DataFrame for ArcLayer
    arcs = merged[[
        "src_lon", "src_lat", "dest_lon", "dest_lat",
        "orig_name", "dest_name", "total_flights", "delayed_count", "pct_delayed", "color"
    ]].copy()

    # DataFrames for ScatterplotLayer (Points)
    origins = merged[["orig_name", "src_lat", "src_lon", "delay_ratio"]].drop_duplicates().rename(
        columns={"orig_name": "airport", "src_lat": "lat", "src_lon": "lon"}
    )
    origins["type"] = "Origin"
    # FIX: correct 'dest_lon' rename (was 'lon' earlier)
    destinations = merged[["dest_name", "dest_lat", "dest_lon", "delay_ratio"]].drop_duplicates().rename(
        columns={"dest_name": "airport", "dest_lat": "lat", "dest_lon": "lon"}
    )
    destinations["type"] = "Destination"
    points = pd.concat([origins, destinations], ignore_index=True)
    points["color"] = points["delay_ratio"].apply(delay_ratio_to_color)
    points["status_label"] = points["delay_ratio"].apply(
        lambda x: "On-Time"
        if x == 0
        else ("Delayed" if x == 1 else "Mixed (%.0f%% delayed)" % (x * 100))
    )
    points["total_flights"] = None
    points["delayed_count"] = None
    points["pct_delayed"] = None

    # Additionally fetch the underlying rows for display if needed (small selection)
    q_rows = f"""
    SELECT
      Flight_Number_Reporting_Airline as fl_num,
      CAST(OriginAirportID AS VARCHAR) as origin,
      CAST(DestAirportID AS VARCHAR) as dest,
      CRSDepTime as crs_dep_time_raw,
      COALESCE(ArrDel15,0) as ArrDel15,
      COALESCE(DepDel15,0) as DepDel15,
      COALESCE(ArrDelayMinutes,0) as ArrDelayMinutes,
      COALESCE(DepDelayMinutes,0) as DepDelayMinutes,
      COALESCE(CarrierDelay,0) as CarrierDelay,
      COALESCE(WeatherDelay,0) as WeatherDelay,
      COALESCE(NASDelay,0) as NASDelay,
      COALESCE(SecurityDelay,0) as SecurityDelay,
      COALESCE(LateAircraftDelay,0) as LateAircraftDelay
    FROM flights
    WHERE {where_sql}
    """
    rows_df = conn.execute(q_rows).fetchdf()
    if not rows_df.empty:
        rows_df["is_delay"] = ((rows_df["ArrDel15"] == 1) | (rows_df["DepDel15"] == 1)).astype(int)
        rows_df["crs_dep_time"] = rows_df["crs_dep_time_raw"].apply(format_time)
        rows_df["crs_dep_time_sort"] = rows_df["crs_dep_time_raw"].fillna(0).astype(int)

        # merge coordinates & names into rows_df for potential table usage
        rows_df = rows_df.merge(geo.rename(
            columns={"Code": "origin", "airport_name": "orig_name", "Latitude": "src_lat", "Longitude": "src_lon"}),
            on="origin", how="left")
        rows_df = rows_df.merge(geo.rename(
            columns={"Code": "dest", "airport_name": "dest_name", "Latitude": "dest_lat", "Longitude": "dest_lon"}),
            on="dest", how="left")

        rows_df = rows_df.sort_values("crs_dep_time_sort").reset_index(drop=True)

    return rows_df, arcs, points


@st.cache_data(show_spinner="Finding available airlines for route...")
def get_airlines_for_route(origin_id, dest_id):
    q = f"""
    SELECT DISTINCT CAST(DOT_ID_Reporting_Airline AS VARCHAR) as code 
    FROM flights 
    WHERE CAST(OriginAirportID AS VARCHAR) = '{origin_id}' 
      AND CAST(DestAirportID AS VARCHAR) = '{dest_id}'
      AND DOT_ID_Reporting_Airline IS NOT NULL
    """
    df_airlines_available = conn.execute(q).fetchdf()
    return sorted([str(x) for x in df_airlines_available['code'].unique() if pd.notna(x)])


@st.cache_data(show_spinner="Preparing training data...")
def prepare_training_data_for_route(origin_id, dest_id, selected_airlines):
    # Build query that returns Year, DayofMonth, DayOfWeek (1..7), Month, DepHour, TotalDelay
    airline_filter = ""
    if selected_airlines:
        sel_airlines_csv = ",".join([f"'{a}'" for a in selected_airlines])
        airline_filter = f"AND CAST(DOT_ID_Reporting_Airline AS VARCHAR) IN ({sel_airlines_csv})"

    q_train = f"""
    SELECT
      Year,
      DayofMonth,
      {DOW_SQL} AS DayOfWeek,
      Month,
      CAST(COALESCE(CRSDepTime,0) AS INTEGER) / 100 AS DepHour,
      (COALESCE(ArrDelayMinutes,0) + COALESCE(DepDelayMinutes,0)) AS TotalDelay,
      CRSDepTime
    FROM flights
    WHERE CAST(OriginAirportID AS VARCHAR) = '{origin_id}'
      AND CAST(DestAirportID AS VARCHAR) = '{dest_id}'
      AND DayofMonth IS NOT NULL
      AND Month IS NOT NULL
      AND CRSDepTime IS NOT NULL
      {airline_filter}
    """
    df_train = conn.execute(q_train).fetchdf()
    # cast int columns correctly
    if not df_train.empty:
        df_train["Year"] = df_train["Year"].astype(int)
        df_train["DayofMonth"] = df_train["DayofMonth"].astype(int)
        df_train["DayOfWeek"] = df_train["DayOfWeek"].astype(int)
        df_train["Month"] = df_train["Month"].astype(int)
        df_train["DepHour"] = df_train["DepHour"].astype(int)
    return df_train


@st.cache_data(show_spinner="Training prediction model...")
def train_knn_model_from_df(df_train, n_neighbors=5):
    if df_train.empty:
        return None
    X = df_train[["DayofMonth", "DayOfWeek", "Month", "DepHour"]].values.astype(float)
    y = df_train["TotalDelay"].values.astype(float)
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X, y)
    return model


def predict_delay_with_neighbors(model, day_of_month, day_of_week, month, dep_hour, df_train):
    if model is None or df_train.empty:
        return 0, pd.DataFrame()
    features = np.array([[day_of_month, day_of_week, month, dep_hour]], dtype=float)
    distances, indices = model.kneighbors(features)
    prediction = model.predict(features)[0]
    neighbors_data = df_train.iloc[indices[0]].copy()
    return max(0, prediction), neighbors_data


# --- UI starts ---
st.title("✈️ RouteLens")
st.subheader("Data-driven Flight Delay Visualization and Prediction")

airline_lookup = load_airline_data()
df_geo = load_geo_data()
min_date, max_date = get_min_max_dates()

tab1, tab2, tab3 = st.tabs(["📊 Past Flight Activity", "📈 Historical Averages", "🔮 Analog Forecaster"])

with tab1:
    st.markdown("### Filters")
    filter_cols = st.columns([1.5, 1.5, 1.5, 1.5, 1.5, 0.5])

    with filter_cols[0]:
        selected_date = st.date_input(
            "Date",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
            format="MM/DD/YYYY",
        )

    airport_id_to_display = get_airports_for_date(selected_date)
    if not airport_id_to_display:
        st.warning(f"No flights available on {selected_date.strftime('%Y-%m-%d')}")
        st.stop()

    origin_display_names = list(airport_id_to_display.values())
    with filter_cols[1]:
        selected_display_origin = st.selectbox("Origin Airport", origin_display_names, label_visibility="visible")

    selected_origin_id = [k for k, v in airport_id_to_display.items() if v == selected_display_origin][0]

    valid_dest_options = get_destinations_for_origin(selected_origin_id, selected_date)
    destination_display_names = list(valid_dest_options.values())

    with filter_cols[2]:
        selected_display_dests = st.multiselect(
            "Destination(s)",
            options=destination_display_names,
            default=None,
            label_visibility="visible"
        )

    if selected_display_dests:
        selected_dest_ids = [k for k, v in valid_dest_options.items() if v == selected_display_dests]
    else:
        selected_dest_ids = ["ALL"]

    available_times = get_available_dep_times(selected_origin_id, selected_dest_ids, selected_date)
    time_options = ["ALL"] + [format_time(t) for t in available_times]
    time_values = ["ALL"] + [str(t) for t in available_times]

    with filter_cols[3]:
        time_display = st.selectbox(
            "Scheduled Departure Time",
            options=range(len(time_options)),
            format_func=lambda i: time_options[i],
            label_visibility="visible"
        )

    time_filter = time_values[time_display]

    with filter_cols[4]:
        delay_filter = st.selectbox("Flight Status", ["ALL", "On-Time", "Delayed"], label_visibility="visible")

    df_raw, arcs, points = get_routes_for_selection(
        selected_origin_id, selected_dest_ids, selected_date, delay_filter, time_filter
    )

    # Check if a specific filter (other than ALL) was applied which might filter out all results
    filter_applied = delay_filter != "ALL" or time_filter != "ALL" or selected_dest_ids != ["ALL"]

    if df_raw is None or (isinstance(df_raw, pd.DataFrame) and df_raw.empty):
        st.warning("No data for this selection.")
        st.stop()

    st.markdown("---")
    st.info(
        f"Showing **{len(df_raw):,}** flights from **{selected_display_origin}** on **{selected_date.strftime('%Y-%m-%d')}**"
    )
    st.markdown("🟢 Green = On-Time   🟡 Mixed   🔴 Delayed")

    # --- Map setup ---
    tooltip = {
        "html": """
            <div style="font-size:13px; line-height:1.2;">
                <b>Airport:</b> {airport} ({type})<br/>
                <b>Total Flights:</b> {total_flights}<br/>
                <b>Delayed Flights:</b> {delayed_count}<br/>
                <b>Status:</b> {status_label}
            </div>
        """,
        "style": {
            "color": "black",
            "backgroundColor": "white",
            "fontSize": "13px",
            "padding": "10px",
            "borderRadius": "6px",
            "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)",
        },
    }

    arc_layer = pdk.Layer(
        "ArcLayer",
        data=arcs,
        get_source_position=["src_lon", "src_lat"],
        get_target_position=["dest_lon", "dest_lat"],
        get_source_color="color",
        get_target_color="color",
        get_width=5,
        pickable=True,
        auto_highlight=True,
    )

    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=points,
        get_position=["lon", "lat"],
        get_radius=25000,
        get_fill_color="color",
        get_line_color=[0, 0, 0, 150],
        get_line_width=150,
        pickable=True,
        auto_highlight=True,
    )

    center_lat = df_raw["src_lat"].mean() if not df_raw["src_lat"].empty else 40.7
    center_lon = df_raw["src_lon"].mean() if not df_raw["src_lon"].empty else -100.0

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=5,
        pitch=45,
    )

    r = pdk.Deck(
        layers=[arc_layer, scatter_layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        height=700,
    )

    st.pydeck_chart(r)
    st.markdown("---")

    st.subheader("📊 Flight Statistics")
    col1, col2, col3, col4 = st.columns(4)
    delayed_count = int(df_raw["is_delay"].sum())
    ontime_count = len(df_raw) - delayed_count
    delayed_pct = (delayed_count / len(df_raw) * 100) if len(df_raw) > 0 else 0

    col1.metric("Total Flights", len(df_raw))
    col2.metric("On-Time", ontime_count)
    col3.metric("Delayed", delayed_count)
    col4.metric("Delayed %", f"{delayed_pct:.1f}%")

    # --- Heatmap View ---
    st.subheader("🗓️ Route Delay Heatmap")
    st.markdown(f"**Origin Airport:** {selected_display_origin}")

    heatmap_data = df_raw.groupby(['dest_name', 'crs_dep_time']).agg({'is_delay': ['sum', 'count']}).reset_index()
    heatmap_data.columns = ['Destination', 'Time', 'Delayed', 'Total']
    heatmap_data['Delay_Ratio'] = heatmap_data['Delayed'] / heatmap_data['Total']
    pivot_data = heatmap_data.pivot(index='Destination', columns='Time', values='Delay_Ratio')

    num_destinations = len(pivot_data.index)
    cell_height = 55 if num_destinations > 4 else 80
    chart_height = max(500, num_destinations * cell_height + 250)

    fig = px.imshow(
        pivot_data,
        labels=dict(
            x="Scheduled Departure Time",
            y="Destination Airport",
            color="Delay Ratio"
        ),
        x=pivot_data.columns,
        y=pivot_data.index,
        color_continuous_scale=['green', 'yellow', 'red'],
        zmin=0,
        zmax=1,
        aspect="auto"
    )

    fig.update_traces(
        hovertemplate="<b>Destination:</b> %{y}<br>"
                      "<b>Time:</b> %{x}<br>"
                      "<b>Delay Ratio:</b> %{z:.1%}<extra></extra>"
    )

    fig.update_layout(
        height=chart_height,
        xaxis_title="Scheduled Departure Time",
        yaxis_title="Destination Airport",
        coloraxis_colorbar=dict(
            title="Delay<br>Ratio",
            tickvals=[0, 0.5, 1],
            ticktext=['On-Time', 'Mixed', 'Delayed'],
            title_font=dict(size=18),
            tickfont=dict(size=16)
        ),
        xaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=15),
            tickangle=45,
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=15)
        ),
        margin=dict(l=180, r=100, t=80, b=100),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=16)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    if st.checkbox("Show raw data table (First 500 rows)", value=False):
        display_cols = ["fl_num", "orig_name", "dest_name", "crs_dep_time", "is_delay"] + delay_cols
        df_display = df_raw[display_cols].copy()
        df_display["is_delay"] = df_display["is_delay"].map({0: "On-Time", 1: "Delayed"})
        st.dataframe(df_display.rename(columns={
            "fl_num": "Flight #", "orig_name": "Origin Airport", "dest_name": "Destination Airport",
            "crs_dep_time": "Scheduled Dep Time", "is_delay": "Status"
        }).head(500), use_container_width=True)

    # --- Delay Breakdown Chart ---
    delayed_flights = df_raw[(df_raw["ArrDel15"] == 1) | (df_raw["DepDel15"] == 1)].copy()

    if not delayed_flights.empty:
        delay_averages = {}
        for col in delay_cols:
            delay_averages[col] = delayed_flights[col].mean()

        # Filter out 0 averages and sort descending
        delay_averages = {k: v for k, v in delay_averages.items() if v > 0}

        if delay_averages:
            st.markdown("---")
            st.markdown("### Average Breakdown of Delays (in Minutes)")
            st.info("Based on the **delayed** flights shown above.")

            chart_df = pd.DataFrame({
                'Delay Type': [name.replace('Delay', '').replace('Aircraft', ' Late Aircraft').replace('NAS', 'National Aviation System')
                               for name in delay_averages.keys()],
                'Average Minutes': list(delay_averages.values())
            }).sort_values('Average Minutes', ascending=False)

            fig_delay = px.bar(
                chart_df,
                y='Delay Type',
                x='Average Minutes',
                orientation='h',
                text='Average Minutes',
                height=400,
                title="Average Delay by Category"
            )

            fig_delay.update_traces(
                texttemplate='%{text:.1f} min',
                textposition='outside',
                marker=dict(
                    color=chart_df['Average Minutes'],
                    colorscale='Viridis',
                    line=dict(color='white', width=1),
                )
            )

            fig_delay.update_layout(
                showlegend=False,
                xaxis_title="Average Delay (Minutes)",
                yaxis_title="",
                margin=dict(l=150, r=100, t=80, b=60),
                plot_bgcolor='rgba(240,240,240,0.5)',
                paper_bgcolor='white',
                font=dict(size=16),
                title_font_size=18,
                xaxis=dict(gridcolor='lightgray', showgrid=True),
                hovermode='closest'
            )

            st.plotly_chart(fig_delay, use_container_width=True)
        else:
            st.info("No individual delay categories found for the selected flights.")

with tab2:
    # --- Added Chart: Average Delay Across All Airlines ---
    st.markdown("### ✈️ Average Delay by Airline")
    q_airline_avg = """
                    SELECT 
                        CAST(DOT_ID_Reporting_Airline AS VARCHAR) as AirlineCode,
                        AVG(COALESCE(ArrDelayMinutes, 0) + COALESCE(DepDelayMinutes, 0)) AS AvgTotalDelay,
                        COUNT(*) AS FlightCount
                    FROM flights
                    WHERE DOT_ID_Reporting_Airline IS NOT NULL
                    GROUP BY AirlineCode
                    HAVING FlightCount > 100
                    ORDER BY AvgTotalDelay DESC
                    """
    airline_avg_df = conn.execute(q_airline_avg).fetchdf()

    if not airline_avg_df.empty:
        airline_avg_df["AirlineName"] = airline_avg_df["AirlineCode"].apply(
            lambda code: f"{code} - {airline_lookup.get(code, 'Unknown')}"
        )

        fig_airline = px.bar(
            airline_avg_df,
            x="AvgTotalDelay",
            y="AirlineName",
            orientation='h',
            title="Overall Average Total Delay (Arrival + Departure) by Airline",
            labels={"AvgTotalDelay": "Avg Total Delay (Minutes)", "AirlineName": "Airline"},
            height=600,
            color="AvgTotalDelay",
            color_continuous_scale="RdYlGn_r"
        )
        fig_airline.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title="Average Total Delay (Minutes)",
            yaxis_title="",
            font=dict(size=14),
            title_font_size=18,
            margin=dict(l=200, r=20, t=60, b=40)
        )
        st.plotly_chart(fig_airline, use_container_width=True)

    st.markdown("---")
    st.markdown("### Historical Average Delays by Month")
    q_monthly = """
                SELECT Month, AVG (COALESCE (ArrDelayMinutes, 0) + COALESCE (DepDelayMinutes, 0)) AS AvgTotalDelay, COUNT (*) AS FlightCount
                FROM flights
                GROUP BY Month
                ORDER BY Month
                """
    monthly_avg = conn.execute(q_monthly).fetchdf()
    if not monthly_avg.empty:
        monthly_avg["MonthName"] = monthly_avg["Month"].apply(lambda m: datetime.date(2024, int(m), 1).strftime('%B'))
        fig_monthly = px.bar(
            monthly_avg,
            x="MonthName",
            y="AvgTotalDelay",
            title="Average Total Delay by Month (Across All Airlines)",
            labels={"AvgTotalDelay": "Avg Total Delay (Minutes)", "MonthName": "Month"},
            height=300,
            color="AvgTotalDelay",
            color_continuous_scale="RdYlGn_r"
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

    st.markdown("---")
    st.markdown("### Filters for Monthly Trend by Airline")
    filter_cols = st.columns([1.5, 1.5, 1.5])
    with filter_cols[0]:
        selected_month = st.selectbox(
            "Month",
            options=range(1, 13),
            format_func=lambda m: datetime.date(2024, m, 1).strftime('%B'),
            label_visibility="visible"
        )
    # available airlines
    df_airlines_available = conn.execute(
        "SELECT DISTINCT CAST(DOT_ID_Reporting_Airline AS VARCHAR) as code FROM flights WHERE DOT_ID_Reporting_Airline IS NOT NULL").fetchdf()
    available_airlines = sorted([str(x) for x in df_airlines_available['code'].unique() if pd.notna(x)])
    airline_display_dict = {}
    for code in available_airlines:
        display_name = f"{code} - {airline_lookup.get(code, 'Unknown')}"
        airline_display_dict[display_name] = code
    airline_display_names = list(airline_display_dict.keys())
    default_display_names = airline_display_names[:5] if len(airline_display_names) > 0 else []

    with filter_cols[1]:
        selected_display_airlines = st.multiselect(
            "Airlines",
            options=airline_display_names,
            default=default_display_names,
            label_visibility="visible"
        )
    selected_airlines = [airline_display_dict[d] for d in
                         selected_display_airlines] if selected_display_airlines else []

    with filter_cols[2]:
        delay_threshold = st.slider(
            "Delay Threshold (min)",
            min_value=0,
            max_value=120,
            value=30,
            step=5,
            label_visibility="visible"
        )

    if not selected_airlines:
        st.warning("Please select at least one airline.")
        st.stop()

    sel_airlines_csv = ",".join([f"'{a}'" for a in selected_airlines])
    q_month_sel = f"""
    SELECT DayofMonth, AVG(COALESCE(ArrDelayMinutes,0)+COALESCE(DepDelayMinutes,0)) AS TotalDelay, COUNT(*) AS FlightCount
    FROM flights
    WHERE Month = {selected_month} AND CAST(DOT_ID_Reporting_Airline AS VARCHAR) IN ({sel_airlines_csv})
    GROUP BY DayofMonth
    ORDER BY DayofMonth
    """
    df_month = conn.execute(q_month_sel).fetchdf()
    if df_month.empty:
        st.warning("No data available for selected filters.")
        st.stop()

    df_month = df_month.rename(columns={"DayofMonth": "Day", "TotalDelay": "AvgDelay"})
    df_month["Color"] = df_month["AvgDelay"].apply(lambda x: "green" if x <= delay_threshold else "yellow")
    fig = px.bar(df_month, x="Day", y="AvgDelay", color="Color",
                 color_discrete_map={"green": "#00CC96", "yellow": "#FFD92F"}, height=500)
    fig.update_layout(xaxis=dict(tickmode="linear", tick0=1, dtick=1))
    fig.add_hline(y=delay_threshold, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### 🔮 Analog Forecaster - Delay Prediction")
    st.markdown(
        "Predicts flight delays based on historical analogs (same day-of-week and day-of-month within the same month).")
    st.markdown("---")

    # --- ORIGIN + DESTINATION dropdowns with names ---
    df_geo = load_geo_data()
    airline_lookup = load_airline_data()

    q_orig = "SELECT DISTINCT CAST(OriginAirportID AS VARCHAR) as code FROM flights WHERE OriginAirportID IS NOT NULL"
    df_orig = conn.execute(q_orig).fetchdf()
    origins_all = sorted(df_orig["code"].astype(str).unique())
    origin_map = {c: f"{c} - {df_geo.loc[df_geo['Code'] == c, 'airport_name'].iloc[0]}"
    if c in df_geo["Code"].values else c for c in origins_all}
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        sel_origin_disp = st.selectbox("Origin Airport", list(origin_map.values()), key="forecast_origin")
    forecast_origin = [k for k, v in origin_map.items() if v == sel_origin_disp][0]

    q_dests = f"""
        SELECT DISTINCT CAST(DestAirportID AS VARCHAR) as code 
        FROM flights 
        WHERE CAST(OriginAirportID AS VARCHAR) = '{forecast_origin}'
          AND DestAirportID IS NOT NULL
    """
    df_dests = conn.execute(q_dests).fetchdf()
    dests_all = sorted(df_dests["code"].astype(str).unique())
    dest_map = {c: f"{c} - {df_geo.loc[df_geo['Code'] == c, 'airport_name'].iloc[0]}"
    if c in df_geo["Code"].values else c for c in dests_all}

    with col2:
        sel_dest_disp = st.selectbox("Destination Airport", list(dest_map.values()), key="forecast_dest")
        forecast_dest = [k for k, v in dest_map.items() if v == sel_dest_disp][0]

    # --- Airlines selection (single select) ---
    avail_air_q = f"""
        SELECT DISTINCT CAST(DOT_ID_Reporting_Airline AS VARCHAR) as code
        FROM flights
        WHERE CAST(OriginAirportID AS VARCHAR) = '{forecast_origin}'
          AND CAST(DestAirportID AS VARCHAR) = '{forecast_dest}'
        ORDER BY code
    """
    airlines_avail = conn.execute(avail_air_q).fetchdf()["code"].astype(str).unique().tolist()
    air_display = {f"{a} - {airline_lookup.get(a, 'Unknown')}": a for a in airlines_avail}

    with col3:
        sel_display_airline = st.selectbox("Airline",
                                           list(air_display.keys()),
                                           key="forecast_airline")
    forecast_airline = air_display[sel_display_airline]

    with col4:
        forecast_date = st.date_input(
            "Date",
            value=datetime.date.today() + datetime.timedelta(days=7),
            format="MM/DD/YYYY",
            key="forecast_date"
        )

    forecast_month = forecast_date.month
    forecast_day = forecast_date.day
    forecast_dow = forecast_date.isoweekday()

    # Get available departure hours (in 1-hour increments) for this airline, route, month, and day-of-week
    q_avail_hours = f"""
        SELECT DISTINCT CAST(COALESCE(CRSDepTime,0) AS INTEGER) / 100 AS DepHour
        FROM flights
        WHERE CAST(OriginAirportID AS VARCHAR) = '{forecast_origin}'
          AND CAST(DestAirportID AS VARCHAR) = '{forecast_dest}'
          AND CAST(DOT_ID_Reporting_Airline AS VARCHAR) = '{forecast_airline}'
          AND Month = {forecast_month}
          AND {DOW_SQL} = {forecast_dow}
          AND CRSDepTime IS NOT NULL
        ORDER BY DepHour
    """
    df_avail_hours = conn.execute(q_avail_hours).fetchdf()

    if df_avail_hours.empty:
        st.error(
            f"No historical flights found for {sel_display_airline} on this route for {forecast_date.strftime('%A')}s in {forecast_date.strftime('%B')}. Cannot determine available departure times.")
        st.stop()

    available_hours = sorted(list(set([int(h) for h in df_avail_hours["DepHour"].unique() if pd.notna(h)])))
    hour_displays = [f"{h:02d}:00" for h in available_hours]

    col1t, col2t, col3t = st.columns([1.5, 1.5, 1])
    with col1t:
        hour_idx = st.selectbox(
            "Planned Departure Time (Hour)",
            range(len(hour_displays)),
            format_func=lambda i: hour_displays[i],
            key="forecast_time"
        )
        forecast_dep_hour = available_hours[hour_idx]
        forecast_time_int = forecast_dep_hour * 100

    # --- Number of neighbors parameter ---
    with col2t:
        n_neighbors = st.slider(
            "Neighbors (k)",
            min_value=3,
            max_value=20,
            value=10,
            step=1,
            key="n_neighbors"
        )

    st.markdown(f"**Route:** {sel_origin_disp} → {sel_dest_disp}  ")
    st.markdown(f"**Date/Time:** {forecast_date.strftime('%A, %B %d, %Y')} at {format_time(forecast_time_int)}")
    st.markdown("---")

    # --- Prepare training data with exact departure times ---
    airline_filter = f"AND CAST(DOT_ID_Reporting_Airline AS VARCHAR) = '{forecast_airline}'"
    q_train = f"""
    SELECT
      Year,
      DayofMonth,
      {DOW_SQL} AS DayOfWeek,
      Month,
      CAST(COALESCE(CRSDepTime,0) AS INTEGER) / 100 AS DepHour,
      CAST(COALESCE(CRSDepTime,0) AS INTEGER) AS DepTime,
      (COALESCE(ArrDelayMinutes,0) + COALESCE(DepDelayMinutes,0)) AS TotalDelay
    FROM flights
    WHERE CAST(OriginAirportID AS VARCHAR) = '{forecast_origin}'
      AND CAST(DestAirportID AS VARCHAR) = '{forecast_dest}'
      AND DayofMonth IS NOT NULL
      AND Month IS NOT NULL
      AND CRSDepTime IS NOT NULL
      {airline_filter}
    """
    df_train = conn.execute(q_train).fetchdf()
    if df_train.empty:
        st.error(f"No historical data found for {sel_display_airline} on this route.")
        st.stop()

    # Cast columns correctly
    df_train["Year"] = df_train["Year"].astype(int)
    df_train["DayofMonth"] = df_train["DayofMonth"].astype(int)
    df_train["DayOfWeek"] = df_train["DayOfWeek"].astype(int)
    df_train["Month"] = df_train["Month"].astype(int)
    df_train["DepHour"] = df_train["DepHour"].astype(int)
    df_train["DepTime"] = df_train["DepTime"].astype(int)

    # --- Analog logic (day-of-week + day-of-month + EXACT DEPARTURE HOUR) ---
    data1 = df_train[(df_train["Month"] == forecast_month) & (df_train["DepHour"] == forecast_dep_hour)]
    data2 = data1[data1["DayOfWeek"] == forecast_dow].copy()
    data3 = data1[data1["DayofMonth"] == forecast_day].copy()
    data2["Group"] = "DOW"
    data3["Group"] = "DOM"
    cand = pd.concat([data2, data3]).drop_duplicates().reset_index(drop=True)

    if cand.empty:
        st.error(f"No analog data available for {forecast_date.strftime('%A, %B %d')} in the historical record.")
        st.stop()

    X = cand[["DayofMonth", "DayOfWeek", "Month", "DepHour"]].values
    y = cand["TotalDelay"].values
    k_eff = min(n_neighbors, len(cand))
    knn = KNeighborsRegressor(n_neighbors=k_eff, weights="distance")
    knn.fit(X, y)
    x_t = np.array([[forecast_day, forecast_dow, forecast_month, forecast_dep_hour]])
    pred = float(knn.predict(x_t)[0])
    distances, indices = knn.kneighbors(x_t)
    nbrs = cand.iloc[indices[0]].copy()
    nbrs["distance"] = distances[0]

    # --- Display result ---
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        status = "✅ On-Time Expected" if pred < 10 else (
            "⚠️ Minor Delay Expected" if pred < 30 else "🔴 Significant Delay Expected")
        st.metric("Predicted Total Delay (Minutes)", f"{pred:.1f}")
        st.markdown(f"**Status:** {status}")

    st.markdown("---")
    st.markdown(f"### 📋 Closest Historical Analogs ({k_eff} Events Used for Prediction)")
    nbrs_disp = nbrs[["Year", "Month", "DayofMonth", "DayOfWeek", "DepTime", "TotalDelay", "Group", "distance"]].copy()
    nbrs_disp["Date"] = nbrs_disp.apply(lambda r: f"{int(r['Year'])}-{int(r['Month']):02d}-{int(r['DayofMonth']):02d}",
                                        axis=1)

    nbrs_disp["DayOfWeek"] = nbrs_disp["DayOfWeek"].map({1: "Monday", 2: "Tuesday", 3: "Wednesday",
                                                         4: "Thursday", 5: "Friday", 6: "Saturday", 7: "Sunday"})
    nbrs_disp["Scheduled Dep Time"] = nbrs_disp["DepTime"].apply(lambda t: format_time(int(t)))
    st.dataframe(nbrs_disp[["Date", "DayOfWeek", "Scheduled Dep Time", "TotalDelay", "Group", "distance"]],
                 use_container_width=True, hide_index=True)

    # --- Route map ---
    try:
        o = df_geo[df_geo["Code"] == forecast_origin].iloc[0]
        d = df_geo[df_geo["Code"] == forecast_dest].iloc[0]
        arc_df = pd.DataFrame([{"src_lat": o["Latitude"], "src_lon": o["Longitude"],
                                "dest_lat": d["Latitude"], "dest_lon": d["Longitude"], "color": [0, 128, 255, 200]}])
        arc_layer = pdk.Layer("ArcLayer", data=arc_df,
                              get_source_position=["src_lon", "src_lat"],
                              get_target_position=["dest_lon", "dest_lat"],
                              get_source_color="color", get_target_color="color",
                              get_width=6)
        scatter_df = pd.DataFrame([
            {"lat": o["Latitude"], "lon": o["Longitude"], "label": "Origin"},
            {"lat": d["Latitude"], "lon": d["Longitude"], "label": "Destination"}
        ])
        scatter_layer = pdk.Layer("ScatterplotLayer", data=scatter_df,
                                  get_position=["lon", "lat"],
                                  get_radius=25000, get_fill_color=[0, 0, 0, 120])
        view_state = pdk.ViewState(latitude=(o["Latitude"] + d["Latitude"]) / 2,
                                   longitude=(o["Longitude"] + d["Longitude"]) / 2, zoom=4, pitch=40)
        st.markdown("### 🗺️ Route Map")
        st.pydeck_chart(pdk.Deck(layers=[arc_layer, scatter_layer],
                                 initial_view_state=view_state,
                                 map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                                 height=500))
    except Exception as e:
        st.warning(f"Map unavailable: {e}")

    st.markdown("---")
    st.markdown("### How Prediction Works")
    st.markdown(f"""
    - **Data Preparation**: Flight data is aggregated to create a training dataset with features like day-of-month, day-of-week, month, and scheduled departure hour.
    - **Model Training**: A K-Nearest Neighbors (KNN) regression model is trained on this historical data. KNN looks for the **k most similar historical flights** (the "neighbors", k={k_eff}) based on the date and time features you selected.
    - **Prediction**: The model then calculates the **weighted average delay** of those {k_eff} historical flights to estimate the delay for your planned flight (weighted by distance/similarity).
    - This method provides a prediction grounded in actual past performance under similar conditions.
    - **Why No Flights on Some Routes?** Some routes may have limited historical data for specific date/time combinations. If you select a very specific date or a rare departure time, there might not be enough historical analogs for that combination.
    """)
