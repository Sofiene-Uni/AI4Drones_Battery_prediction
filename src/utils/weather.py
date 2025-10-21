


def get_weather_data(lat, lon, alt, date, hour, minute, second):
    """
    Dummy function to return weather data for a given GPS point and datetime components.
    Replace with actual API calls later.

    Args:
        lat, lon, alt: GPS coordinates
        date: datetime.date object
        hour, minute, second: time components

    Returns:
        wind_magnitude (m/s)
        wind_direction_deg (meteorological, FROM)
        temperature_c (°C)
        humidity_percent (%)
        air_density_kg_m3 (kg/m^3)
    """
    # Dummy values
    wind_magnitude = 5.0
    wind_direction_deg = 90.0
    temperature_c = 25.0
    humidity_percent = 50.0
    air_density_kg_m3 = 1.225

    return wind_magnitude, wind_direction_deg, temperature_c, humidity_percent, air_density_kg_m3

