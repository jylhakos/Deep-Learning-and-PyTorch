"""
Weather Service Module for Real-time Temperature Data
Handles geocoding and weather API calls for electricity forecasting
"""

import requests
import json
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time

class WeatherService:
    """
    Service class for fetching real-time weather data using Open-Meteo API
    """
    
    def __init__(self):
        self.base_url = "https://api.open-meteo.com/v1/forecast"
        self.geocoder = Nominatim(user_agent="electricity_forecasting_app")
        
        # Cache for coordinates to avoid repeated geocoding
        self.coordinates_cache = {}
        
        # Default location for Lazio, Italy
        self.default_location = {
            'city': 'Roma',
            'region': 'Lazio',
            'country': 'Italy'
        }
    
    def get_coordinates(self, city: str, region: str = "Lazio", country: str = "Italy") -> Optional[Tuple[float, float]]:
        """
        Get latitude and longitude coordinates for a given location using geocoding
        
        Args:
            city: City name (e.g., "Roma")
            region: Region/state name (e.g., "Lazio") 
            country: Country name (e.g., "Italy")
            
        Returns:
            Tuple of (latitude, longitude) or None if geocoding fails
        """
        # Create cache key
        location_key = f"{city}, {region}, {country}"
        
        # Check cache first
        if location_key in self.coordinates_cache:
            print(f"Using cached coordinates for {location_key}")
            return self.coordinates_cache[location_key]
        
        try:
            print(f"Geocoding location: {location_key}")
            
            # Try with full address first
            location = self.geocoder.geocode(location_key, timeout=10)
            
            if location is None:
                # Try with just city and country
                fallback_key = f"{city}, {country}"
                print(f"Trying fallback geocoding: {fallback_key}")
                location = self.geocoder.geocode(fallback_key, timeout=10)
            
            if location is None:
                # Try with just region and country
                region_key = f"{region}, {country}"
                print(f"Trying region geocoding: {region_key}")
                location = self.geocoder.geocode(region_key, timeout=10)
            
            if location:
                coordinates = (location.latitude, location.longitude)
                self.coordinates_cache[location_key] = coordinates
                
                print(f"✅ Geocoding successful:")
                print(f"   Location: {location.address}")
                print(f"   Coordinates: {coordinates[0]:.4f}, {coordinates[1]:.4f}")
                
                return coordinates
            else:
                print(f"❌ Geocoding failed for {location_key}")
                return None
                
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            print(f"❌ Geocoding error: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected geocoding error: {e}")
            return None
    
    def get_default_coordinates(self) -> Tuple[float, float]:
        """
        Get default coordinates for Lazio, Italy (Roma)
        
        Returns:
            Tuple of (latitude, longitude) for Roma, Lazio
        """
        # Roma, Lazio coordinates (approximate center)
        roma_coordinates = (41.9028, 12.4964)
        
        print(f"Using default coordinates for Roma, Lazio: {roma_coordinates}")
        return roma_coordinates
    
    def fetch_weather_forecast(self, latitude: float, longitude: float, days: int = 7) -> Optional[Dict]:
        """
        Fetch weather forecast data from Open-Meteo API
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate  
            days: Number of forecast days (default 7)
            
        Returns:
            Weather data dictionary or None if request fails
        """
        try:
            print(f"Fetching weather forecast for coordinates: {latitude:.4f}, {longitude:.4f}")
            
            # Prepare API parameters
            params = {
                'latitude': latitude,
                'longitude': longitude,
                'daily': 'temperature_2m_max,temperature_2m_min,temperature_2m_mean',
                'timezone': 'Europe/Rome',
                'forecast_days': min(days, 16)  # Open-Meteo supports up to 16 days
            }
            
            # Make API request
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            weather_data = response.json()
            
            print(f"✅ Weather data fetched successfully")
            print(f"   Timezone: {weather_data.get('timezone', 'Unknown')}")
            print(f"   Forecast days: {len(weather_data.get('daily', {}).get('time', []))}")
            
            return weather_data
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Weather API request failed: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected weather API error: {e}")
            return None
    
    def get_tomorrow_temperature(self, city: str = None, region: str = "Lazio", country: str = "Italy") -> Optional[float]:
        """
        Get tomorrow's average temperature for a specific location
        
        Args:
            city: City name (defaults to Roma if None)
            region: Region name (default "Lazio")
            country: Country name (default "Italy")
            
        Returns:
            Tomorrow's average temperature in Celsius or None if failed
        """
        # Use default city if none provided
        if city is None:
            city = self.default_location['city']
        
        print(f"\n🌡️  Getting tomorrow's temperature for {city}, {region}, {country}")
        print("-" * 60)
        
        # Get coordinates
        coordinates = self.get_coordinates(city, region, country)
        
        if coordinates is None:
            print("⚠️  Geocoding failed, using default coordinates for Roma, Lazio")
            coordinates = self.get_default_coordinates()
        
        # Fetch weather data
        weather_data = self.fetch_weather_forecast(coordinates[0], coordinates[1], days=3)
        
        if weather_data is None:
            print("❌ Failed to fetch weather data")
            return None
        
        try:
            # Extract tomorrow's temperature
            daily_data = weather_data['daily']
            dates = daily_data['time']
            temps_mean = daily_data.get('temperature_2m_mean', [])
            temps_max = daily_data.get('temperature_2m_max', [])
            temps_min = daily_data.get('temperature_2m_min', [])
            
            # Get tomorrow's date
            tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            
            # Find tomorrow's temperature
            tomorrow_temp = None
            for i, date in enumerate(dates):
                if date == tomorrow:
                    if temps_mean and i < len(temps_mean):
                        tomorrow_temp = temps_mean[i]
                    elif temps_max and temps_min and i < len(temps_max) and i < len(temps_min):
                        # Calculate average if mean not available
                        tomorrow_temp = (temps_max[i] + temps_min[i]) / 2
                    break
            
            if tomorrow_temp is not None:
                print(f"✅ Tomorrow's temperature ({tomorrow}): {tomorrow_temp:.1f}°C")
                
                # Show additional info if available
                if len(dates) > 0:
                    print(f"   Forecast location: {weather_data.get('timezone', 'Unknown')}")
                    if temps_max and temps_min:
                        idx = dates.index(tomorrow) if tomorrow in dates else 0
                        if idx < len(temps_max) and idx < len(temps_min):
                            print(f"   Temperature range: {temps_min[idx]:.1f}°C - {temps_max[idx]:.1f}°C")
                
                return tomorrow_temp
            else:
                print(f"❌ Tomorrow's temperature not found in forecast data")
                return None
                
        except (KeyError, IndexError, ValueError) as e:
            print(f"❌ Error parsing weather data: {e}")
            return None
    
    def get_current_weather(self, latitude: float, longitude: float) -> Optional[Dict]:
        """
        Get current weather conditions
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            Current weather data dictionary or None if failed
        """
        try:
            params = {
                'latitude': latitude,
                'longitude': longitude,
                'current_weather': True,
                'timezone': 'Europe/Rome'
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            print(f"❌ Current weather request failed: {e}")
            return None
    
    def get_extended_forecast(self, city: str = None, days: int = 7) -> Optional[Dict]:
        """
        Get extended weather forecast for multiple days
        
        Args:
            city: City name (defaults to Roma)
            days: Number of forecast days
            
        Returns:
            Extended forecast data or None if failed
        """
        if city is None:
            city = self.default_location['city']
        
        print(f"Getting {days}-day forecast for {city}, Lazio, Italy")
        
        # Get coordinates
        coordinates = self.get_coordinates(city, "Lazio", "Italy")
        
        if coordinates is None:
            coordinates = self.get_default_coordinates()
        
        # Fetch extended forecast
        weather_data = self.fetch_weather_forecast(coordinates[0], coordinates[1], days)
        
        if weather_data:
            print(f"✅ Extended forecast retrieved for {days} days")
            return weather_data
        else:
            print(f"❌ Failed to get extended forecast")
            return None

class WeatherDataProcessor:
    """
    Process weather data for electricity forecasting models
    """
    
    def __init__(self):
        self.weather_service = WeatherService()
    
    def prepare_forecast_input(self, historical_data: list, city: str = None) -> Optional[Dict]:
        """
        Prepare complete input data for electricity forecasting including tomorrow's weather
        
        Args:
            historical_data: List of historical electricity consumption and weather data
            city: City name for weather forecast (optional)
            
        Returns:
            Complete input data dictionary ready for model prediction
        """
        print("\n🔮 Preparing forecast input with real-time weather data")
        print("=" * 60)
        
        # Get tomorrow's temperature
        tomorrow_temp = self.weather_service.get_tomorrow_temperature(city)
        
        if tomorrow_temp is None:
            print("⚠️  Using fallback temperature estimation")
            # Fallback: use recent temperature trend
            if historical_data and len(historical_data) > 0:
                recent_temps = [item.get('temperature', 15) for item in historical_data[-7:]]
                tomorrow_temp = sum(recent_temps) / len(recent_temps)
                print(f"   Estimated temperature based on recent data: {tomorrow_temp:.1f}°C")
            else:
                tomorrow_temp = 15.0  # Default reasonable temperature
                print(f"   Using default temperature: {tomorrow_temp:.1f}°C")
        
        # Prepare complete forecast input
        forecast_input = {
            'historical_data': historical_data,
            'next_day_temperature': tomorrow_temp,
            'weather_source': 'open_meteo_api',
            'forecast_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
            'generated_at': datetime.now().isoformat()
        }
        
        print(f"✅ Forecast input prepared:")
        print(f"   Historical records: {len(historical_data)}")
        print(f"   Next day temperature: {tomorrow_temp:.1f}°C")
        print(f"   Forecast date: {forecast_input['forecast_date']}")
        
        return forecast_input

def test_weather_service():
    """
    Test function for the weather service
    """
    print("🧪 TESTING WEATHER SERVICE")
    print("=" * 40)
    
    weather_service = WeatherService()
    
    # Test 1: Geocoding
    print("\n1. Testing geocoding...")
    coords = weather_service.get_coordinates("Roma", "Lazio", "Italy")
    if coords:
        print(f"✅ Roma coordinates: {coords}")
    else:
        print("❌ Geocoding failed")
    
    # Test 2: Weather forecast
    print("\n2. Testing weather forecast...")
    if coords:
        weather_data = weather_service.fetch_weather_forecast(coords[0], coords[1], 3)
        if weather_data:
            print("✅ Weather forecast retrieved")
            daily_data = weather_data.get('daily', {})
            if 'time' in daily_data:
                print(f"   Forecast dates: {daily_data['time']}")
        else:
            print("❌ Weather forecast failed")
    
    # Test 3: Tomorrow's temperature
    print("\n3. Testing tomorrow's temperature...")
    tomorrow_temp = weather_service.get_tomorrow_temperature("Roma")
    if tomorrow_temp:
        print(f"✅ Tomorrow's temperature: {tomorrow_temp}°C")
    else:
        print("❌ Tomorrow's temperature failed")
    
    # Test 4: Weather data processor
    print("\n4. Testing weather data processor...")
    processor = WeatherDataProcessor()
    
    # Sample historical data
    sample_historical = [
        {"date": "2024-12-25", "load": 14500, "temperature": 8.5},
        {"date": "2024-12-26", "load": 15200, "temperature": 6.2},
        {"date": "2024-12-27", "load": 14800, "temperature": 7.1}
    ]
    
    forecast_input = processor.prepare_forecast_input(sample_historical, "Roma")
    if forecast_input:
        print("✅ Forecast input prepared successfully")
        print(f"   Temperature source: {forecast_input.get('weather_source')}")
    else:
        print("❌ Forecast input preparation failed")

if __name__ == "__main__":
    test_weather_service()
