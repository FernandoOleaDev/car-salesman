import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
import re
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchCriteria(Enum):
    MAKE = "make"
    MODEL = "model"
    COLOR = "color"
    BODY_STYLE = "body_style"
    PRICE_RANGE = "price_range"
    MILEAGE_RANGE = "mileage_range"
    FUEL_TYPE = "fuel_type"
    FEATURES = "features"
    TRUNK_SPACE = "trunk_space"

@dataclass
class CarSearchResult:
    """Structured car search result"""
    car_id: str
    year: int
    make: str
    model: str
    body_style: str
    color: str
    mileage: int
    price: int
    fuel_type: str
    engine: str
    transmission: str
    safety_rating: int
    trunk_space_liters: int
    features: List[str]
    condition: str
    location: str
    vin: str
    match_score: float
    match_reasons: List[str]

class EnhancedInventoryManager:
    """Advanced inventory management system with intelligent search capabilities"""
    
    def __init__(self, inventory_path: str = "data/enhanced_inventory.csv"):
        self.inventory_path = inventory_path
        self.inventory_df = None
        self.search_history = []
        self.load_inventory()
    
    def load_inventory(self) -> bool:
        """Load inventory with enhanced error handling and validation"""
        try:
            logger.info(f"Loading enhanced inventory from: {self.inventory_path}")
            self.inventory_df = pd.read_csv(self.inventory_path)
            
            # Validate required columns
            required_columns = [
                'year', 'make', 'model', 'body_styles', 'color', 'mileage', 
                'price', 'fuel_type', 'engine', 'transmission', 'safety_rating',
                'trunk_space_liters', 'features', 'condition', 'location', 'vin'
            ]
            
            missing_columns = [col for col in required_columns if col not in self.inventory_df.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                self.inventory_df = None # Ensure inventory is not used if malformed
                return False
            
            # Add status column if it doesn't exist
            if 'status' not in self.inventory_df.columns:
                self.inventory_df['status'] = 'Available'
            else:
                # Ensure existing status values are standardized, e.g., fill NaNs
                self.inventory_df['status'] = self.inventory_df['status'].fillna('Available')

            # Clean and process data
            self.inventory_df['features_list'] = self.inventory_df['features'].apply(self._parse_features)
            self.inventory_df['body_styles_list'] = self.inventory_df['body_styles'].apply(self._parse_body_styles)
            self.inventory_df['search_text'] = self.inventory_df.apply(self._create_search_text, axis=1)
            
            logger.info(f"✅ Enhanced inventory loaded successfully: {len(self.inventory_df)} vehicles")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading inventory: {e}")
            return False
    
    def _parse_features(self, features_str: str) -> List[str]:
        """Parse features string into list"""
        if pd.isna(features_str):
            return []
        return [f.strip() for f in features_str.split(',')]
    
    def _parse_body_styles(self, body_styles_str: str) -> List[str]:
        """Parse body styles string into list"""
        if pd.isna(body_styles_str):
            return []
        # Handle both JSON-like format and simple string
        if body_styles_str.startswith('['):
            try:
                return json.loads(body_styles_str.replace("'", '"'))
            except:
                return [body_styles_str]
        return [body_styles_str]
    
    def _create_search_text(self, row) -> str:
        """Create searchable text for each vehicle"""
        text_parts = [
            str(row['year']),
            row['make'],
            row['model'],
            row['color'],
            row['fuel_type'],
            row['condition'],
            ' '.join(row['features_list']),
            ' '.join(row['body_styles_list']),
            row.get('status', 'Available') # Include status in search text
        ]
        return ' '.join(text_parts).lower()
    
    def intelligent_search(self, query: str, max_results: int = 10) -> List[CarSearchResult]:
        """Advanced intelligent search with natural language processing"""
        if self.inventory_df is None or self.inventory_df.empty:
            logger.warning("No inventory data available for search.")
            return []
        
        logger.info(f"🔍 Performing intelligent search: '{query}'")
        
        # Filter out reserved cars by default from the initial DataFrame copy
        active_inventory_df = self.inventory_df[self.inventory_df['status'] == 'Available'].copy()
        if active_inventory_df.empty:
            logger.info("No 'Available' vehicles in inventory to search.")
            # Optionally, search all if query implies looking for reserved/sold, but for now, only available.
            # For a customer-facing search, this is the correct behavior.
            return []
        
        # Parse query for specific criteria
        search_criteria = self._parse_search_query(query)
        
        # Apply filters based on criteria to the active inventory
        filtered_df = self._apply_search_filters(active_inventory_df, search_criteria)
        
        # Calculate relevance scores
        scored_results = self._calculate_relevance_scores(filtered_df, query, search_criteria)
        
        # Convert to structured results
        results = self._convert_to_search_results(scored_results, max_results)
        
        # Log search
        self.search_history.append({
            'timestamp': datetime.now(),
            'query': query,
            'results_count': len(results),
            'criteria': search_criteria
        })
        
        logger.info(f"✅ Found {len(results)} matching vehicles")
        return results
    
    def _parse_search_query(self, query: str) -> Dict[str, Any]:
        """Parse natural language query into search criteria"""
        query_lower = query.lower()
        criteria = {}
        
        # Extract price range
        price_patterns = [
            r'menos de (\d+)',
            r'bajo (\d+)',
            r'máximo (\d+)',
            r'hasta (\d+)',
            r'entre (\d+) y (\d+)',
            r'(\d+) a (\d+)',
            r'presupuesto de (\d+)'
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, query_lower)
            if match:
                if len(match.groups()) == 1:
                    criteria['max_price'] = int(match.group(1))
                elif len(match.groups()) == 2:
                    criteria['min_price'] = int(match.group(1))
                    criteria['max_price'] = int(match.group(2))
                break
        
        # Extract mileage preferences
        mileage_patterns = [
            r'pocos kilómetros',
            r'bajo kilometraje',
            r'menos de (\d+) km',
            r'máximo (\d+) kilómetros'
        ]
        
        for pattern in mileage_patterns:
            if re.search(pattern, query_lower):
                if 'pocos' in pattern or 'bajo' in pattern:
                    criteria['max_mileage'] = 20000
                else:
                    match = re.search(r'(\d+)', pattern)
                    if match:
                        criteria['max_mileage'] = int(match.group(1))
                break
        
        # Extract colors
        colors = ['rojo', 'negro', 'blanco', 'azul', 'gris', 'verde', 'amarillo', 'naranja']
        for color in colors:
            if color in query_lower:
                criteria['color'] = color.capitalize()
                break
        
        # Extract body styles
        body_styles = {
            'sedan': ['sedan', 'sedán'],
            'suv': ['suv', 'todoterreno', 'camioneta'],
            'pickup': ['pickup', 'pick-up', 'camioneta'],
            'hatchback': ['hatchback', 'compacto'],
            'coupe': ['coupe', 'coupé', 'deportivo'],
            'van': ['van', 'furgoneta', 'monovolumen']
        }
        
        for style, keywords in body_styles.items():
            if any(keyword in query_lower for keyword in keywords):
                criteria['body_style'] = style
                break
        
        # Extract makes
        makes = ['audi', 'bmw', 'mercedes', 'toyota', 'honda', 'ford', 'volkswagen', 
                'nissan', 'hyundai', 'kia', 'mazda', 'subaru', 'lexus', 'acura',
                'infiniti', 'volvo', 'cadillac', 'genesis', 'jaguar', 'land rover',
                'porsche', 'tesla', 'aston martin', 'ferrari', 'lamborghini']
        
        for make in makes:
            if make in query_lower:
                criteria['make'] = make.title()
                break
        
        # Extract fuel type preferences
        fuel_types = {
            'híbrido': ['híbrido', 'hibrido', 'hybrid'],
            'eléctrico': ['eléctrico', 'electrico', 'electric'],
            'gasolina': ['gasolina', 'gasoline'],
            'diesel': ['diesel', 'diésel']
        }
        
        for fuel_type, keywords in fuel_types.items():
            if any(keyword in query_lower for keyword in keywords):
                criteria['fuel_type'] = fuel_type.capitalize()
                break
        
        # Extract feature requirements
        feature_keywords = {
            'seguridad': ['seguro', 'seguridad', 'safety'],
            'lujo': ['lujo', 'luxury', 'premium'],
            'deportivo': ['deportivo', 'sport', 'performance'],
            'familiar': ['familiar', 'family', 'bebé', 'niños'],
            'maletero': ['maletero', 'trunk', 'espacio', 'carga'],
            'tecnología': ['tecnología', 'tech', 'navegación', 'pantalla']
        }
        
        criteria['required_features'] = []
        for feature, keywords in feature_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                criteria['required_features'].append(feature)
        
        return criteria
    
    def _apply_search_filters(self, df: pd.DataFrame, criteria: Dict[str, Any]) -> pd.DataFrame:
        """Apply search filters based on criteria"""
        
        # Price filters
        if 'min_price' in criteria:
            df = df[df['price'] >= criteria['min_price']]
        if 'max_price' in criteria:
            df = df[df['price'] <= criteria['max_price']]
        
        # Mileage filters
        if 'max_mileage' in criteria:
            df = df[df['mileage'] <= criteria['max_mileage']]
        
        # Color filter
        if 'color' in criteria:
            df = df[df['color'].str.lower() == criteria['color'].lower()]
        
        # Make filter
        if 'make' in criteria:
            df = df[df['make'].str.lower() == criteria['make'].lower()]
        
        # Body style filter
        if 'body_style' in criteria:
            df = df[df['body_styles_list'].apply(
                lambda styles: any(criteria['body_style'].lower() in style.lower() for style in styles)
            )]
        
        # Fuel type filter
        if 'fuel_type' in criteria:
            df = df[df['fuel_type'].str.lower() == criteria['fuel_type'].lower()]
        
        return df
    
    def _calculate_relevance_scores(self, df: pd.DataFrame, query: str, criteria: Dict[str, Any]) -> pd.DataFrame:
        """Calculate relevance scores for search results"""
        if df.empty:
            return df
        
        df = df.copy()
        df['relevance_score'] = 0.0
        df['match_reasons'] = [[] for _ in range(len(df))]
        
        query_words = query.lower().split()
        
        for idx, row in df.iterrows():
            score = 0.0
            reasons = []
            
            # Text matching score
            text_matches = sum(1 for word in query_words if word in row['search_text'])
            if text_matches > 0:
                score += (text_matches / len(query_words)) * 30
                reasons.append(f"Coincidencia de texto ({text_matches}/{len(query_words)} palabras)")
            
            # Criteria matching bonuses
            if 'color' in criteria and row['color'].lower() == criteria['color'].lower():
                score += 25
                reasons.append(f"Color exacto: {criteria['color']}")
            
            if 'make' in criteria and row['make'].lower() == criteria['make'].lower():
                score += 30
                reasons.append(f"Marca exacta: {criteria['make']}")
            
            if 'body_style' in criteria:
                if any(criteria['body_style'].lower() in style.lower() for style in row['body_styles_list']):
                    score += 25
                    reasons.append(f"Tipo de carrocería: {criteria['body_style']}")
            
            # Condition bonus
            if row['condition'] == 'Excelente':
                score += 10
                reasons.append("Condición excelente")
            elif row['condition'] == 'Muy bueno':
                score += 5
                reasons.append("Muy buena condición")
            
            # Low mileage bonus
            if row['mileage'] < 15000:
                score += 15
                reasons.append("Bajo kilometraje")
            elif row['mileage'] < 25000:
                score += 10
                reasons.append("Kilometraje moderado")
            
            # Safety rating bonus
            if row['safety_rating'] == 5:
                score += 10
                reasons.append("Máxima calificación de seguridad")
            
            # Feature matching
            if 'required_features' in criteria:
                feature_matches = 0
                for req_feature in criteria['required_features']:
                    if any(req_feature.lower() in feature.lower() for feature in row['features_list']):
                        feature_matches += 1
                        reasons.append(f"Característica requerida: {req_feature}")
                
                if feature_matches > 0:
                    score += feature_matches * 15
            
            df.at[idx, 'relevance_score'] = score
            df.at[idx, 'match_reasons'] = reasons
        
        return df.sort_values('relevance_score', ascending=False)
    
    def _convert_to_search_results(self, df: pd.DataFrame, max_results: int) -> List[CarSearchResult]:
        """Convert DataFrame to structured search results"""
        results = []
        
        for idx, row in df.head(max_results).iterrows():
            result = CarSearchResult(
                car_id=str(idx),
                year=int(row['year']),
                make=row['make'],
                model=row['model'],
                body_style=', '.join(row['body_styles_list']),
                color=row['color'],
                mileage=int(row['mileage']),
                price=int(row['price']),
                fuel_type=row['fuel_type'],
                engine=row['engine'],
                transmission=row['transmission'],
                safety_rating=int(row['safety_rating']),
                trunk_space_liters=int(row['trunk_space_liters']),
                features=row['features_list'],
                condition=row['condition'],
                location=row['location'],
                vin=row['vin'],
                match_score=row['relevance_score'],
                match_reasons=row['match_reasons']
            )
            results.append(result)
        
        return results
    
    def get_car_by_vin(self, vin: str) -> Optional[CarSearchResult]:
        """Get a single car's details by its VIN, including status."""
        if self.inventory_df is None:
            logger.warning("Inventory not loaded, cannot get car by VIN.")
            return None
        
        car_series = self.inventory_df[self.inventory_df['vin'].str.strip() == vin.strip()].iloc[0] if not self.inventory_df[self.inventory_df['vin'].str.strip() == vin.strip()].empty else None
        
        if car_series is not None:
            return CarSearchResult(
                car_id=str(car_series.name), # Use DataFrame index as a simple ID
                year=int(car_series['year']),
                make=car_series['make'],
                model=car_series['model'],
                body_style=', '.join(self._parse_body_styles(car_series['body_styles'])),
                color=car_series['color'],
                mileage=int(car_series['mileage']),
                price=int(car_series['price']),
                fuel_type=car_series['fuel_type'],
                engine=car_series['engine'],
                transmission=car_series['transmission'],
                safety_rating=int(car_series['safety_rating']),
                trunk_space_liters=int(car_series['trunk_space_liters']),
                features=self._parse_features(car_series['features']),
                condition=car_series['condition'],
                location=car_series['location'],
                vin=car_series['vin'],
                match_score=1.0, # Direct match by VIN
                match_reasons=['Direct VIN lookup']
            )
        logger.warning(f"VIN {vin} not found in inventory.")
        return None

    def reserve_vehicle(self, vin: str) -> bool:
        """Reserve a vehicle by setting its status to 'Reserved' and save."""
        if self.inventory_df is None:
            logger.error("Inventory not loaded. Cannot reserve vehicle.")
            return False

        vin_stripped = vin.strip()
        vehicle_index = self.inventory_df[self.inventory_df['vin'].str.strip() == vin_stripped].index

        if not vehicle_index.empty:
            idx = vehicle_index[0]
            current_status = self.inventory_df.loc[idx, 'status']
            if current_status == 'Available':
                self.inventory_df.loc[idx, 'status'] = 'Reserved'
                try:
                    self.inventory_df.to_csv(self.inventory_path, index=False)
                    logger.info(f"✅ Vehicle {vin_stripped} successfully reserved.")
                    return True
                except Exception as e:
                    logger.error(f"❌ Error saving inventory after reserving {vin_stripped}: {e}")
                    # Optionally revert status change if save fails
                    self.inventory_df.loc[idx, 'status'] = 'Available' 
                    return False
            else:
                logger.warning(f"⚠️ Vehicle {vin_stripped} could not be reserved. Current status: {current_status}")
                return False
        else:
            logger.warning(f"⚠️ Vehicle {vin_stripped} not found for reservation.")
            return False
    
    def update_car_status(self, vin: str, status: str) -> bool:
        """Update car status (sold, reserved, etc.)"""
        if self.inventory_df is None:
            return False
        
        try:
            # Add status column if it doesn't exist
            if 'status' not in self.inventory_df.columns:
                self.inventory_df['status'] = 'Disponible'
            
            # Update status
            mask = self.inventory_df['vin'] == vin
            if mask.any():
                self.inventory_df.loc[mask, 'status'] = status
                logger.info(f"✅ Updated car {vin} status to: {status}")
                return True
            else:
                logger.warning(f"❌ Car with VIN {vin} not found")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error updating car status: {e}")
            return False
    
    def get_inventory_stats(self) -> Dict[str, Any]:
        """Get comprehensive inventory statistics"""
        if self.inventory_df is None or self.inventory_df.empty:
            return {}
        
        stats = {
            'total_vehicles': len(self.inventory_df),
            'total_value': self.inventory_df['price'].sum(),
            'average_price': self.inventory_df['price'].mean(),
            'average_mileage': self.inventory_df['mileage'].mean(),
            'makes_count': self.inventory_df['make'].nunique(),
            'top_makes': self.inventory_df['make'].value_counts().head(5).to_dict(),
            'body_styles': self.inventory_df['body_styles'].value_counts().head(5).to_dict(),
            'fuel_types': self.inventory_df['fuel_type'].value_counts().to_dict(),
            'conditions': self.inventory_df['condition'].value_counts().to_dict(),
            'price_ranges': {
                'under_30k': len(self.inventory_df[self.inventory_df['price'] < 30000]),
                '30k_to_50k': len(self.inventory_df[(self.inventory_df['price'] >= 30000) & (self.inventory_df['price'] < 50000)]),
                '50k_to_100k': len(self.inventory_df[(self.inventory_df['price'] >= 50000) & (self.inventory_df['price'] < 100000)]),
                'over_100k': len(self.inventory_df[self.inventory_df['price'] >= 100000])
            }
        }
        
        return stats
    
    def format_search_results_for_agent(self, results: List[CarSearchResult], max_display: int = 5) -> str:
        """Format search results for agent consumption"""
        if not results:
            return "❌ No se encontraron vehículos que coincidan con los criterios de búsqueda."
        
        output = f"🎯 **Encontré {len(results)} vehículos excelentes para ti:**\n\n"
        
        for i, car in enumerate(results[:max_display], 1):
            # Price formatting
            price_formatted = f"€{car.price:,}".replace(',', '.')
            
            # Mileage formatting
            mileage_formatted = f"{car.mileage:,} km".replace(',', '.')
            
            # Features preview (first 3)
            features_preview = ', '.join(car.features[:3])
            if len(car.features) > 3:
                features_preview += f" y {len(car.features) - 3} más"
            
            output += f"**{i}. {car.year} {car.make} {car.model}** ({car.body_style})\n"
            output += f"   🎨 Color: {car.color} | 📏 {mileage_formatted} | 💰 {price_formatted}\n"
            output += f"   ⛽ {car.fuel_type} | ⭐ {car.safety_rating}/5 estrellas | 🧳 {car.trunk_space_liters}L maletero\n"
            output += f"   ✨ {features_preview}\n"
            output += f"   📍 Ubicación: {car.location} | 🏆 Condición: {car.condition}\n"
            
            # Match reasons
            if car.match_reasons:
                reasons = ', '.join(car.match_reasons[:2])
                output += f"   🎯 **Por qué es perfecto:** {reasons}\n"
            
            output += f"   📋 VIN: `{car.vin}`\n\n"
        
        if len(results) > max_display:
            output += f"... y {len(results) - max_display} opciones más disponibles.\n\n"
        
        output += "💡 **¿Te interesa alguno de estos vehículos? ¡Puedo darte más detalles o programar una prueba de manejo!**"
        
        return output

# Global instance for easy access
inventory_manager = EnhancedInventoryManager()

def get_inventory_manager() -> EnhancedInventoryManager:
    """Get the global inventory manager instance"""
    return inventory_manager 