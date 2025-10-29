from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pyodbc
import json
import time
from openai import OpenAI
from google import genai
from google.genai import types
import os
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__, static_folder='public')
CORS(app)

# Optional: API clients for embedding/image generation (not required for demo)
openai_client = None
gemini_client = None

# Create directory for generated images if it doesn't exist
IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'public', 'images', 'properties')
os.makedirs(IMAGES_DIR, exist_ok=True)

# Default SQL Server connection parameters
DEFAULT_SERVER = '.'
DEFAULT_DATABASE = 'SemanticShoresDB'

def get_db_connection(server=None, database=None):
    """Create and return a database connection with optional server and database parameters"""
    server = server or DEFAULT_SERVER
    database = database or DEFAULT_DATABASE

    connection_string = (
        'DRIVER={ODBC Driver 17 for SQL Server};'
        f'SERVER={server};'
        f'DATABASE={database};'
        'Trusted_Connection=yes;'
    )
    return pyodbc.connect(connection_string)

def format_vector_for_sql(vector_list):
    """Convert Python list to SQL Server VECTOR format"""
    vector_str = '[' + ','.join(map(str, vector_list)) + ']'
    return vector_str

# Real Minnesota city coordinates for diverse geo locations
MINNESOTA_CITIES = {
    'Minneapolis': (44.9778, -93.2650),
    'St. Paul': (44.9537, -93.0900),
    'St Paul': (44.9537, -93.0900),
    'Rochester': (44.0121, -92.4802),
    'Duluth': (46.7867, -92.1005),
    'Bloomington': (44.8408, -93.2983),
    'Brooklyn Park': (45.0941, -93.3563),
    'Plymouth': (45.0105, -93.4555),
    'Woodbury': (44.9239, -92.9594),
    'Maple Grove': (45.0725, -93.4558),
    'Blaine': (45.1608, -93.2350),
    'Lakeville': (44.6497, -93.2427),
    'Eagan': (44.8041, -93.1669),
    'Eden Prairie': (44.8547, -93.4708),
    'Burnsville': (44.7677, -93.2777),
    'Minnetonka': (44.9211, -93.4688),
    'St. Cloud': (45.5579, -94.1632),
    'St Cloud': (45.5579, -94.1632),
    'Edina': (44.8897, -93.3499),
    'Coon Rapids': (45.1200, -93.2877),
    'Maplewood': (44.9531, -92.9952),
    'Mankato': (44.1636, -93.9993),
    'Moorhead': (46.8738, -96.7678),
    'Shakopee': (44.7973, -93.5266),
    'Cottage Grove': (44.8277, -92.9438),
    'Richfield': (44.8833, -93.2833),
    'Roseville': (45.0061, -93.1566),
    'Inver Grove Heights': (44.8483, -93.0466),
    'Andover': (45.2333, -93.2911),
    'Apple Valley': (44.7319, -93.2177),
    'Prior Lake': (44.7133, -93.4227),
    'Savage': (44.7789, -93.3363),
    'Chaska': (44.7894, -93.6022),
    'Fridley': (45.0861, -93.2633),
    'Owatonna': (44.0838, -93.2260),
    'Chanhassen': (44.8622, -93.5308),
    'White Bear Lake': (45.0847, -93.0099),
    'Ramsey': (45.2611, -93.4497),
    'Northfield': (44.4583, -93.1616),
    'Stillwater': (45.0564, -92.8060),
    'Faribault': (44.2950, -93.2688),
    'Winona': (44.0499, -91.6393),
    'Hopkins': (44.9252, -93.4680),
    'New Brighton': (45.0655, -93.2019),
    'Golden Valley': (44.9889, -93.3500),
    'Columbia Heights': (45.0408, -93.2630),
    'Crystal': (45.0327, -93.3600),
    'St. Louis Park': (44.9486, -93.3483),
    'St Louis Park': (44.9486, -93.3483),
    'Hastings': (44.7416, -92.8521)
}

def get_property_coordinates(city, property_id):
    """Get real coordinates for a property based on its city with minor randomization"""
    # Get base coordinates for the city
    base_coords = MINNESOTA_CITIES.get(city)

    if not base_coords:
        # Default to Minneapolis if city not found
        base_coords = MINNESOTA_CITIES['Minneapolis']

    # Add small random offset within city (roughly 0-2 miles in each direction)
    # Use property_id as seed for consistent placement
    import random
    random.seed(property_id)

    lat_variance = random.uniform(-0.02, 0.02)  # ~1.4 miles north/south
    lon_variance = random.uniform(-0.02, 0.02)  # ~1.4 miles east/west

    return (base_coords[0] + lat_variance, base_coords[1] + lon_variance)

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('public', 'index.html')

@app.route('/images/properties/<filename>')
def serve_property_image(filename):
    """Serve generated property images"""
    return send_from_directory(IMAGES_DIR, filename)

@app.route('/api/scenarios', methods=['GET'])
def get_scenarios():
    """Get available search scenarios"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        query = """
            SELECT DISTINCT
                CASE
                    WHEN search_phrase LIKE '%pool%' THEN 'pool'
                    WHEN search_phrase LIKE '%pet%' THEN 'pet'
                    WHEN search_phrase LIKE '%outdoor%' THEN 'outdoor'
                    WHEN search_phrase LIKE '%first%time%buyer%' THEN 'first-time-buyer'
                    WHEN search_phrase LIKE '%empty%nester%' OR search_phrase LIKE '%single%level%' THEN 'empty-nester'
                END AS category,
                search_phrase
            FROM dbo.search_phrases
            WHERE search_phrase LIKE '%pool%'
               OR search_phrase LIKE '%pet%'
               OR search_phrase LIKE '%outdoor%'
               OR search_phrase LIKE '%first%time%buyer%'
               OR search_phrase LIKE '%single%level%'
            ORDER BY category, search_phrase
        """

        cursor.execute(query)
        rows = cursor.fetchall()

        results = []
        for row in rows:
            results.append({
                'category': row.category,
                'search_phrase': row.search_phrase
            })

        cursor.close()
        conn.close()

        return jsonify(results)

    except Exception as e:
        print(f"Error in get_scenarios: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-embedding', methods=['POST'])
def generate_embedding():
    """Generate OpenAI embedding for custom search text"""
    try:
        data = request.json
        search_text = data.get('text', '')

        if not search_text:
            return jsonify({'error': 'No text provided'}), 400

        # Generate embedding using OpenAI
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=search_text,
            encoding_format="float"
        )

        embedding = response.data[0].embedding

        return jsonify({
            'embedding': embedding,
            'dimensions': len(embedding),
            'model': 'text-embedding-3-small'
        })

    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-property-image', methods=['POST'])
def generate_property_image():
    """Generate property image using Gemini 2.5 Flash Image based on description"""
    try:
        data = request.json
        property_id = data.get('property_id', 'unknown')
        description = data.get('description', '')
        property_type = data.get('property_type', 'home')
        address = data.get('address', '')

        if not description:
            return jsonify({'error': 'No description provided'}), 400

        # Check if image already exists
        image_filename = f'property_{property_id}.png'
        image_path = os.path.join(IMAGES_DIR, image_filename)

        if os.path.exists(image_path):
            return jsonify({
                'success': True,
                'image_url': f'/images/properties/{image_filename}',
                'cached': True
            })

        # Create descriptive narrative prompt (recommended approach for Gemini 2.5 Flash Image)
        prompt = f"""Create a beautiful, professional real estate photograph showing the exterior of this property.

This is a {property_type} located in Minneapolis, Minnesota. {description}

The scene should capture welcoming curb appeal with natural daylight, well-maintained landscaping, and a clear blue sky. Show the front entrance in an inviting way that highlights the property's best features. The photography style should be professional real estate photography - sharp, well-lit, and appealing to potential home buyers. Make it look realistic and photo-realistic, as if taken by a professional real estate photographer on a beautiful day."""

        # Generate image with Gemini 2.5 Flash Image
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash-image',
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_modalities=['Image'],
                image_config=types.ImageConfig(aspect_ratio='16:9')
            )
        )

        # Extract and save the generated image
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                image_data = part.inline_data.data
                image = Image.open(BytesIO(image_data))
                image.save(image_path)

                print(f"Generated image for property {property_id}: {image_filename}")

                return jsonify({
                    'success': True,
                    'image_url': f'/images/properties/{image_filename}',
                    'cached': False
                })

        return jsonify({'error': 'No image generated'}), 500

    except Exception as e:
        print(f"Error generating image: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['POST'])
def search_properties():
    """Perform vector search on properties"""
    try:
        data = request.json
        search_phrase = data.get('searchPhrase', '')
        top_n = data.get('topN', 10)
        price_min = data.get('priceMin')
        price_max = data.get('priceMax')
        bedrooms = data.get('bedrooms')
        use_custom_embedding = data.get('useCustomEmbedding', False)
        custom_embedding = data.get('customEmbedding')
        server = data.get('server')
        database = data.get('database')

        start_time = time.time()

        conn = get_db_connection(server, database)
        cursor = conn.cursor()

        # Build WHERE clause
        where_clause = 'p.description_vector IS NOT NULL'
        if price_min:
            where_clause += f' AND p.list_price >= {price_min}'
        if price_max:
            where_clause += f' AND p.list_price <= {price_max}'
        if bedrooms:
            where_clause += f' AND p.bedrooms = {bedrooms}'

        if use_custom_embedding and custom_embedding:
            # Use the provided custom embedding
            search_vector_str = format_vector_for_sql(custom_embedding)
        else:
            # Get vector from search_phrases table
            vector_query = """
                SELECT CAST(search_vector AS VARCHAR(MAX)) AS search_vector
                FROM dbo.search_phrases
                WHERE search_phrase = ?
            """
            cursor.execute(vector_query, search_phrase)
            row = cursor.fetchone()

            if not row:
                return jsonify({'error': 'Search phrase not found'}), 404

            search_vector_str = row.search_vector

        # Perform vector search using the vector string directly
        search_query = f"""
            DECLARE @search_vector VECTOR(1536);
            SET @search_vector = CAST('{search_vector_str}' AS VECTOR(1536));

            SELECT TOP {top_n}
                p.property_id,
                p.street_address,
                p.city,
                p.state_code,
                p.zip_code,
                p.property_type,
                p.bedrooms,
                p.bathrooms,
                p.square_feet,
                p.lot_size_sqft,
                p.list_price,
                p.year_built,
                p.listing_description,
                p.image_filename,
                CAST(VECTOR_DISTANCE('cosine', p.description_vector, @search_vector) AS DECIMAL(8,6)) AS similarity_score
            FROM dbo.properties p
            WHERE {where_clause}
            ORDER BY VECTOR_DISTANCE('cosine', p.description_vector, @search_vector);
        """

        cursor.execute(search_query)
        rows = cursor.fetchall()

        results = []
        for i, row in enumerate(rows):
            # Get real coordinates based on city
            latitude, longitude = get_property_coordinates(row.city, row.property_id)

            # Create property image URL (placeholder for now)
            image_url = f"/api/property-image/{row.property_id}" if row.image_filename else None

            results.append({
                'property_id': row.property_id,
                'street_address': row.street_address,
                'city': row.city,
                'state': row.state_code,
                'zipcode': row.zip_code,
                'property_type': row.property_type,
                'bedrooms': row.bedrooms,
                'bathrooms': row.bathrooms,
                'sqft': row.square_feet,
                'lot_size_sqft': row.lot_size_sqft,
                'list_price': row.list_price,
                'year_built': row.year_built,
                'latitude': latitude,
                'longitude': longitude,
                'listing_description': row.listing_description,
                'similarity_score': float(row.similarity_score),
                'image_url': image_url
            })

        # Generate SQL code for display
        sql_code = f"""-- SQL Server 2025 Vector Search Query
-- Database: SemanticShoresDB (100,000 properties)
-- Model: text-embedding-3-small (1536 dimensions)

DECLARE @search_vector VECTOR(1536);

-- Get pre-computed embedding for search phrase
SELECT @search_vector = search_vector
FROM dbo.search_phrases
WHERE search_phrase = '{search_phrase}';

-- Perform semantic similarity search
SELECT TOP {top_n}
    property_id,
    street_address,
    city,
    state_code,
    zip_code,
    property_type,
    bedrooms,
    bathrooms,
    square_feet,
    list_price,
    year_built,
    LEFT(listing_description, 150) + '...' AS description_preview,
    CAST(VECTOR_DISTANCE('cosine', description_vector, @search_vector) AS DECIMAL(8,6)) AS similarity_score
FROM dbo.properties p
WHERE {where_clause}
ORDER BY VECTOR_DISTANCE('cosine', description_vector, @search_vector);

-- Lower similarity_score = more similar (closer in vector space)
-- Cosine distance ranges from 0 (identical) to 2 (opposite)"""

        cursor.close()
        conn.close()

        elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Extract first few vector values for educational display
        vector_preview = search_vector_str[1:search_vector_str.find(',', 200)] + '...'

        # Also send the full vector for when user wants to see all 1536 dimensions
        vector_full = search_vector_str

        return jsonify({
            'results': results,
            'sqlCode': sql_code,
            'resultCount': len(results),
            'searchTime': round(elapsed_time, 2),
            'searchMetadata': {
                'searchPhrase': search_phrase,
                'model': 'text-embedding-3-small',
                'dimensions': 1536,
                'vectorPreview': vector_preview,
                'vectorFull': vector_full,
                'distanceMetric': 'cosine',
                'explanation': 'Your search query was converted to a 1536-dimensional vector using OpenAI text-embedding-3-small. SQL Server compares this vector against property description vectors using cosine distance (0=identical, 2=opposite). Lower scores = more similar properties.',
                'transparency': 'This search phrase was pre-embedded in the SemanticShoresDB sample database to save you API costs and improve demo performance.'
            }
        })

    except Exception as e:
        print(f"Error in search: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/property/<int:property_id>', methods=['GET'])
def get_property(property_id):
    """Get detailed property information"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        query = """
            SELECT
                p.*,
                a.agent_name,
                a.email AS agent_email,
                a.phone AS agent_phone,
                a.rating AS agent_rating
            FROM dbo.properties p
            LEFT JOIN dbo.agents a ON p.agent_id = a.agent_id
            WHERE p.property_id = ?
        """

        cursor.execute(query, property_id)
        row = cursor.fetchone()

        if not row:
            return jsonify({'error': 'Property not found'}), 404

        result = {
            'property_id': row.property_id,
            'street_address': row.street_address,
            'city': row.city,
            'state': row.state_code if hasattr(row, 'state_code') else None,
            'zipcode': row.zip_code if hasattr(row, 'zip_code') else None,
            'property_type': row.property_type,
            'bedrooms': row.bedrooms,
            'bathrooms': row.bathrooms,
            'sqft': row.square_feet if hasattr(row, 'square_feet') else None,
            'lot_size_sqft': row.lot_size_sqft,
            'list_price': row.list_price,
            'year_built': row.year_built,
            'listing_description': row.listing_description,
            'agent_name': row.agent_name if hasattr(row, 'agent_name') else None,
            'agent_email': row.agent_email if hasattr(row, 'agent_email') else None,
            'agent_phone': row.agent_phone if hasattr(row, 'agent_phone') else None,
            'agent_rating': float(row.agent_rating) if hasattr(row, 'agent_rating') and row.agent_rating else None
        }

        cursor.close()
        conn.close()

        return jsonify(result)

    except Exception as e:
        print(f"Error getting property: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print('\nVector Search Demo running at http://localhost:3000')
    print('Database: SemanticShoresDB (100,000 properties)')
    print('Ready for semantic real estate search!\n')
    app.run(host='0.0.0.0', port=3000, debug=False, use_reloader=False)
