import pytest
from app import app
import json

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_home_route(client):
    # Test the '/' route to ensure it renders the index page correctly
    response = client.get('/')
    assert response.status_code == 200
    assert b"<title>" in response.data  # Check for a <title> tag in the HTML

def test_prediction_get(client):
    # Test the '/prediction' GET route to ensure it renders the prediction page correctly
    response = client.get('/prediction')
    assert response.status_code == 200
    assert b"<form" in response.data  # Check for a form in the HTML content

def test_prediction_post_success(client):
    
    data = {
        "neighbourhood": "City of Los Angeles",
        "room_type": "Private room",
        "accommodates": 2,
        "bathrooms": 1.0,
        "availability_365": 200,
        "bedrooms": 1,
        "minimum_nights": 3,
        "beds": 1
    }

    response = client.post(
        '/prediction',
        data=json.dumps(data),
        content_type='application/json'
    )

    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['success'] is True
    assert 'prediction' in json_data
    assert isinstance(json_data['prediction'], float) 

def test_prediction_post_failure(client):
    
    data = {
        "neighbourhood": "Brooklyn",
    }

    response = client.post(
        '/prediction',
        data=json.dumps(data),
        content_type='application/json'
    )

    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['success'] is False
    assert 'error' in json_data
