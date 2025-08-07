
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import numpy as np

from clients.data_clients.soil_client import SoilDataClient

class TestSoilDataClient(unittest.TestCase):

    def setUp(self):
        self.client = SoilDataClient()
        self.canadian_coords = (50.0, -100.0)
        self.international_coords = (40.0, -30.0)

    @patch('requests.Session.get')
    def test_get_soilgrids_data_success(self, mock_get):
        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "properties": {
                "layers": [
                    {
                        "name": "clay",
                        "unit_measure": {"mapped_units": "g/kg"},
                        "depths": [
                            {"label": "0-5cm", "values": {"mean": 150}},
                        ]
                    }
                ]
            }
        }
        mock_get.return_value = mock_response

        result = self.client.get_soilgrids_data(
            self.international_coords,
            properties=['clay'],
            depths=['0-5cm']
        )

        self.assertTrue(result['success'])
        self.assertIn('soil_properties_by_depth', result)
        self.assertIn('clay', result['soil_properties_by_depth'])
        self.assertIn('0-5cm', result['soil_properties_by_depth']['clay'])
        self.assertEqual(result['soil_properties_by_depth']['clay']['0-5cm']['value'], 15.0)

    @patch('requests.Session.get')
    def test_get_canadian_soil_data_success(self, mock_get):
        # Mock AAFC API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "features": [
                {
                    "attributes": {
                        "CLI_CLASS": "3",
                        "TEXTURE": "Loam"
                    }
                }
            ]
        }
        mock_get.return_value = mock_response

        result = self.client.get_canadian_soil_data(self.canadian_coords)

        self.assertTrue(result['success'])
        self.assertIn('soil_properties', result)
        self.assertEqual(result['soil_properties']['capability_class'], '3')
        self.assertEqual(result['soil_properties']['texture'], 'Loam')

    def test_get_canadian_soil_data_outside_canada(self):
        result = self.client.get_canadian_soil_data(self.international_coords)
        self.assertFalse(result['success'])
        self.assertEqual(result['error'], 'Coordinates are outside Canada bounds')

    @patch('clients.data_clients.soil_client.SoilDataClient.get_canadian_soil_data')
    def test_get_soil_properties_for_point_canada(self, mock_get_canadian):
        mock_get_canadian.return_value = {'success': True, 'data': 'canadian_data'}
        result = self.client.get_soil_properties_for_point(self.canadian_coords)
        self.assertTrue(result['success'])
        self.assertEqual(result['data'], 'canadian_data')

    @patch('clients.data_clients.soil_client.SoilDataClient.get_soilgrids_data')
    @patch('clients.data_clients.soil_client.SoilDataClient.get_canadian_soil_data')
    def test_get_soil_properties_for_point_international(self, mock_get_canadian, mock_get_soilgrids):
        mock_get_canadian.return_value = {'success': False, 'error': 'Outside Canada'}
        mock_get_soilgrids.return_value = {'success': True, 'data': 'soilgrids_data'}

        result = self.client.get_soil_properties_for_point(self.international_coords)
        self.assertTrue(result['success'])
        self.assertEqual(result['data'], 'soilgrids_data')

    @patch('clients.data_clients.soil_client.SoilDataClient.get_soilgrids_data')
    @patch('rasterio.open')
    def test_get_soil_raster_for_watershed(self, mock_rasterio_open, mock_get_soilgrids):
        mock_get_soilgrids.return_value = {
            'success': True,
            'soil_properties_by_depth': {
                'clay': {'0-5cm': {'value': 15.0}},
                'sand': {'0-5cm': {'value': 40.0}},
                'phh2o': {'0-5cm': {'value': 6.5}}
            }
        }

        bbox = (-100.0, 50.0, -99.0, 51.0)
        output_path = Path('/tmp/test_soil.tif')
        result = self.client.get_soil_raster_for_watershed(bbox, output_path)

        self.assertTrue(result['success'])
        self.assertEqual(result['soil_file'], str(output_path))
        self.assertEqual(result['sample_values']['clay_percent'], 15.0)
        mock_rasterio_open.assert_called_once()

if __name__ == '__main__':
    unittest.main()
