# need to add this to be able to import the different python files
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import api
import requests

def get_image_from_wikipedia():
    request =  requests.get('https://upload.wikimedia.org/wikipedia/commons/2/22/Canada_Search_and_Rescue.jpg')

    if request.status_code == 200:
        return request.content

    return None

def test_create_image_description():
    image = get_image_from_wikipedia()

    if image is None:
        assert False

    description = api.get_description_from_image_with_groq_cloud(image)

    assert description is not None
