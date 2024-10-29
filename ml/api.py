import requests
import json

def get_drone_rules(url_server, drone_ip, drone_secret, mission_name):
    url = f"{url_server}/api/drone/getRules"
    params = {
        "ip": drone_ip,
        "secret": drone_secret,
        "missionName": mission_name
    }

    try:
        response = requests.get(url, params=params)

        # Check for successful response
        if response.status_code == 200:
            rules = response.json()
            return parse_rules(rules)  # Parse the rules after successful retrieval
        elif response.status_code == 403:
            print("Error: Invalid drone ID or secret.")
        elif response.status_code == 400:
            print("Error: Validation error.")
            print(response.json().get('errors'))
        else:
            print("Error: An unexpected error occurred.")

    except requests.exceptions.RequestException as e:
        print("Error making request:", e)

def parse_rules(rules):
    parsed_rules = {}

    for rule in rules:
        # Check if the detectedObject condition contains "person"
        condition = json.loads(rule['condition'])  # Parse the JSON condition string

        # Check if the condition is detected object and the rule is active
        if ("detectedObject" in condition
                and condition["detectedObject"]
                and rule.get("isActive")):
            confidence = condition.get("confidence")  # Get confidence from condition

            # Ensure confidence is valid before adding to parsed rules
            if confidence and isinstance(confidence, dict):
                parsed_rules[rule['id']] = {
                    "object": condition["detectedObject"],
                    "confidence": confidence,
                    "action": rule['action'],
                }

    return parsed_rules
