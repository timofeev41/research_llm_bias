#!/usr/bin/env python3
"""
Download responses from Firebase Firestore using service account
"""

import json
import os
from datetime import datetime

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    print("Firebase Admin SDK not installed. Install with: pip install firebase-admin")
    DEPENDENCIES_AVAILABLE = False

def serialize_for_json(obj):
    """Convert Firebase objects to JSON-serializable format"""
    if hasattr(obj, 'isoformat'):  # datetime objects
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):  # custom objects like DatetimeWithNanoseconds
        # Try to get a string representation
        try:
            return str(obj)
        except:
            return repr(obj)
    elif isinstance(obj, dict):
        # Handle nested dictionaries
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Handle nested lists
        return [serialize_for_json(item) for item in obj]
    else:
        return obj

def init_firestore():
    if not DEPENDENCIES_AVAILABLE:
        raise Exception("Firebase dependencies not available")

    service_account_file = "SERVICE_ACC_JSON_HERE.json"

    try:
        # Check if service account file exists
        if not os.path.exists(service_account_file):
            raise FileNotFoundError(f"Service account file not found: {service_account_file}")

        # Initialize Firebase Admin SDK
        cred = credentials.Certificate(service_account_file)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)

        return firestore.client()

    except Exception as e:
        print(f"Error initializing Firebase: {e}")
        raise

def serialize_for_json(obj):
    """Convert Firebase objects to JSON-serializable format"""
    if hasattr(obj, 'isoformat'):  # datetime objects
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):  # custom objects like DatetimeWithNanoseconds
        # Try to get a string representation
        try:
            return str(obj)
        except:
            return repr(obj)
    elif isinstance(obj, dict):
        # Handle nested dictionaries
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Handle nested lists
        return [serialize_for_json(item) for item in obj]
    else:
        return obj

def download_responses(db, collection_name="responses"):
    """Download all documents from the specified collection"""
    print(f"Downloading documents from collection: {collection_name}")

    try:
        docs = db.collection(collection_name).stream()
        responses = []

        for doc in docs:
            doc_data = doc.to_dict()
            doc_data["_doc_id"] = doc.id

            # Convert any non-serializable objects
            for key, value in doc_data.items():
                if hasattr(value, 'isoformat') or hasattr(value, '__dict__'):
                    doc_data[key] = serialize_for_json(value)

            responses.append(doc_data)
            print(f"Downloaded document: {doc.id}")

        print(f"Total documents downloaded: {len(responses)}")
        return responses

    except Exception as e:
        print(f"Error downloading documents: {e}")
        raise

def save_to_files(responses, base_dir="responses"):
    """Save each response to separate JSON file named by taskId"""
    import os

    # Create directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)

    saved_files = []

    for response in responses:
        task_id = response.get("taskId", "unknown")
        doc_id = response.get("_doc_id", "unknown")

        filename = f"{base_dir}/{task_id}.json"

        # Ensure data is JSON serializable
        serializable_data = serialize_for_json(response)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)

        saved_files.append(filename)
        print(f"Saved: {filename} (doc: {doc_id})")

    print(f"\nTotal files saved: {len(saved_files)}")
    return saved_files

def save_to_file(data, filename=None):
    """Save data to JSON file (legacy function for compatibility)"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"responses_{timestamp}.json"

    # Ensure all data is JSON serializable
    serializable_data = serialize_for_json(data)

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)

    print(f"Data saved to: {filename}")
    return filename

def main():
    if not DEPENDENCIES_AVAILABLE:
        print("Please install Firebase Admin SDK first:")
        print("pip install firebase-admin")
        return

    try:
        # Initialize Firestore
        db = init_firestore()

        # Download responses
        responses = download_responses(db, "survey_results")

        if responses:
            # Save each response to separate file
            saved_files = save_to_files(responses)
            print(f"\nDownloaded {len(responses)} responses successfully!")
            print("Files saved in 'responses/' directory")

            # Show summary
            task_ids = set()
            for resp in responses:
                if "taskId" in resp:
                    task_ids.add(resp["taskId"])

            print(f"Task IDs found: {sorted(task_ids)}")
        else:
            print("No documents found in the 'responses' collection.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()