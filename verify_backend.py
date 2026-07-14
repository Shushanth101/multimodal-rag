import os
import sys

# Ensure backend can be imported from root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health():
    print("Testing /api/health...")
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    print("Health check: OK")

def test_list_files():
    print("Testing /api/files...")
    response = client.get("/api/files")
    assert response.status_code == 200
    print("Files list retrieved:", response.json())
    print("List files check: OK")

def test_static_ui():
    print("Testing static root serve...")
    response = client.get("/")
    assert response.status_code == 200
    print("Static UI serve check: OK")

if __name__ == "__main__":
    print("Running automated backend checks...")
    try:
        test_health()
        test_list_files()
        test_static_ui()
        print("\nAll automated integration checks PASSED successfully!")
    except AssertionError as e:
        print("\nAssertion error during checks:", e)
        sys.exit(1)
    except Exception as e:
        print("\nUnexpected error during checks:", e)
        sys.exit(1)
