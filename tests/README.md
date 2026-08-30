# FinVet Test Suite

## Quick Start

### Install Test Dependencies
```bash
pip install pytest pytest-asyncio pytest-cov httpx faker
```

### Run All Tests
```bash
# From project root
pytest tests/ -v

# With coverage
pytest tests/ --cov=src/finvet --cov-report=html

# Run specific test file
pytest tests/unit/test_database_connection.py -v

# Run specific test
pytest tests/unit/test_api_keys.py::test_generate_api_key_format -v
```

### Run Tests by Category
```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Fast tests (skip slow integration)
pytest tests/ -m "not slow" -v
```

## Test Structure

```
tests/
├── conftest.py              # Pytest configuration & fixtures
├── unit/                    # Unit tests (fast, isolated)
│   ├── test_database_connection.py
│   ├── test_api_keys.py
│   └── ...
├── integration/             # Integration tests (slower)
│   ├── test_api_verify.py
│   └── ...
├── performance/             # Performance & load tests
├── accuracy/                # Accuracy benchmark tests
└── security/                # Security & penetration tests
```

## Current Tests

### ✅ Database Connection Tests
- PostgreSQL connection health
- Connection pooling
- Table existence verification

### ✅ API Key Tests  
- Key generation format
- Hash determinism
- Format validation
- Uniqueness & entropy

## Next Steps

See **TEST_PLAN.md** for comprehensive test plan covering:
- Authentication tests
- Agent tests (SEC, News, Market)
- Graph/workflow tests
- API endpoint tests
- Performance tests
- Accuracy tests
- Security tests

## Writing New Tests

### Example Unit Test
```python
def test_my_feature():
    """Test description."""
    from src.finvet.module import my_function
    
    result = my_function(input_data)
    
    assert result == expected_output
```

### Example Integration Test
```python
@pytest.mark.asyncio
async def test_api_endpoint(test_api_key):
    """Test API endpoint."""
    from httpx import AsyncClient
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/verify",
            json={"claim": "test claim"},
            headers={"Authorization": f"Bearer {test_api_key}"}
        )
    
    assert response.status_code == 200
```

## Test Fixtures

Available fixtures from `conftest.py`:
- `test_db_engine` - Test database engine
- `test_db_session` - Database session for test
- `test_api_key` - Valid API key for testing
- `sample_claim` - Sample claim text

## CI/CD

Tests run automatically on:
- Every commit (GitHub Actions)
- Pull requests
- Before deployment

Minimum requirements:
- All tests must pass
- Code coverage > 70% (measured 72% on 2026-08-25)
- No security vulnerabilities

## Troubleshooting

**Tests fail with database error:**
```bash
# Make sure PostgreSQL is running
docker-compose up -d
```

The schema is created on API startup — `init_db()` runs in the lifespan hook,
so there is no separate initialisation step.

**Import errors:**
```bash
# Make sure you're in the virtual environment
source .venv/bin/activate

# Install in development mode
pip install -e .
```
