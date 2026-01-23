"""Quick test to verify holidays loading works correctly."""
from config import load_holidays

# Test loading both types
model_holidays = load_holidays(holiday_type="model")
business_holidays = load_holidays(holiday_type="business")

print(f"Model holidays years: {sorted(model_holidays.keys())}")
print(f"Business holidays years: {sorted(business_holidays.keys())}")
print(f"2025 Tet (model): {model_holidays[2025]['tet'][:3]}")
print(f"2025 Tet (business): {business_holidays[2025]['tet'][:3]}")
print(f"2026 in model holidays: {2026 in model_holidays}")
print(f"2026 in business holidays: {2026 in business_holidays}")
