# My First Python Code
# A simple HRM (Human Resource Management) example

# --- Variables ---
company_name = "Acme Corp"
department = "Engineering"

# --- Employee dictionary ---
employee = {
    "name": "Alice Johnson",
    "age": 30,
    "role": "Software Engineer",
    "salary": 75000,
    "is_full_time": True
}

# --- Function to greet an employee ---
def greet_employee(emp):
    print(f"Welcome to {company_name}, {emp['name']}!")
    print(f"Department : {department}")
    print(f"Role       : {emp['role']}")
    print(f"Full-time  : {emp['is_full_time']}")

# --- Function to calculate annual bonus ---
def calculate_bonus(salary, performance_rating):
    """
    Returns a bonus based on performance rating (1-5).
    5 = excellent, 1 = needs improvement
    """
    bonus_rates = {5: 0.20, 4: 0.15, 3: 0.10, 2: 0.05, 1: 0.02}
    rate = bonus_rates.get(performance_rating, 0)
    return salary * rate

# --- List of employees ---
team = [
    {"name": "Alice Johnson", "salary": 75000, "rating": 5},
    {"name": "Bob Smith",     "salary": 65000, "rating": 3},
    {"name": "Carol White",   "salary": 80000, "rating": 4},
]

# --- Main program ---
print("=" * 40)
print("   HR Management System - Demo")
print("=" * 40)

greet_employee(employee)

print("\n--- Team Bonus Report ---")
for member in team:
    bonus = calculate_bonus(member["salary"], member["rating"])
    print(f"{member['name']:<15} | Rating: {member['rating']} | Bonus: ${bonus:,.2f}")

print("\nDone! Python is awesome.")
