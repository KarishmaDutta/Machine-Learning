import pandas as pd

students = pd.DataFrame({
    "Student Name": ["Amit", "Riya", "Sourav", "Neha", "Rahul"],
    "Roll Number": [15, 17, 19, 20, 40],
    "Marks": [72, 85, 68, 91, 77],
    "Attendance": [88, 92, 76, 95, 81]
})

result = students.query("Marks > 80")

print("Students with Marks > 80:")
print(result)
