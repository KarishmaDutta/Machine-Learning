import pandas as pd
import numpy as np

data = {
    "Student Name": ["Rahul", "Pritam", "Samrat", "Anubhav", "Ankit"],
    "Roll Number": [34, 33, 43, 11, 7],
    "Marks": [72, 85, 91, 77, 68],
    "Attendance": [88, 92, 76, 95, 81]
}

df = pd.DataFrame(data)

conditions = [
    df["Marks"] >= 90,
    df["Marks"] >= 80,
    df["Marks"] >= 70
]

grades = ["A", "B", "C"]

df["Grade"] = np.select(conditions, grades, default="D")

print(df)
