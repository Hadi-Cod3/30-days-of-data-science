import numpy as np

rng = np.random.default_rng(seed=2)
scores = rng.integers(low=0, high=100, size=(30, 5))
courses = ["Maths", "English", "Physics", "Chemistry", "Biology"]
# Averages
student_avg = np.mean(scores, axis=1)
course_avg = np.mean(scores, axis=0)
# Best per course, and top student overall
best_scores = scores.max(axis=0)
best_students = scores.argmax(axis=0) + 1
top_student = student_avg.argmax() + 1
# Students with at least one score below 50
below_fifty = scores < 50
below_fifty_count = np.count_nonzero(below_fifty, axis=1)
students_below_fifty = np.count_nonzero(below_fifty_count > 0)
check = np.count_nonzero(below_fifty.any(axis=1))
# +5 bonus to one course, capped at 100
bonus_col = 1
curved = scores.copy()
curved[:, bonus_col] = np.minimum(scores[:, bonus_col] + 5, 100)
ranks = (-student_avg).argsort().argsort() + 1

print("Shape:", scores.shape)
print("Student averages:", student_avg.round(1))
print("Course averages:")
for c, avg in zip(courses, course_avg):
    print(f"  {c}: {avg:.1f}")
for c, s, who in zip(courses, best_scores, best_students):
    print(f"Best in {c}: {s} (student {who})")
print("Top student overall:", top_student)
print("Students with a score below 50:", students_below_fifty, "| check:", check)
print("Student with most low scores:", below_fifty_count.argmax() + 1)
print("Bonus course before:", scores[:, bonus_col])
print("Bonus course after: ", curved[:, bonus_col])
print("Ranks:", ranks)
