import random

# Define templates for delivery reviews
templates = [
    "My package arrived {adjective} because {reason}.",
    "The delivery was {adjective} and the packaging was {condition}.",
    "I was very {emotion} with the delivery as it was {adjective} and {reason}.",
]

# Define possible values for placeholders
adjectives = ["late", "on time", "delayed", "early", "unexpectedly late"]
reasons = [
    "due to heavy traffic",
    "because of weather issues",
    "due to a logistical error",
    "as per schedule"
]
conditions = ["damaged", "intact", "poor", "excellent"]
emotions = ["disappointed", "satisfied", "frustrated", "pleased"]

def generate_synthetic_review():
    template = random.choice(templates)
    review = template.format(
        adjective=random.choice(adjectives),
        reason=random.choice(reasons),
        condition=random.choice(conditions),
        emotion=random.choice(emotions)
    )
    return review

# Generate 6000 synthetic delivery reviews
synthetic_delivery_reviews = [generate_synthetic_review() for _ in range(6000)]

# Optionally, write synthetic reviews to a file
with open("synthetic_delivery_reviews.txt", "w", encoding="utf-8") as f:
    for review in synthetic_delivery_reviews:
        f.write(review + "\n")

print("Generated and saved 6000 synthetic delivery reviews.")
