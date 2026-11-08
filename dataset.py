"""Create a reproducible synthetic job-application dataset."""

from collections import Counter
import random


SEED = 20261104
CATEGORIES = (
    "Software Engineer",
    "Data Analyst",
    "Product Manager",
    "HR Executive",
    "Sales Associate",
)
STATUSES = (
    "Applied",
    "Screening",
    "Interview Scheduled",
    "Offered",
    "Rejected",
)

_SALARY_RANGES_INR = {
    "Software Engineer": (600_000, 3_600_000),
    "Data Analyst": (450_000, 2_400_000),
    "Product Manager": (900_000, 4_200_000),
    "HR Executive": (360_000, 1_800_000),
    "Sales Associate": (300_000, 1_500_000),
}
_FLAG_PROBABILITY = 0.20
_RECORD_COUNT_PER_CATEGORY = 10


def generate_job_applications(seed: int = SEED) -> list[dict[str, object]]:
    """Return deterministic, category-balanced synthetic application records."""
    rng = random.Random(seed)
    records: list[dict[str, object]] = []
    application_number = 1

    for category in CATEGORIES:
        minimum_salary, maximum_salary = _SALARY_RANGES_INR[category]
        statuses = list(STATUSES) * (
            _RECORD_COUNT_PER_CATEGORY // len(STATUSES)
        )
        rng.shuffle(statuses)

        for status in statuses:
            records.append(
                {
                    "record_id": f"JAPP-{application_number:04d}",
                    "category": category,
                    "status": status,
                    "expected_salary_inr": rng.randrange(
                        minimum_salary, maximum_salary + 1, 10_000
                    ),
                    "days_since_created": rng.randint(0, 30),
                    "flagged_priority_review": rng.random() < _FLAG_PROBABILITY,
                }
            )
            application_number += 1

    rng.shuffle(records)
    return records


JOB_APPLICATIONS = generate_job_applications()


def print_report(records: list[dict[str, object]] = JOB_APPLICATIONS) -> None:
    """Print dataset counts and the priority-review percentage."""
    category_counts = Counter(str(record["category"]) for record in records)
    status_counts = Counter(str(record["status"]) for record in records)
    flagged_count = sum(
        record["flagged_priority_review"] is True for record in records
    )
    percentage = 100 * flagged_count / len(records)

    print(f"Total records: {len(records)}")
    print("Count per category:")
    for category in CATEGORIES:
        print(f"  {category}: {category_counts[category]}")
    print("Count per status:")
    for status in STATUSES:
        print(f"  {status}: {status_counts[status]}")
    print(
        "Priority review: "
        f"{flagged_count}/{len(records)} = {percentage:.1f}%"
    )


if __name__ == "__main__":
    print_report()
