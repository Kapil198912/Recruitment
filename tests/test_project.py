import re
import unittest
from collections import Counter

from dataset import (
    CATEGORIES,
    JOB_APPLICATIONS,
    STATUSES,
    generate_job_applications,
)
from rag_pipeline import load_documents, sentence_chunks


class DatasetTests(unittest.TestCase):
    def test_seeded_records_cover_required_values_and_ranges(self):
        self.assertEqual(JOB_APPLICATIONS, generate_job_applications())
        self.assertGreaterEqual(len(JOB_APPLICATIONS), 40)
        self.assertEqual(len({record["record_id"] for record in JOB_APPLICATIONS}), len(JOB_APPLICATIONS))
        category_counts = Counter(record["category"] for record in JOB_APPLICATIONS)
        status_counts = Counter(record["status"] for record in JOB_APPLICATIONS)
        self.assertTrue(all(category_counts[category] >= 3 for category in CATEGORIES))
        self.assertTrue(all(status_counts[status] >= 1 for status in STATUSES))
        for record in JOB_APPLICATIONS:
            self.assertEqual(
                set(record),
                {
                    "record_id",
                    "category",
                    "status",
                    "expected_salary_inr",
                    "days_since_created",
                    "flagged_priority_review",
                },
            )
            self.assertIn(record["category"], CATEGORIES)
            self.assertIn(record["status"], STATUSES)
            self.assertGreaterEqual(record["expected_salary_inr"], 300_000)
            self.assertLessEqual(record["expected_salary_inr"], 4_200_000)
            self.assertIsInstance(record["days_since_created"], int)
            self.assertGreaterEqual(record["days_since_created"], 0)
            self.assertLessEqual(record["days_since_created"], 30)
            self.assertIsInstance(record["flagged_priority_review"], bool)
        flagged_percentage = (
            100
            * sum(record["flagged_priority_review"] for record in JOB_APPLICATIONS)
            / len(JOB_APPLICATIONS)
        )
        self.assertGreaterEqual(flagged_percentage, 10)
        self.assertLessEqual(flagged_percentage, 30)


class KnowledgeBaseTests(unittest.TestCase):
    def test_required_documents_have_two_to_five_sentences(self):
        documents = load_documents()
        self.assertGreaterEqual(len(documents), 12)
        for document in documents:
            body = "\n".join(
                line
                for line in document.text.splitlines()
                if not line.lstrip().startswith("#")
            )
            sentences = [
                sentence
                for sentence in re.split(r"(?<=[.!?])\s+", body.strip())
                if sentence.strip()
            ]
            self.assertGreaterEqual(len(sentences), 2, document.doc_id)
            self.assertLessEqual(len(sentences), 5, document.doc_id)
        chunks = sentence_chunks(documents)
        self.assertTrue(all(chunk.doc_id in {doc.doc_id for doc in documents} for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
