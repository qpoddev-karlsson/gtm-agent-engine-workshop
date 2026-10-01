import copy
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LANGSMITH_TRACING", "false")

from gtm_agent import data_service
from gtm_agent import gtm_agent


class ProspectFlowTests(unittest.TestCase):
    def setUp(self):
        self.original_stacks = {
            prospect_id: copy.deepcopy(data_service.PROSPECTS[prospect_id]["tech_stack"])
            for prospect_id in ("LEAD-39002", "LEAD-71001")
        }
        data_service._PROFILES.clear()

    def tearDown(self):
        for prospect_id, tech_stack in self.original_stacks.items():
            data_service.PROSPECTS[prospect_id]["tech_stack"] = tech_stack
        data_service._PROFILES.clear()

    def _score_with_captured_input(self, profile, offering_id):
        captured = {}

        def invoke(messages):
            captured["profile"] = messages[1]["content"].split(
                "\n\nProspect profile:\n", 1
            )[1]
            return SimpleNamespace(
                model_dump=lambda: {
                    "score": 100,
                    "justification": "All required technologies are present.",
                    "rubric_breakdown": {
                        "revenue_fit": 100,
                        "tech_stack_match": 100,
                        "segment_fit": 100,
                    },
                }
            )

        fake_llm = SimpleNamespace(invoke=invoke)
        with patch.object(gtm_agent, "_scoring_llm", fake_llm):
            result = gtm_agent.score_prospect.func(
                profile, data_service.get_offering(offering_id)
            )
        return result, captured["profile"]

    def test_terraform_update_is_persisted_and_used_for_scoring(self):
        update = data_service.update_prospect_info("LEAD-39002", "Terraform")
        profile = gtm_agent.build_prospect_profile.func("LEAD-39002")
        result, serialized_profile = self._score_with_captured_input(
            profile["prospect_profile"], "OFFER-10004"
        )

        self.assertTrue(update["updated"])
        self.assertIn("Terraform", update["tech_stack"])
        self.assertIn("Terraform", profile["prospect_profile"]["tech_stack"])
        self.assertIn("Terraform", serialized_profile)
        self.assertEqual(result["score"], 100)

    def test_kafka_update_is_persisted_and_used_for_scoring(self):
        update = data_service.update_prospect_info("LEAD-71001", "Kafka")
        profile = gtm_agent.build_prospect_profile.func("LEAD-71001")
        result, serialized_profile = self._score_with_captured_input(
            profile["prospect_profile"], "OFFER-10005"
        )

        self.assertTrue(update["updated"])
        self.assertIn("Kafka", update["tech_stack"])
        self.assertIn("Kafka", profile["prospect_profile"]["tech_stack"])
        self.assertIn("Kafka", serialized_profile)
        self.assertEqual(result["score"], 100)

    def test_scoring_preserves_supplied_tech_stack(self):
        profile = {
            "prospect_id": "LEAD-39002",
            "tech_stack": ["Terraform"],
        }

        _, serialized_profile = self._score_with_captured_input(
            profile, "OFFER-10004"
        )

        self.assertIn('"Terraform"', serialized_profile)


if __name__ == "__main__":
    unittest.main()
