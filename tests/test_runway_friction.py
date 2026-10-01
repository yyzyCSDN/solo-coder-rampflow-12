import unittest
from datetime import datetime, timezone, timedelta

from rampflow.runway_friction import (
    FrictionReading,
    RunwayFrictionOffice,
    classify_mu,
    action_code,
)
from rampflow.weather import WeatherObservation

UTC = timezone.utc
D = lambda h, m=0: datetime(2026, 1, 1, h, m, tzinfo=UTC)


def obs(at, braking="good"):
    return WeatherObservation("APT", at, 270, 10.0, None, 9000, 1500, -2.0, None, braking)


def reading(rid, segment, vehicle, at, mu, runway="09L"):
    return FrictionReading(rid, runway, segment, vehicle, at, mu)


def full_coverage(office, at, mu_by_segment=None, runway="09L"):
    mu_by_segment = mu_by_segment or {"touchdown": 0.42, "midpoint": 0.41, "rollout": 0.43}
    for i, (segment, mu) in enumerate(sorted(mu_by_segment.items())):
        office.ingest(reading(f"R{i}a", segment, "V1", at, mu, runway))
        office.ingest(reading(f"R{i}b", segment, "V2", at + timedelta(minutes=1), mu + 0.02, runway))


class RunwayFrictionTests(unittest.TestCase):
    def test_mu_classification_bands(self):
        self.assertEqual(classify_mu(0.45), 5)
        self.assertEqual(classify_mu(0.37), 4)
        self.assertEqual(classify_mu(0.33), 3)
        self.assertEqual(classify_mu(0.27), 2)
        self.assertEqual(classify_mu(0.20), 1)
        self.assertEqual(action_code("good_to_medium"), 4)
        self.assertIsNone(action_code("bogus"))

    def test_publish_combines_vehicles_segments_and_weather(self):
        office = RunwayFrictionOffice()
        full_coverage(office, D(10))
        report = office.publish("RCR-1", "09L", D(10, 20), obs(D(10, 15)))
        self.assertEqual(report.revision, 1)
        self.assertEqual(report.assessment.overall_code, 5)
        self.assertEqual(len(report.assessment.segments), 3)
        basis = " ".join(report.assessment.basis)
        self.assertIn("V1", basis)
        self.assertIn("V2", basis)
        self.assertIn("worst segment governs", basis)

    def test_uncovered_segment_blocks_publication(self):
        office = RunwayFrictionOffice()
        office.ingest(reading("R1", "touchdown", "V1", D(10), 0.42))
        office.ingest(reading("R2", "midpoint", "V1", D(10), 0.41))
        assessment = office.assess("09L", D(10, 20), obs(D(10, 15)))
        self.assertFalse(assessment.publishable)
        self.assertIsNone(assessment.overall_code)
        self.assertTrue(any(h.startswith("segment-uncovered: rollout") for h in assessment.holds))
        with self.assertRaises(RuntimeError):
            office.publish("RCR-2", "09L", D(10, 20), obs(D(10, 15)))

    def test_conflicting_readings_block_publication(self):
        office = RunwayFrictionOffice()
        full_coverage(office, D(10))
        office.ingest(reading("RX", "midpoint", "V3", D(10, 2), 0.10))
        assessment = office.assess("09L", D(10, 20), obs(D(10, 15)))
        self.assertFalse(assessment.publishable)
        conflict = [h for h in assessment.holds if h.startswith("segment-conflict: midpoint")]
        self.assertEqual(len(conflict), 1)
        self.assertIn("V3=0.10", conflict[0])
        with self.assertRaises(RuntimeError):
            office.publish("RCR-3", "09L", D(10, 20), obs(D(10, 15)))

    def test_weather_downgrades_mu_based_code(self):
        office = RunwayFrictionOffice()
        full_coverage(office, D(10))
        assessment = office.assess("09L", D(10, 20), obs(D(10, 15), braking="poor"))
        self.assertTrue(assessment.publishable)
        self.assertEqual(assessment.overall_code, 1)
        self.assertIn("downgrades", " ".join(assessment.basis))

    def test_stale_or_missing_weather_blocks_publication(self):
        office = RunwayFrictionOffice()
        full_coverage(office, D(10))
        stale = office.assess("09L", D(12), obs(D(10, 15)))
        self.assertTrue(any(h.startswith("weather-stale") for h in stale.holds))
        missing = office.assess("09L", D(10, 20))
        self.assertTrue(any(h.startswith("weather-unavailable") for h in missing.holds))

    def test_stale_reading_blocks_publication(self):
        office = RunwayFrictionOffice(freshness_min=30)
        full_coverage(office, D(10))
        assessment = office.assess("09L", D(11), obs(D(11)))
        self.assertFalse(assessment.publishable)
        self.assertTrue(any(h.startswith("segment-stale") for h in assessment.holds))

    def test_late_data_issues_revision_without_rewriting_history(self):
        office = RunwayFrictionOffice()
        full_coverage(office, D(10))
        first = office.publish("RCR-4", "09L", D(10, 20), obs(D(10, 15)))
        self.assertEqual(first.assessment.overall_code, 5)
        # Second survey round: both vehicles re-measure the rollout third as ice forms.
        office.ingest(reading("L1", "rollout", "V1", D(10, 25), 0.24))
        office.ingest(reading("L2", "rollout", "V2", D(10, 26), 0.25))
        second = office.revise("RCR-4", D(10, 40), obs(D(10, 35)), note="late rollout survey")
        self.assertEqual(second.revision, 2)
        self.assertEqual(second.supersedes, 1)
        self.assertEqual(second.assessment.overall_code, 1)
        self.assertIn("superseded", " ".join(second.assessment.basis))
        history = office.history("RCR-4")
        self.assertEqual([r.revision for r in history], [1, 2])
        self.assertEqual(history[0].assessment.overall_code, 5)
        self.assertIs(office.latest("RCR-4"), second)
        with self.assertRaises(RuntimeError):
            office.publish("RCR-4", "09L", D(11), obs(D(11)))

    def test_revision_refused_when_evidence_still_conflicts(self):
        office = RunwayFrictionOffice()
        full_coverage(office, D(10))
        office.publish("RCR-5", "09L", D(10, 20), obs(D(10, 15)))
        office.ingest(reading("C1", "midpoint", "V3", D(10, 30), 0.05))
        with self.assertRaises(RuntimeError):
            office.revise("RCR-5", D(10, 40), obs(D(10, 35)))
        self.assertEqual(office.latest("RCR-5").revision, 1)

    def test_vehicle_resurvey_supersedes_own_earlier_reading(self):
        office = RunwayFrictionOffice()
        office.ingest(reading("R1", "touchdown", "V1", D(10), 0.44))
        office.ingest(reading("R2", "touchdown", "V1", D(10, 20), 0.22))
        assessment = office.assess("09L", D(10, 30), obs(D(10, 25)))
        holds = [h for h in assessment.holds if h.startswith("segment-conflict")]
        self.assertEqual(holds, [])
        touchdown = next(s for s in assessment.segments if s.segment == "touchdown")
        self.assertEqual(touchdown.code, 1)
        self.assertEqual(touchdown.reading_ids, ("R2",))

    def test_ingest_idempotent_but_payload_immutable(self):
        office = RunwayFrictionOffice()
        a = office.ingest(reading("R1", "touchdown", "V1", D(10), 0.42))
        b = office.ingest(reading("R1", "touchdown", "V1", D(10), 0.42))
        self.assertIs(a, b)
        with self.assertRaises(ValueError):
            office.ingest(reading("R1", "touchdown", "V1", D(10), 0.30))
        with self.assertRaises(ValueError):
            office.ingest(reading("R9", "threshold", "V1", D(10), 0.40))


if __name__ == "__main__":
    unittest.main()
