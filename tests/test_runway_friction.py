import unittest
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone, timedelta

from rampflow.runway_friction import (
    BrakingAdvisory,
    AdvisoryError,
    BrakingAdvisoryLedger,
    BrakingAssessor,
    BrakingCondition,
    FrictionReading,
    RunwayThird as T,
    RunwayWeather,
    WeatherState as W,
    REASON_MISSING_SEGMENT,
    REASON_MEASUREMENT_SPAN,
    REASON_READING_CONFLICT,
    REASON_WEATHER_CONTEXT,
    REASON_WEATHER_CONFLICT,
)

UTC = timezone.utc
D = lambda h, m=0: datetime(2026, 1, 15, h, m, tzinfo=UTC)


def reading(third, vehicle, mu, at, weather=W.RAIN, runway="27L"):
    return FrictionReading(
        runway_id=runway,
        third=third,
        vehicle_id=vehicle,
        measured_at=at,
        mu=mu,
        weather_state=weather,
    )


def full_set(rain_at=D(8, 0)):
    """Two vehicles, three thirds, consistent wet-runway readings."""
    return [
        reading(T.TOUCHDOWN, "truck-A", 0.68, D(8, 2)),
        reading(T.MIDPOINT, "truck-A", 0.66, D(8, 4)),
        reading(T.ROLLOUT, "truck-A", 0.65, D(8, 6)),
        reading(T.TOUCHDOWN, "truck-B", 0.70, D(8, 3)),
        reading(T.MIDPOINT, "truck-B", 0.67, D(8, 5)),
        reading(T.ROLLOUT, "truck-B", 0.64, D(8, 7)),
    ], [RunwayWeather("27L", rain_at, W.RAIN, "steady rain")]


class AssessmentTests(unittest.TestCase):
    def test_consistent_evidence_publishes_worst_third(self):
        assessor = BrakingAssessor()
        rows, weather = full_set()
        result = assessor.assess("27L", rows, weather)

        self.assertTrue(result.publishable)
        self.assertEqual(result.withhold_reasons, ())
        self.assertEqual(result.overall_condition, BrakingCondition.MEDIUM)
        self.assertEqual(len(result.segments), 3)
        self.assertIn("PUBLISH", result.explanation)
        for seg in result.segments:
            self.assertEqual(seg.weather_state, W.RAIN)
            self.assertEqual(len(seg.vehicles), 2)

    def test_missing_third_withholds_without_forcing_conclusion(self):
        assessor = BrakingAssessor()
        rows, weather = full_set()
        rows = [r for r in rows if r.third is not T.ROLLOUT]
        result = assessor.assess("27L", rows, weather)

        self.assertFalse(result.publishable)
        self.assertIsNone(result.overall_condition)
        self.assertIn(REASON_MISSING_SEGMENT, result.withhold_reasons)
        self.assertIn("WITHHELD", result.explanation)
        self.assertIn("rollout", result.explanation)

    def test_disagreeing_readings_withhold(self):
        assessor = BrakingAssessor()
        rows, weather = full_set()
        # One vehicle reports near-good braking, the other poor, same third.
        rows.append(reading(T.TOUCHDOWN, "truck-C", 0.30, D(8, 3)))
        result = assessor.assess("27L", rows, weather)

        self.assertFalse(result.publishable)
        self.assertIn(REASON_READING_CONFLICT, result.withhold_reasons)
        td = next(s for s in result.segments if s.third is T.TOUCHDOWN)
        self.assertGreater(td.mu_spread, 0.10)

    def test_mu_spread_within_tolerance_but_categories_apart_withholds(self):
        # Tight category gap with mu spread allowed proves the two checks are
        # independent: same third, mu close, but classifications straddle a
        # cut-off (0.71 good-to-medium vs 0.69 medium on the default profile).
        rows = [
            reading(T.TOUCHDOWN, "truck-X", 0.71, D(8, 2)),
            reading(T.TOUCHDOWN, "truck-Y", 0.69, D(8, 3)),
            reading(T.MIDPOINT, "truck-X", 0.67, D(8, 4)),
            reading(T.MIDPOINT, "truck-Y", 0.66, D(8, 5)),
            reading(T.ROLLOUT, "truck-X", 0.65, D(8, 6)),
            reading(T.ROLLOUT, "truck-Y", 0.64, D(8, 7)),
        ]
        weather = [RunwayWeather("27L", D(8, 0), W.RAIN)]
        self.assertTrue(BrakingAssessor().assess("27L", rows, weather).publishable)
        strict = BrakingAssessor(max_mu_spread=0.20, max_category_gap=0)
        result = strict.assess("27L", rows, weather)
        self.assertFalse(result.publishable)
        self.assertIn(REASON_READING_CONFLICT, result.withhold_reasons)

    def test_weather_context_required_per_reading(self):
        assessor = BrakingAssessor()
        rows, _ = full_set()
        for r in rows:
            object.__setattr__(r, "weather_state", W.UNKNOWN)
        result = assessor.assess("27L", rows, weather=[])
        self.assertFalse(result.publishable)
        self.assertIn(REASON_WEATHER_CONTEXT, result.withhold_reasons)

    def test_weather_observation_supplies_context(self):
        assessor = BrakingAssessor()
        rows, _ = full_set()
        for r in rows:
            object.__setattr__(r, "weather_state", W.UNKNOWN)
        weather = [RunwayWeather("27L", D(7, 50), W.RAIN)]
        result = assessor.assess("27L", rows, weather)
        self.assertTrue(result.publishable)

    def test_friction_better_than_weather_supports_is_contradiction(self):
        assessor = BrakingAssessor()
        rows, _ = full_set()
        for r in rows:
            object.__setattr__(r, "weather_state", W.ICE)
        weather = [RunwayWeather("27L", D(8, 0), W.ICE, "glaze ice")]
        result = assessor.assess("27L", rows, weather)
        self.assertFalse(result.publishable)
        self.assertIn(REASON_WEATHER_CONFLICT, result.withhold_reasons)

    def test_friction_worse_than_weather_downgrades_and_publishes(self):
        # Rain baseline allows good-to-medium; poor readings downgrade legally.
        assessor = BrakingAssessor()
        rows = [
            reading(T.TOUCHDOWN, "truck-A", 0.44, D(8, 2)),
            reading(T.MIDPOINT, "truck-A", 0.42, D(8, 4)),
            reading(T.ROLLOUT, "truck-A", 0.41, D(8, 6)),
        ]
        weather = [RunwayWeather("27L", D(8, 0), W.RAIN)]
        result = assessor.assess("27L", rows, weather)
        self.assertTrue(result.publishable)
        self.assertEqual(result.overall_condition, BrakingCondition.POOR)

    def test_reading_span_too_wide_withholds(self):
        assessor = BrakingAssessor(max_window_minutes=30)
        rows = [
            reading(T.TOUCHDOWN, "truck-A", 0.66, D(8, 0)),
            reading(T.MIDPOINT, "truck-A", 0.66, D(8, 10)),
            reading(T.ROLLOUT, "truck-A", 0.66, D(8, 45)),
        ]
        result = assessor.assess("27L", rows)
        self.assertFalse(result.publishable)
        self.assertIn(REASON_MEASUREMENT_SPAN, result.withhold_reasons)

    def test_reading_validation(self):
        with self.assertRaises(ValueError):
            FrictionReading("27L", T.TOUCHDOWN, "x", D(8), 1.5)
        with self.assertRaises(ValueError):
            FrictionReading("", T.TOUCHDOWN, "x", D(8), 0.5)

    def test_assessor_requires_single_runway(self):
        assessor = BrakingAssessor()
        rows, _ = full_set()
        rows.append(reading(T.ROLLOUT, "truck-D", 0.6, D(8, 7), runway="09R"))
        with self.assertRaises(ValueError):
            assessor.assess("27L", rows)

    def test_empty_readings_rejected(self):
        with self.assertRaises(ValueError):
            BrakingAssessor().assess("27L", [])


class AdvisoryLedgerTests(unittest.TestCase):
    def setUp(self):
        self.assessor = BrakingAssessor()
        self.ledger = BrakingAdvisoryLedger()

    def test_issue_and_lookup(self):
        rows, weather = full_set()
        assessment = self.assessor.assess("27L", rows, weather)
        advisory = self.ledger.issue("27L-BRAKE", assessment, D(8, 15))

        self.assertEqual(advisory.version, 1)
        self.assertEqual(advisory.condition, BrakingCondition.MEDIUM)
        self.assertFalse(advisory.is_revision)
        self.assertIn("medium", advisory.text)
        self.assertIs(self.ledger.current("27L-BRAKE"), advisory)

    def test_withheld_assessment_cannot_be_issued(self):
        rows, weather = full_set()
        rows = [r for r in rows if r.third is not T.ROLLOUT]
        assessment = self.assessor.assess("27L", rows, weather)
        with self.assertRaises(AdvisoryError):
            self.ledger.issue("27L-BRAKE", assessment, D(8, 15))

    def test_duplicate_issue_rejected(self):
        rows, weather = full_set()
        assessment = self.assessor.assess("27L", rows, weather)
        self.ledger.issue("27L-BRAKE", assessment, D(8, 15))
        with self.assertRaises(AdvisoryError):
            self.ledger.issue("27L-BRAKE", assessment, D(8, 20))

    def test_late_data_creates_revision_original_unchanged(self):
        rows, weather = full_set()
        first = self.assessor.assess("27L", rows, weather)
        v1 = self.ledger.issue("27L-BRAKE", first, D(8, 15))
        v1_text = v1.text

        # Supplementary readings for a later run show degraded rollout braking.
        later = [
            reading(T.TOUCHDOWN, "truck-A", 0.58, D(9, 32)),
            reading(T.MIDPOINT, "truck-A", 0.53, D(9, 34)),
            reading(T.ROLLOUT, "truck-A", 0.51, D(9, 36)),
        ]
        revised_assessment = self.assessor.assess("27L", later, weather)
        v2 = self.ledger.revise(
            "27L-BRAKE", revised_assessment, D(9, 45),
            note="supplementary rollout readings",
        )

        self.assertEqual(v2.version, 2)
        self.assertEqual(v2.supersedes_version, 1)
        self.assertTrue(v2.is_revision)
        self.assertEqual(v2.condition, BrakingCondition.MEDIUM_TO_POOR)
        self.assertIs(self.ledger.current("27L-BRAKE"), v2)

        # The already-published advisory is still retrievable and untouched.
        stored_v1 = self.ledger.version("27L-BRAKE", 1)
        self.assertIs(stored_v1, v1)
        self.assertEqual(stored_v1.version, 1)
        self.assertEqual(stored_v1.text, v1_text)
        self.assertEqual(stored_v1.condition, BrakingCondition.MEDIUM)
        self.assertIsNone(stored_v1.supersedes_version)

        history = self.ledger.history("27L-BRAKE")
        self.assertEqual([a.version for a in history], [1, 2])
        self.assertEqual(len(self.ledger.all_current()), 1)

    def test_revision_of_unknown_key_rejected(self):
        rows, weather = full_set()
        assessment = self.assessor.assess("27L", rows, weather)
        with self.assertRaises(AdvisoryError):
            self.ledger.revise("nope", assessment, D(9))

    def test_revision_from_withheld_assessment_rejected(self):
        rows, weather = full_set()
        self.ledger.issue("27L-BRAKE", self.assessor.assess("27L", rows, weather), D(8, 15))
        rows = [r for r in rows if r.third is not T.ROLLOUT]
        withheld = self.assessor.assess("27L", rows, weather)
        with self.assertRaises(AdvisoryError):
            self.ledger.revise("27L-BRAKE", withheld, D(9))

    def test_published_advisory_is_immutable(self):
        rows, weather = full_set()
        advisory = self.ledger.issue(
            "27L-BRAKE", self.assessor.assess("27L", rows, weather), D(8, 15)
        )
        with self.assertRaises(FrozenInstanceError):
            advisory.condition = BrakingCondition.NIL  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            advisory.version = 2  # type: ignore[misc]

    def test_advisory_rejects_bad_supersede_chain(self):
        rows, weather = full_set()
        assessment = self.assessor.assess("27L", rows, weather)
        with self.assertRaises(ValueError):
            BrakingAdvisory(
                advisory_key="k",
                version=1,
                runway_id="27L",
                published_at=D(8, 15),
                effective_from=D(8, 7),
                condition=BrakingCondition.MEDIUM,
                text="x",
                assessment=assessment,
                supersedes_version=2,
            )


if __name__ == "__main__":
    unittest.main()
