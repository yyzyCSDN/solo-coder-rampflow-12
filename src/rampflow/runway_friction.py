"""Runway friction measurement fusion and braking-action advisories.

Several friction testers report one reading per runway third after a run in
precipitation.  Operators must fuse those readings with the prevailing runway
weather into an explainable braking assessment.  An advisory is only published
when every third is covered and the evidence agrees; contradictory readings,
missing thirds, or weather/friction mismatch withhold publication instead of
forcing a conclusion.

Published advisories are immutable.  Data that arrives later produces a new
revision with a new version number; the original advisory stays on record.

Threshold values below follow the illustrative bands quoted in ICAO guidance
material for continuous-friction equipment.  Operational sites must configure
their own calibrated device profiles and approved downgrade criteria; the
defaults are laboratory values, not authority-approved minima.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from statistics import median
from typing import Iterable, Mapping, Sequence

from .timeutil import utc


class RunwayThird(str, Enum):
    TOUCHDOWN = "touchdown"
    MIDPOINT = "midpoint"
    ROLLOUT = "rollout"


class BrakingCondition(str, Enum):
    GOOD = "good"
    GOOD_TO_MEDIUM = "good-to-medium"
    MEDIUM = "medium"
    MEDIUM_TO_POOR = "medium-to-poor"
    POOR = "poor"
    NIL = "nil"


# Ordering used when taking the worst segment: NIL is the most restrictive.
BRAKING_ORDER: tuple[BrakingCondition, ...] = (
    BrakingCondition.GOOD,
    BrakingCondition.GOOD_TO_MEDIUM,
    BrakingCondition.MEDIUM,
    BrakingCondition.MEDIUM_TO_POOR,
    BrakingCondition.POOR,
    BrakingCondition.NIL,
)


class WeatherState(str, Enum):
    """Prevailing runway surface/weather regime for the measurement window."""
    DRY = "dry"
    RAIN = "rain"
    SNOW_SLUSH = "snow-or-slush"
    COMPACTED_SNOW = "compacted-snow"
    ICE = "ice"
    UNKNOWN = "unknown"


# Most restrictive braking category that weather alone supports.  Friction
# readings may always downgrade the weather assessment; readings claiming
# meaningfully better braking than the weather allows are contradictory.
WEATHER_BASELINE: Mapping[WeatherState, BrakingCondition] = {
    WeatherState.DRY: BrakingCondition.GOOD,
    WeatherState.RAIN: BrakingCondition.GOOD_TO_MEDIUM,
    WeatherState.SNOW_SLUSH: BrakingCondition.MEDIUM,
    WeatherState.COMPACTED_SNOW: BrakingCondition.MEDIUM_TO_POOR,
    WeatherState.ICE: BrakingCondition.POOR,
    WeatherState.UNKNOWN: BrakingCondition.GOOD,
}


@dataclass(frozen=True)
class RunwayWeather:
    """Weather observed at the runway for one measurement window."""
    runway_id: str
    observed_at: datetime
    state: WeatherState
    note: str = ""

    def __post_init__(self):
        object.__setattr__(self, "observed_at", utc(self.observed_at))
        if not self.runway_id:
            raise ValueError("runway_id required")


@dataclass(frozen=True)
class FrictionReading:
    """One tester's friction coefficient (mu) for one runway third."""
    runway_id: str
    third: RunwayThird
    vehicle_id: str
    measured_at: datetime
    mu: float
    weather_state: WeatherState = WeatherState.UNKNOWN
    def __post_init__(self):
        object.__setattr__(self, "measured_at", utc(self.measured_at))
        if not self.runway_id or not self.vehicle_id:
            raise ValueError("runway_id and vehicle_id required")
        if not 0.0 <= self.mu <= 1.0:
            raise ValueError("friction coefficient must be within 0..1")


@dataclass(frozen=True)
class DeviceProfile:
    """Calibrated mu cut-offs for one continuous-friction device type.

    Boundaries map measured mu to the six reportable braking categories.
    ``good_at_or_above`` is the minimum mu reported as good; values below
    ``nil_below`` are reported as nil braking.
    """
    device_type: str
    good_at_or_above: float
    medium_good_at_or_above: float
    medium_at_or_above: float
    medium_poor_at_or_above: float
    poor_at_or_above: float

    def __post_init__(self):
        bands = (
            self.good_at_or_above,
            self.medium_good_at_or_above,
            self.medium_at_or_above,
            self.medium_poor_at_or_above,
            self.poor_at_or_above,
        )
        if any(not 0.0 <= b <= 1.0 for b in bands):
            raise ValueError("device cut-offs must be within 0..1")
        if any(a <= b for a, b in zip(bands, bands[1:])):
            raise ValueError("device cut-offs must descend from good to poor")

    def classify(self, mu: float) -> BrakingCondition:
        if mu >= self.good_at_or_above:
            return BrakingCondition.GOOD
        if mu >= self.medium_good_at_or_above:
            return BrakingCondition.GOOD_TO_MEDIUM
        if mu >= self.medium_at_or_above:
            return BrakingCondition.MEDIUM
        if mu >= self.medium_poor_at_or_above:
            return BrakingCondition.MEDIUM_TO_POOR
        if mu >= self.poor_at_or_above:
            return BrakingCondition.POOR
        return BrakingCondition.NIL


# Illustrative bands from ICAO guidance for self-wetting continuous-friction
# equipment.  Sites must replace these with their approved calibration.
DEFAULT_DEVICE_PROFILES: Mapping[str, DeviceProfile] = {
    "saab": DeviceProfile("saab", 0.82, 0.74, 0.66, 0.52, 0.38),
    "gripTester": DeviceProfile("gripTester", 0.82, 0.74, 0.66, 0.52, 0.38),
    "mu-meter": DeviceProfile("mu-meter", 0.82, 0.72, 0.60, 0.48, 0.36),
    "default": DeviceProfile("default", 0.80, 0.70, 0.60, 0.50, 0.40),
}


def _rank(condition: BrakingCondition) -> int:
    return BRAKING_ORDER.index(condition)


def _worst(conditions: Iterable[BrakingCondition]) -> BrakingCondition:
    return max(conditions, key=_rank)


@dataclass(frozen=True)
class SegmentEvidence:
    """Fused readings for a single runway third."""
    third: RunwayThird
    vehicles: tuple[str, ...]
    measured_at: tuple[datetime, ...]
    mus: tuple[float, ...]
    consensus_mu: float
    consensus_condition: BrakingCondition
    weather_state: WeatherState
    mu_spread: float


# Machine-stable withholding reason codes.  Any non-empty set blocks issue.
REASON_MISSING_SEGMENT = "missing-segment"
REASON_MEASUREMENT_SPAN = "measurement-span-too-wide"
REASON_WEATHER_CONTEXT = "weather-context-missing"
REASON_READING_CONFLICT = "readings-disagree"
REASON_WEATHER_CONFLICT = "weather-friction-mismatch"

REASON_TEXT: Mapping[str, str] = {
    REASON_MISSING_SEGMENT: "one or more runway thirds were not measured",
    REASON_MEASUREMENT_SPAN: "readings were taken too far apart to describe one runway state",
    REASON_WEATHER_CONTEXT: "a reading has no contemporaneous runway weather context",
    REASON_READING_CONFLICT: "vehicle readings for the same third disagree beyond tolerance",
    REASON_WEATHER_CONFLICT: "measured friction contradicts the prevailing weather regime",
}


@dataclass(frozen=True)
class BrakingAssessment:
    """Explainable fusion result.  Publishable only when no reason withheld it."""
    runway_id: str
    window_start: datetime
    window_end: datetime
    publishable: bool
    overall_condition: BrakingCondition | None
    segments: tuple[SegmentEvidence, ...]
    withhold_reasons: tuple[str, ...] = ()
    weather_states: tuple[WeatherState, ...] = ()
    basis: str = ""

    @property
    def explanation(self) -> str:
        lines = [f"Runway {self.runway_id} braking assessment"]
        if self.publishable:
            lines.append(
                f"PUBLISH: overall braking action {self.overall_condition.value}"
            )
        else:
            lines.append("WITHHELD: do not publish an advisory")
            for code in self.withhold_reasons:
                lines.append(f"  - [{code}] {REASON_TEXT.get(code, code)}")
        for seg in self.segments:
            lines.append(
                f"  {seg.third.value}: mu={seg.consensus_mu:.2f} "
                f"braking={seg.consensus_condition.value} "
                f"weather={seg.weather_state.value} "
                f"vehicles={','.join(seg.vehicles)} "
                f"readings={['%.2f' % x for x in seg.mus]}"
            )
        missing = [
            t.value for t in (RunwayThird.TOUCHDOWN, RunwayThird.MIDPOINT, RunwayThird.ROLLOUT)
            if not any(s.third is t for s in self.segments)
        ]
        if missing:
            lines.append(f"  unmeasured thirds: {', '.join(missing)}")
        if self.basis:
            lines.append(f"  basis: {self.basis}")
        return "\n".join(lines)


class BrakingAssessor:
    """Fuse multi-vehicle friction readings with runway weather.

    The assessor is deliberately conservative:

    * every runway third must carry at least one reading;
    * readings for one third must agree in mu spread and in classified
      category, otherwise the evidence is reported as conflicting;
    * each reading must sit in a known weather regime;
    * measured friction cannot claim better braking than the prevailing
      weather regime supports (weather may be downgraded, not upgraded).

    Any failed check produces a withholding reason; a withheld assessment
    never carries an advisory.
    """

    def __init__(
        self,
        device_profiles: Mapping[str, DeviceProfile] | None = None,
        max_window_minutes: int = 30,
        max_mu_spread: float = 0.10,
        max_category_gap: int = 1,
        weather_upgrade_tolerance: int = 1,
    ):
        self.profiles = dict(device_profiles or DEFAULT_DEVICE_PROFILES)
        if "default" not in self.profiles:
            raise ValueError("a 'default' device profile is required")
        self.max_window = timedelta(minutes=max_window_minutes)
        self.max_mu_spread = max_mu_spread
        self.max_category_gap = max_category_gap
        self.weather_upgrade_tolerance = weather_upgrade_tolerance

    def _profile_for(self, vehicle_id: str) -> DeviceProfile:
        if vehicle_id in self.profiles:
            return self.profiles[vehicle_id]
        return self.profiles["default"]

    def assess(
        self,
        runway_id: str,
        readings: Sequence[FrictionReading],
        weather: Sequence[RunwayWeather] = (),
    ) -> BrakingAssessment:
        rows = [r for r in readings if r.runway_id == runway_id]
        if not rows:
            raise ValueError("no friction readings for runway")
        if any(r.runway_id != runway_id for r in readings):
            raise ValueError("readings mix multiple runways")

        weather_rows = sorted(
            (w for w in weather if w.runway_id == runway_id),
            key=lambda w: w.observed_at,
        )
        starts = sorted(r.measured_at for r in rows)
        window_start, window_end = starts[0], starts[-1]

        reasons: list[str] = []
        groups: dict[RunwayThird, list[FrictionReading]] = {
            t: [] for t in RunwayThird
        }
        for r in rows:
            groups[r.third].append(r)

        # Coverage: a runway state cannot be claimed with an unmeasured third.
        if any(not groups[t] for t in RunwayThird):
            reasons.append(REASON_MISSING_SEGMENT)

        # Whole-runway timing coherence.
        if window_end - window_start > self.max_window:
            reasons.append(REASON_MEASUREMENT_SPAN)

        weather_states: list[WeatherState] = []
        segments: list[SegmentEvidence] = []

        for third in RunwayThird:
            items = sorted(groups[third], key=lambda r: r.vehicle_id)
            if not items:
                continue

            states = {self._weather_context(r, weather_rows) for r in items}
            if None in states:
                reasons.append(REASON_WEATHER_CONTEXT)
            known = [s for s in states if s is not None]
            if known:
                segment_weather = _worst_weather(known)
                weather_states.append(segment_weather)
            else:
                segment_weather = WeatherState.UNKNOWN

            classified = [
                (r, self._profile_for(r.vehicle_id).classify(r.mu))
                for r in items
            ]
            ranks = [_rank(c) for _, c in classified]
            mus = [r.mu for r in items]
            spread = max(mus) - min(mus)

            # Two testers reporting different calibrated devices can differ a
            # little in mu but must not disagree in category beyond tolerance.
            if max(ranks) - min(ranks) > self.max_category_gap or (
                len(mus) > 1 and spread > self.max_mu_spread
            ):
                reasons.append(REASON_READING_CONFLICT)

            consensus_mu = round(median(mus), 3)
            ordered_ranks = sorted(_rank(c) for _, c in classified)
            mid = len(ordered_ranks) // 2
            # Even-sized groups take the more restrictive middle category.
            consensus_index = (
                ordered_ranks[mid]
                if len(ordered_ranks) % 2
                else max(ordered_ranks[mid - 1], ordered_ranks[mid])
            )
            consensus_condition = BRAKING_ORDER[consensus_index]

            # Weather cross-check: weather sets the best believable category.
            if known:
                baseline = WEATHER_BASELINE[segment_weather]
                if (
                    _rank(baseline) - _rank(consensus_condition)
                    > self.weather_upgrade_tolerance
                ):
                    reasons.append(REASON_WEATHER_CONFLICT)

            segments.append(
                SegmentEvidence(
                    third=third,
                    vehicles=tuple(r.vehicle_id for r in items),
                    measured_at=tuple(r.measured_at for r in items),
                    mus=tuple(mus),
                    consensus_mu=consensus_mu,
                    consensus_condition=consensus_condition,
                    weather_state=segment_weather,
                    mu_spread=round(spread, 3),
                )
            )

        publishable = not reasons
        overall = (
            _worst(s.consensus_condition for s in segments)
            if publishable and segments
            else None
        )
        basis = self._basis(segments, weather_rows) if publishable else ""
        return BrakingAssessment(
            runway_id=runway_id,
            window_start=window_start,
            window_end=window_end,
            publishable=publishable,
            overall_condition=overall,
            segments=tuple(segments),
            withhold_reasons=tuple(dict.fromkeys(reasons)),
            weather_states=tuple(dict.fromkeys(weather_states)),
            basis=basis,
        )

    @staticmethod
    def _weather_context(
        reading: FrictionReading, weather_rows: Sequence[RunwayWeather]
    ) -> WeatherState | None:
        # An explicit reading-tagged regime overrides; otherwise use the last
        # runway observation at or before the measurement.
        prior = [w for w in weather_rows if w.observed_at <= reading.measured_at]
        observed = prior[-1].state if prior else None
        if reading.weather_state is not WeatherState.UNKNOWN:
            return reading.weather_state
        return observed

    @staticmethod
    def _basis(
        segments: Sequence[SegmentEvidence], weather_rows: Sequence[RunwayWeather]
    ) -> str:
        parts = []
        for seg in segments:
            parts.append(
                f"{seg.third.value} median mu {seg.consensus_mu:.2f} "
                f"from {len(seg.vehicles)} vehicle(s)"
            )
        if weather_rows:
            latest = weather_rows[-1]
            parts.append(
                f"weather {latest.state.value} observed {latest.observed_at:%Y-%m-%d %H:%M}Z"
            )
        return "; ".join(parts)


def _worst_weather(states: Iterable[WeatherState]) -> WeatherState:
    severity = {
        WeatherState.DRY: 0,
        WeatherState.RAIN: 1,
        WeatherState.SNOW_SLUSH: 2,
        WeatherState.COMPACTED_SNOW: 3,
        WeatherState.ICE: 4,
        WeatherState.UNKNOWN: 5,
    }
    return max(states, key=lambda s: severity[s])


@dataclass(frozen=True)
class BrakingAdvisory:
    """Issued, externally-distributed advisory.  Immutable once published.

    A revision is a brand-new advisory object that references the prior
    version; the previous object is never rewritten.
    """
    advisory_key: str
    version: int
    runway_id: str
    published_at: datetime
    effective_from: datetime
    condition: BrakingCondition
    text: str
    assessment: BrakingAssessment
    supersedes_version: int | None = None
    note: str = ""

    def __post_init__(self):
        object.__setattr__(self, "published_at", utc(self.published_at))
        object.__setattr__(self, "effective_from", utc(self.effective_from))
        if self.version < 1:
            raise ValueError("advisory version starts at 1")
        if not self.assessment.publishable:
            raise ValueError("cannot issue an advisory from a withheld assessment")
        if self.supersedes_version is not None and self.supersedes_version >= self.version:
            raise ValueError("revision must supersede an earlier version")
        if not self.text:
            raise ValueError("advisory text required")

    @property
    def is_revision(self) -> bool:
        return self.supersedes_version is not None


class AdvisoryError(RuntimeError):
    pass


class BrakingAdvisoryLedger:
    """Append-only registry of braking advisories.

    ``issue`` creates version 1 of an advisory key.  ``revise`` appends a new
    version that supersedes the previous one; the old record remains readable
    via ``history`` and nothing in the ledger is ever mutated or deleted.
    """

    def __init__(self):
        self._records: dict[str, list[BrakingAdvisory]] = {}

    def issue(
        self,
        advisory_key: str,
        assessment: BrakingAssessment,
        published_at: datetime,
        text: str | None = None,
        note: str = "",
    ) -> BrakingAdvisory:
        if advisory_key in self._records:
            raise AdvisoryError(f"{advisory_key} already issued; revise instead")
        if not assessment.publishable or assessment.overall_condition is None:
            raise AdvisoryError("assessment withheld; advisory cannot be issued")
        body = text or self._default_text(assessment)
        advisory = BrakingAdvisory(
            advisory_key=advisory_key,
            version=1,
            runway_id=assessment.runway_id,
            published_at=published_at,
            effective_from=assessment.window_end,
            condition=assessment.overall_condition,
            text=body,
            assessment=assessment,
            note=note,
        )
        self._records[advisory_key] = [advisory]
        return advisory

    def revise(
        self,
        advisory_key: str,
        assessment: BrakingAssessment,
        published_at: datetime,
        text: str | None = None,
        note: str = "",
    ) -> BrakingAdvisory:
        history = self._records.get(advisory_key)
        if not history:
            raise AdvisoryError(f"{advisory_key} unknown; issue it first")
        if not assessment.publishable or assessment.overall_condition is None:
            raise AdvisoryError("assessment withheld; revision cannot be issued")
        prior = history[-1]
        body = text or self._default_text(assessment)
        revised = BrakingAdvisory(
            advisory_key=advisory_key,
            version=prior.version + 1,
            runway_id=assessment.runway_id,
            published_at=published_at,
            effective_from=assessment.window_end,
            condition=assessment.overall_condition,
            text=body,
            assessment=assessment,
            supersedes_version=prior.version,
            note=note,
        )
        history.append(revised)
        return revised

    @staticmethod
    def _default_text(assessment: BrakingAssessment) -> str:
        return (
            f"Runway {assessment.runway_id} braking action "
            f"{assessment.overall_condition.value}"
        )

    def current(self, advisory_key: str) -> BrakingAdvisory:
        history = self._records.get(advisory_key)
        if not history:
            raise KeyError(advisory_key)
        return history[-1]

    def version(self, advisory_key: str, version: int) -> BrakingAdvisory:
        for advisory in self._records.get(advisory_key, ()):
            if advisory.version == version:
                return advisory
        raise KeyError((advisory_key, version))

    def history(self, advisory_key: str) -> tuple[BrakingAdvisory, ...]:
        return tuple(self._records.get(advisory_key, ()))

    def all_current(self) -> tuple[BrakingAdvisory, ...]:
        return tuple(rows[-1] for rows in self._records.values())
