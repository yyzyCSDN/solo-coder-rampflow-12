from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime,timedelta
from typing import Iterable
from .timeutil import utc
from .weather import WeatherObservation

RUNWAY_SEGMENTS=("touchdown","midpoint","rollout")
CODE_LABELS={0:"nil",1:"poor",2:"medium-to-poor",3:"medium",4:"good-to-medium",5:"good"}
_MU_BANDS=((0.40,5),(0.36,4),(0.30,3),(0.26,2),(0.0,1))
_ACTION_CODES={"good":5,"good-to-medium":4,"medium":3,"medium-to-poor":2,"poor":1,"nil":0}

def classify_mu(mu:float)->int:
    for floor,code in _MU_BANDS:
        if mu>=floor:return code
    return 1

def action_code(label:str)->int|None:
    return _ACTION_CODES.get(label.strip().lower().replace("_","-"))

@dataclass(frozen=True)
class FrictionReading:
    reading_id:str
    runway:str
    segment:str
    vehicle_id:str
    measured_at:datetime
    mu:float
    def __post_init__(self):
        object.__setattr__(self,"measured_at",utc(self.measured_at))
        if not self.reading_id or not self.runway or not self.vehicle_id:raise ValueError("reading identity required")
        if not 0.0<=self.mu<=1.0:raise ValueError("mu outside 0..1")

@dataclass(frozen=True)
class SegmentAssessment:
    segment:str
    code:int
    mu_min:float
    mu_max:float
    reading_ids:tuple[str,...]
    basis:tuple[str,...]
    @property
    def label(self)->str:return CODE_LABELS[self.code]

@dataclass(frozen=True)
class RunwayAssessment:
    runway:str
    assessed_at:datetime
    segments:tuple[SegmentAssessment,...]
    overall_code:int|None
    holds:tuple[str,...]
    basis:tuple[str,...]
    @property
    def publishable(self)->bool:return not self.holds
    @property
    def overall_label(self)->str|None:
        return CODE_LABELS[self.overall_code] if self.overall_code is not None else None

@dataclass(frozen=True)
class PublishedReport:
    report_id:str
    runway:str
    revision:int
    issued_at:datetime
    assessment:RunwayAssessment
    supersedes:int|None=None
    note:str=""
    def __post_init__(self):
        object.__setattr__(self,"issued_at",utc(self.issued_at))
        if self.revision<1:raise ValueError("revision must be positive")
        if not self.assessment.publishable:raise ValueError("cannot publish an assessment with holds")

class RunwayFrictionOffice:
    def __init__(self,segments:Iterable[str]=RUNWAY_SEGMENTS,freshness_min:int=60,mu_tolerance:float=0.08,code_tolerance:int=1,weather_max_age_min:int=30):
        if freshness_min<1:raise ValueError("freshness window must be positive")
        self.segments=tuple(segments);self.freshness_min=freshness_min
        self.mu_tolerance=mu_tolerance;self.code_tolerance=code_tolerance
        self.weather_max_age_min=weather_max_age_min
        self._readings:dict[str,FrictionReading]={}
        self._reports:dict[str,dict[int,PublishedReport]]={}

    def ingest(self,reading:FrictionReading)->FrictionReading:
        if reading.segment not in self.segments:raise ValueError(f"unknown runway segment {reading.segment}")
        existing=self._readings.get(reading.reading_id)
        if existing is not None:
            if existing!=reading:raise ValueError("reading id reused with different payload")
            return existing
        self._readings[reading.reading_id]=reading;return reading

    def _weather_code(self,at:datetime,weather:WeatherObservation|None,holds:list[str])->int|None:
        if weather is None:
            holds.append("weather-unavailable: no weather observation supplied");return None
        age=at-weather.observed_at
        if age<timedelta(0):
            holds.append("weather-in-future: observation is after the assessment time");return None
        if age>timedelta(minutes=self.weather_max_age_min):
            holds.append(f"weather-stale: observation is {int(age.total_seconds()//60)} min old (limit {self.weather_max_age_min})");return None
        code=action_code(weather.braking_action)
        if code is None:
            holds.append(f"weather-label-unknown: braking action {weather.braking_action!r} not recognized");return None
        return code

    def assess(self,runway:str,at:datetime,weather:WeatherObservation|None=None)->RunwayAssessment:
        at=utc(at);holds:list[str]=[];basis:list[str]=[];segments_out:list[SegmentAssessment]=[]
        weather_code=self._weather_code(at,weather,holds)
        horizon=timedelta(minutes=self.freshness_min)
        for segment in self.segments:
            measured=[r for r in self._readings.values() if r.runway==runway and r.segment==segment and r.measured_at<=at]
            fresh=[r for r in measured if at-r.measured_at<=horizon]
            if not fresh:
                if measured:holds.append(f"segment-stale: {segment} newest reading is older than {self.freshness_min} min")
                else:holds.append(f"segment-uncovered: {segment} has no usable friction reading")
                continue
            latest:dict[str,FrictionReading]={}
            for r in fresh:
                prev=latest.get(r.vehicle_id)
                if prev is None or r.measured_at>prev.measured_at:latest[r.vehicle_id]=r
            current=sorted(latest.values(),key=lambda r:r.reading_id)
            superseded=len(fresh)-len(current)
            mus=[r.mu for r in current];codes=[classify_mu(m) for m in mus]
            if max(mus)-min(mus)>self.mu_tolerance or max(codes)-min(codes)>self.code_tolerance:
                detail=",".join(f"{r.vehicle_id}={r.mu:.2f}" for r in sorted(current,key=lambda x:x.mu))
                holds.append(f"segment-conflict: {segment} readings disagree ({detail})")
                continue
            code=classify_mu(min(mus))
            vehicles=",".join(sorted(latest))
            note=f" from {len(current)} vehicle(s) [{vehicles}]"
            if superseded:note+=f", {superseded} earlier reading(s) superseded by newer ones"
            seg_basis=[f"{segment}: mu {min(mus):.2f}-{max(mus):.2f}{note} -> {CODE_LABELS[code]} ({code})"]
            if weather_code is not None and weather_code<code:
                seg_basis.append(f"{segment}: weather braking action {CODE_LABELS[weather_code]} ({weather_code}) downgrades mu-based {CODE_LABELS[code]} ({code})")
                code=weather_code
            segments_out.append(SegmentAssessment(segment,code,min(mus),max(mus),tuple(sorted(r.reading_id for r in current)),tuple(seg_basis)))
        overall=None
        if not holds:
            overall=min(s.code for s in segments_out)
            for s in segments_out:basis.extend(s.basis)
            basis.append(f"runway {runway}: worst segment governs -> {CODE_LABELS[overall]} ({overall})")
        return RunwayAssessment(runway,at,tuple(segments_out),overall,tuple(holds),tuple(basis))

    def publish(self,report_id:str,runway:str,at:datetime,weather:WeatherObservation|None=None,note:str="")->PublishedReport:
        if report_id in self._reports:raise RuntimeError(f"report {report_id} already published; issue a revision instead")
        assessment=self.assess(runway,at,weather)
        if not assessment.publishable:raise RuntimeError("assessment not publishable: "+"; ".join(assessment.holds))
        report=PublishedReport(report_id,runway,1,at,assessment,None,note)
        self._reports[report_id]={1:report}
        return report

    def revise(self,report_id:str,at:datetime,weather:WeatherObservation|None=None,note:str="")->PublishedReport:
        if report_id not in self._reports:raise RuntimeError(f"unknown report {report_id}")
        runway=self.latest(report_id).runway
        assessment=self.assess(runway,at,weather)
        if not assessment.publishable:raise RuntimeError("revision not publishable: "+"; ".join(assessment.holds))
        revision=max(self._reports[report_id])+1
        report=PublishedReport(report_id,runway,revision,at,assessment,revision-1,note)
        self._reports[report_id][revision]=report
        return report

    def latest(self,report_id:str)->PublishedReport:
        revisions=self._reports[report_id]
        return revisions[max(revisions)]

    def history(self,report_id:str)->tuple[PublishedReport,...]:
        return tuple(self._reports[report_id][r] for r in sorted(self._reports[report_id]))
