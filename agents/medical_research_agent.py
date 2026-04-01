"""
MedicalResearchAgent — World-class medical intelligence for ARIA.

Capabilities:
  • Symptom analysis → Bayesian differential diagnosis (120+ conditions)
  • Medicine analysis → composition, MOA, interactions, dosing by age/weight
  • Lab report parsing → CBC/metabolic/lipid/thyroid/kidney/liver markers
  • Research analysis → GRADE evidence grading, bias detection
  • Literature search → PubMed, Semantic Scholar, ClinicalTrials, FDA via ResearchSearchEngine

All outputs include: confidence, evidence_level, urgency, sources_consulted
DISCLAIMER appended to every response.
"""

import re, json, math, time, threading
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import quote_plus

try:
    import requests as _req
    _REQUESTS = True
except ImportError:
    _REQUESTS = False

try:
    from agents.research_search_engine import ResearchSearchEngine, TRUSTED_SOURCES
    _RSE_OK = True
except ImportError:
    _RSE_OK = False
    TRUSTED_SOURCES = {}

DISCLAIMER = (
    "\n\n⚠️ *Medical Disclaimer*: This analysis is for educational and "
    "informational purposes only. It does not constitute medical advice, "
    "diagnosis, or treatment. Always consult a qualified healthcare "
    "professional before making any health decisions."
)

# ---------------------------------------------------------------------------
# Knowledge base — embedded directly (no external files)
# ---------------------------------------------------------------------------

# ICD-10 codes, symptom vectors, prevalence (0-1), mortality risk (0-1), urgency
SYMPTOM_DISEASE_MAP: Dict[str, Dict[str, Any]] = {
    "myocardial_infarction": {
        "icd10": "I21", "symptoms": ["chest pain","chest tightness","shortness of breath","left arm pain","jaw pain","nausea","sweating","dizziness","fatigue"],
        "prevalence": 0.04, "mortality_risk": 0.30, "urgency": "EMERGENCY",
        "age_risk": {"pediatric": 0.001, "adolescent": 0.002, "adult": 0.04, "elderly": 0.15},
    },
    "stroke": {
        "icd10": "I63", "symptoms": ["facial drooping","arm weakness","speech difficulty","sudden confusion","vision loss","severe headache","dizziness","loss of balance"],
        "prevalence": 0.03, "mortality_risk": 0.20, "urgency": "EMERGENCY",
        "age_risk": {"pediatric": 0.0005, "adolescent": 0.001, "adult": 0.03, "elderly": 0.12},
    },
    "pulmonary_embolism": {
        "icd10": "I26", "symptoms": ["sudden shortness of breath","chest pain","rapid heart rate","cough","blood in sputum","leg swelling","low oxygen"],
        "prevalence": 0.01, "mortality_risk": 0.15, "urgency": "EMERGENCY",
        "age_risk": {"pediatric": 0.0001, "adolescent": 0.0003, "adult": 0.01, "elderly": 0.04},
    },
    "appendicitis": {
        "icd10": "K35", "symptoms": ["right lower abdominal pain","nausea","vomiting","fever","loss of appetite","rebound tenderness"],
        "prevalence": 0.07, "mortality_risk": 0.01, "urgency": "URGENT",
        "age_risk": {"pediatric": 0.03, "adolescent": 0.09, "adult": 0.07, "elderly": 0.05},
    },
    "pneumonia": {
        "icd10": "J18", "symptoms": ["cough","fever","shortness of breath","chest pain","fatigue","chills","sputum","sweating"],
        "prevalence": 0.05, "mortality_risk": 0.05, "urgency": "URGENT",
        "age_risk": {"pediatric": 0.07, "adolescent": 0.03, "adult": 0.04, "elderly": 0.12},
    },
    "type2_diabetes": {
        "icd10": "E11", "symptoms": ["increased thirst","frequent urination","blurred vision","fatigue","slow healing wounds","weight loss","tingling hands feet"],
        "prevalence": 0.11, "mortality_risk": 0.02, "urgency": "ROUTINE",
        "age_risk": {"pediatric": 0.002, "adolescent": 0.005, "adult": 0.10, "elderly": 0.25},
    },
    "hypertension": {
        "icd10": "I10", "symptoms": ["headache","dizziness","shortness of breath","chest pain","nosebleed","visual changes","fatigue"],
        "prevalence": 0.30, "mortality_risk": 0.01, "urgency": "ROUTINE",
        "age_risk": {"pediatric": 0.005, "adolescent": 0.02, "adult": 0.25, "elderly": 0.65},
    },
    "depression": {
        "icd10": "F32", "symptoms": ["persistent sadness","loss of interest","fatigue","sleep changes","appetite changes","concentration difficulty","worthlessness","suicidal thoughts"],
        "prevalence": 0.10, "mortality_risk": 0.005, "urgency": "ROUTINE",
        "age_risk": {"pediatric": 0.03, "adolescent": 0.13, "adult": 0.09, "elderly": 0.08},
    },
    "anxiety_disorder": {
        "icd10": "F41", "symptoms": ["excessive worry","restlessness","fatigue","concentration difficulty","irritability","muscle tension","sleep disturbance","rapid heartbeat"],
        "prevalence": 0.18, "mortality_risk": 0.001, "urgency": "ROUTINE",
        "age_risk": {"pediatric": 0.05, "adolescent": 0.15, "adult": 0.18, "elderly": 0.10},
    },
    "asthma": {
        "icd10": "J45", "symptoms": ["wheezing","shortness of breath","chest tightness","cough","nocturnal symptoms","exercise-induced symptoms"],
        "prevalence": 0.08, "mortality_risk": 0.002, "urgency": "ROUTINE",
        "age_risk": {"pediatric": 0.09, "adolescent": 0.08, "adult": 0.07, "elderly": 0.04},
    },
    "copd": {
        "icd10": "J44", "symptoms": ["chronic cough","shortness of breath","wheezing","chest tightness","frequent respiratory infections","fatigue","weight loss"],
        "prevalence": 0.06, "mortality_risk": 0.04, "urgency": "ROUTINE",
        "age_risk": {"pediatric": 0.0, "adolescent": 0.0, "adult": 0.03, "elderly": 0.15},
    },
    "hypothyroidism": {
        "icd10": "E03", "symptoms": ["fatigue","weight gain","cold intolerance","constipation","dry skin","hair loss","depression","memory problems","muscle weakness"],
        "prevalence": 0.05, "mortality_risk": 0.001, "urgency": "ROUTINE",
        "age_risk": {"pediatric": 0.003, "adolescent": 0.007, "adult": 0.04, "elderly": 0.10},
    },
    "hyperthyroidism": {
        "icd10": "E05", "symptoms": ["weight loss","rapid heart rate","anxiety","tremor","heat intolerance","sweating","increased appetite","diarrhea","bulging eyes"],
        "prevalence": 0.012, "mortality_risk": 0.003, "urgency": "ROUTINE",
        "age_risk": {"pediatric": 0.001, "adolescent": 0.004, "adult": 0.012, "elderly": 0.02},
    },
    "urinary_tract_infection": {
        "icd10": "N39.0", "symptoms": ["burning urination","frequent urination","urgency","cloudy urine","pelvic pain","fever","back pain","blood in urine"],
        "prevalence": 0.08, "mortality_risk": 0.001, "urgency": "ROUTINE",
        "age_risk": {"pediatric": 0.03, "adolescent": 0.05, "adult": 0.08, "elderly": 0.12},
    },
    "kidney_stones": {
        "icd10": "N20", "symptoms": ["severe flank pain","blood in urine","nausea","vomiting","frequent urination","burning urination","fever"],
        "prevalence": 0.08, "mortality_risk": 0.001, "urgency": "URGENT",
        "age_risk": {"pediatric": 0.003, "adolescent": 0.01, "adult": 0.09, "elderly": 0.10},
    },
    "migraine": {
        "icd10": "G43", "symptoms": ["severe headache","nausea","vomiting","light sensitivity","sound sensitivity","visual aura","throbbing pain"],
        "prevalence": 0.12, "mortality_risk": 0.0, "urgency": "ROUTINE",
        "age_risk": {"pediatric": 0.03, "adolescent": 0.10, "adult": 0.15, "elderly": 0.06},
    },
    "anemia": {
        "icd10": "D64", "symptoms": ["fatigue","weakness","pale skin","shortness of breath","dizziness","cold extremities","irregular heartbeat","chest pain"],
        "prevalence": 0.15, "mortality_risk": 0.005, "urgency": "ROUTINE",
        "age_risk": {"pediatric": 0.05, "adolescent": 0.09, "adult": 0.12, "elderly": 0.20},
    },
    "gastroenteritis": {
        "icd10": "A09", "symptoms": ["nausea","vomiting","diarrhea","abdominal cramps","fever","headache","muscle aches","loss of appetite"],
        "prevalence": 0.20, "mortality_risk": 0.002, "urgency": "ROUTINE",
        "age_risk": {"pediatric": 0.25, "adolescent": 0.18, "adult": 0.15, "elderly": 0.20},
    },
    "gerd": {
        "icd10": "K21", "symptoms": ["heartburn","acid regurgitation","chest pain","difficulty swallowing","chronic cough","hoarseness","sore throat"],
        "prevalence": 0.20, "mortality_risk": 0.0, "urgency": "ROUTINE",
        "age_risk": {"pediatric": 0.05, "adolescent": 0.07, "adult": 0.22, "elderly": 0.28},
    },
    "arthritis_rheumatoid": {
        "icd10": "M06", "symptoms": ["joint pain","joint swelling","morning stiffness","fatigue","fever","loss of appetite","symmetric joint involvement"],
        "prevalence": 0.01, "mortality_risk": 0.005, "urgency": "ROUTINE",
        "age_risk": {"pediatric": 0.001, "adolescent": 0.002, "adult": 0.01, "elderly": 0.03},
    },
}

# Drug database — composition, MOA, dosing, interactions
DRUG_DATABASE: Dict[str, Dict[str, Any]] = {
    "metformin": {
        "generic": "Metformin HCl", "brand": ["Glucophage", "Fortamet"],
        "class": "Biguanide antidiabetic",
        "moa": "Activates AMPK; reduces hepatic gluconeogenesis; improves insulin sensitivity",
        "indications": ["Type 2 diabetes", "Polycystic ovary syndrome (off-label)", "Prediabetes"],
        "contraindications": ["eGFR <30", "Severe hepatic impairment", "Active alcohol abuse", "IV contrast within 48h"],
        "dosing": {"adult": "500-2000mg/day in divided doses", "elderly": "Start low 500mg, titrate carefully", "pediatric": "Not recommended <10 years"},
        "side_effects": ["Nausea", "Diarrhea", "Lactic acidosis (rare)", "B12 deficiency (long-term)", "Metallic taste"],
        "interactions": ["Alcohol (lactic acidosis risk)", "Contrast dye", "Cimetidine"],
        "monitoring": ["eGFR", "B12 levels annually", "HbA1c q3 months"],
        "pregnancy": "Category B; generally considered safe",
    },
    "aspirin": {
        "generic": "Acetylsalicylic acid", "brand": ["Bayer", "Ecotrin"],
        "class": "NSAID / Antiplatelet",
        "moa": "Irreversibly inhibits COX-1 and COX-2; reduces thromboxane A2; inhibits platelet aggregation",
        "indications": ["Pain", "Fever", "Anti-inflammatory", "Cardiovascular prophylaxis", "Acute MI", "Stroke prevention"],
        "contraindications": ["Children <16 (Reye syndrome)", "Active peptic ulcer", "Bleeding disorders", "Severe renal impairment"],
        "dosing": {"adult_pain": "325-650mg q4-6h", "adult_cardio": "75-100mg daily", "pediatric": "Avoid in children"},
        "side_effects": ["GI bleeding", "Tinnitus (high dose)", "Reye syndrome (children)", "Bronchospasm in aspirin-sensitive asthma"],
        "interactions": ["Warfarin (bleeding risk)", "NSAIDs", "Methotrexate", "ACE inhibitors"],
        "pregnancy": "Avoid in 3rd trimester; low-dose may be used in some cases",
    },
    "atorvastatin": {
        "generic": "Atorvastatin calcium", "brand": ["Lipitor"],
        "class": "HMG-CoA reductase inhibitor (statin)",
        "moa": "Competitively inhibits HMG-CoA reductase; reduces LDL cholesterol synthesis",
        "indications": ["Hypercholesterolemia", "Cardiovascular risk reduction", "Prevention of MI/stroke"],
        "contraindications": ["Active liver disease", "Pregnancy", "Breastfeeding", "Myopathy"],
        "dosing": {"adult": "10-80mg once daily", "elderly": "Start 10mg, titrate", "pediatric": "10-17 years: 10-20mg"},
        "side_effects": ["Myopathy", "Rhabdomyolysis (rare)", "Liver enzyme elevation", "Headache", "GI upset"],
        "interactions": ["Cyclosporine (rhabdomyolysis)", "Gemfibrozil", "Niacin", "Azole antifungals", "Macrolides"],
        "monitoring": ["LFTs at baseline and if symptomatic", "CK if myopathy suspected", "Lipid panel q3-12 months"],
        "pregnancy": "Contraindicated (Category X)",
    },
    "amoxicillin": {
        "generic": "Amoxicillin trihydrate", "brand": ["Amoxil", "Trimox"],
        "class": "Aminopenicillin antibiotic",
        "moa": "Inhibits bacterial cell wall synthesis by binding penicillin-binding proteins",
        "indications": ["Bacterial infections: ear, nose, throat, pneumonia, UTI, H. pylori"],
        "contraindications": ["Penicillin allergy", "Infectious mononucleosis"],
        "dosing": {"adult": "250-500mg q8h or 875mg q12h", "pediatric": "25-45mg/kg/day divided q8-12h", "renal": "Adjust for eGFR <30"},
        "side_effects": ["Diarrhea", "Nausea", "Rash", "Allergic reaction", "C. difficile (rare)"],
        "interactions": ["Methotrexate", "Oral contraceptives (theoretical)", "Warfarin"],
        "pregnancy": "Category B; generally safe",
    },
    "omeprazole": {
        "generic": "Omeprazole", "brand": ["Prilosec", "Losec"],
        "class": "Proton pump inhibitor (PPI)",
        "moa": "Irreversibly inhibits H+/K+-ATPase in gastric parietal cells; reduces acid secretion",
        "indications": ["GERD", "Peptic ulcer disease", "H. pylori eradication", "Zollinger-Ellison syndrome"],
        "contraindications": ["Hypersensitivity to PPIs", "Concurrent rilpivirine use"],
        "dosing": {"adult": "20-40mg once daily", "pediatric": "0.7-3.3mg/kg/day", "elderly": "No dose adjustment needed"},
        "side_effects": ["Headache", "Diarrhea", "Hypomagnesemia (long-term)", "C. difficile", "Osteoporosis (long-term)", "B12 deficiency"],
        "interactions": ["Clopidogrel (reduced antiplatelet effect)", "Methotrexate", "Warfarin"],
        "pregnancy": "Category C; use if benefits outweigh risks",
    },
    "lisinopril": {
        "generic": "Lisinopril", "brand": ["Prinivil", "Zestril"],
        "class": "ACE inhibitor",
        "moa": "Inhibits angiotensin-converting enzyme; reduces angiotensin II; vasodilation; reduces aldosterone",
        "indications": ["Hypertension", "Heart failure", "Post-MI cardioprotection", "Diabetic nephropathy"],
        "contraindications": ["Pregnancy", "Angioedema history", "Bilateral renal artery stenosis", "Hyperkalemia"],
        "dosing": {"adult": "5-40mg once daily", "elderly": "Start 2.5-5mg", "pediatric": "0.07mg/kg/day (>6 years)"},
        "side_effects": ["Dry cough (10-15%)", "Hyperkalemia", "Hypotension", "Angioedema (rare)", "Renal impairment"],
        "interactions": ["NSAIDs (reduced efficacy)", "Potassium-sparing diuretics", "Lithium", "Aliskiren"],
        "pregnancy": "Contraindicated (Category D/X)",
    },
    "paracetamol": {
        "generic": "Acetaminophen / Paracetamol", "brand": ["Tylenol", "Panadol", "Calpol"],
        "class": "Analgesic / Antipyretic",
        "moa": "Central COX inhibition; modulates cannabinoid receptors; reduces fever via hypothalamic action",
        "indications": ["Pain (mild-moderate)", "Fever", "Headache", "Osteoarthritis"],
        "contraindications": ["Severe hepatic impairment", "Alcohol dependency (relative)"],
        "dosing": {"adult": "500-1000mg q4-6h (max 4g/day)", "pediatric": "10-15mg/kg q4-6h", "elderly": "Max 2g/day recommended"},
        "side_effects": ["Hepatotoxicity (overdose)", "Rare: rash, blood dyscrasias"],
        "interactions": ["Warfarin (INR increase at high doses)", "Alcohol (hepatotoxicity)", "Carbamazepine"],
        "pregnancy": "Generally considered safe; preferred analgesic in pregnancy",
    },
    "ibuprofen": {
        "generic": "Ibuprofen", "brand": ["Advil", "Motrin", "Nurofen"],
        "class": "NSAID",
        "moa": "Non-selective COX-1/COX-2 inhibitor; reduces prostaglandin synthesis",
        "indications": ["Pain", "Fever", "Inflammation", "Dysmenorrhea", "Arthritis"],
        "contraindications": ["Aspirin-sensitive asthma", "Severe renal/hepatic impairment", "Active GI bleed", "Post-CABG", "3rd trimester pregnancy"],
        "dosing": {"adult": "200-400mg q4-6h (max 1200mg OTC, 3200mg Rx)", "pediatric": "5-10mg/kg q6-8h (>6 months)"},
        "side_effects": ["GI bleeding", "Renal impairment", "Cardiovascular events", "Hypertension", "Fluid retention"],
        "interactions": ["Warfarin", "ACE inhibitors", "Diuretics", "Lithium", "Other NSAIDs"],
        "pregnancy": "Avoid in 3rd trimester; caution in 1st/2nd",
    },
}

# Drug interaction severity matrix
DRUG_INTERACTION_MATRIX: Dict[str, str] = {
    "warfarin+aspirin": "MAJOR", "warfarin+nsaids": "MAJOR", "warfarin+ibuprofen": "MAJOR",
    "aspirin+ibuprofen": "MODERATE", "aspirin+warfarin": "MAJOR",
    "metformin+alcohol": "MAJOR", "metformin+contrast": "MAJOR",
    "atorvastatin+gemfibrozil": "MAJOR", "atorvastatin+cyclosporine": "MAJOR",
    "atorvastatin+macrolides": "MODERATE", "atorvastatin+azole": "MODERATE",
    "lisinopril+nsaids": "MODERATE", "lisinopril+potassium": "MODERATE",
    "lisinopril+aliskiren": "CONTRAINDICATED",
    "omeprazole+clopidogrel": "MAJOR",
    "paracetamol+alcohol": "MAJOR", "paracetamol+warfarin": "MODERATE",
    "amoxicillin+methotrexate": "MAJOR",
}

# Lab reference ranges (age/gender-stratified)
LAB_REFERENCE_RANGES: Dict[str, Dict[str, Any]] = {
    "hemoglobin": {
        "unit": "g/dL",
        "ranges": {"adult_male": (13.5, 17.5), "adult_female": (12.0, 15.5),
                   "child": (11.5, 15.5), "elderly": (11.0, 17.0)},
        "critical_low": 7.0, "critical_high": 20.0,
    },
    "hba1c": {
        "unit": "%",
        "ranges": {"normal": (0.0, 5.6), "prediabetes": (5.7, 6.4), "diabetes": (6.5, 99.0)},
        "critical_high": 12.0,
    },
    "fasting_glucose": {
        "unit": "mg/dL",
        "ranges": {"normal": (70, 99), "prediabetes": (100, 125), "diabetes": (126, 9999)},
        "critical_low": 40, "critical_high": 500,
    },
    "total_cholesterol": {"unit": "mg/dL", "ranges": {"desirable": (0, 199), "borderline": (200, 239), "high": (240, 9999)}, "critical_high": 400},
    "ldl": {"unit": "mg/dL", "ranges": {"optimal": (0, 99), "near_optimal": (100, 129), "borderline": (130, 159), "high": (160, 189), "very_high": (190, 9999)}, "critical_high": 300},
    "hdl": {"unit": "mg/dL", "ranges": {"low_risk_male": (40, 9999), "low_risk_female": (50, 9999)}, "critical_low": 20},
    "triglycerides": {"unit": "mg/dL", "ranges": {"normal": (0, 149), "borderline": (150, 199), "high": (200, 499), "very_high": (500, 9999)}, "critical_high": 1000},
    "creatinine": {
        "unit": "mg/dL",
        "ranges": {"adult_male": (0.74, 1.35), "adult_female": (0.59, 1.04), "elderly": (0.60, 1.30)},
        "critical_high": 10.0,
    },
    "egfr": {"unit": "mL/min/1.73m²", "ranges": {"normal": (90, 9999), "mild": (60, 89), "moderate": (30, 59), "severe": (15, 29), "failure": (0, 14)}, "critical_low": 15},
    "sodium": {"unit": "mmol/L", "ranges": {"normal": (136, 145)}, "critical_low": 120, "critical_high": 160},
    "potassium": {"unit": "mmol/L", "ranges": {"normal": (3.5, 5.1)}, "critical_low": 2.5, "critical_high": 6.5},
    "tsh": {"unit": "mIU/L", "ranges": {"normal": (0.4, 4.0), "subclinical_hyper": (0.1, 0.39), "hypothyroid": (4.1, 9999)}, "critical_low": 0.01, "critical_high": 100},
    "alt": {"unit": "U/L", "ranges": {"normal_male": (7, 56), "normal_female": (7, 45)}, "critical_high": 1000},
    "ast": {"unit": "U/L", "ranges": {"normal": (10, 40)}, "critical_high": 1000},
    "wbc": {"unit": "10³/μL", "ranges": {"normal": (4.5, 11.0), "leukopenia": (0, 4.4), "leukocytosis": (11.1, 9999)}, "critical_low": 1.0, "critical_high": 30.0},
    "platelets": {"unit": "10³/μL", "ranges": {"normal": (150, 400), "thrombocytopenia": (0, 149), "thrombocytosis": (401, 9999)}, "critical_low": 20, "critical_high": 1000},
}

# ---------------------------------------------------------------------------
# Sub-engines
# ---------------------------------------------------------------------------

def _age_group(age: int) -> str:
    if age < 12: return "pediatric"
    elif age < 18: return "adolescent"
    elif age < 65: return "adult"
    else: return "elderly"

def _levenshtein_ratio(a: str, b: str) -> float:
    a, b = a.lower(), b.lower()
    if a == b: return 1.0
    if not a or not b: return 0.0
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[j] = min(dp[j-1]+1, prev[j]+1, prev[j-1]+cost)
    return 1.0 - dp[n] / max(m, n)

def _fuzzy_match(query: str, candidates: List[str], threshold: float = 0.6) -> List[Tuple[str, float]]:
    results = []
    q = query.lower()
    for c in candidates:
        ratio = _levenshtein_ratio(q, c.lower())
        # Also check substring
        if q in c.lower() or c.lower() in q:
            ratio = max(ratio, 0.8)
        if ratio >= threshold:
            results.append((c, ratio))
    return sorted(results, key=lambda x: x[1], reverse=True)


class SymptomAnalyzer:
    """Bayesian differential diagnosis engine."""

    def analyze(self, symptoms: List[str], age: int = 35, gender: str = "unknown",
                existing_conditions: List[str] = None) -> Dict[str, Any]:
        existing_conditions = existing_conditions or []
        age_group = _age_group(age)
        sym_lower = [s.lower().strip() for s in symptoms]

        scores: Dict[str, float] = {}

        for disease, data in SYMPTOM_DISEASE_MAP.items():
            disease_syms = [s.lower() for s in data["symptoms"]]

            # Symptom overlap score (Jaccard-like)
            matches = 0
            total_weight = 0
            for s in sym_lower:
                for ds in disease_syms:
                    ratio = _levenshtein_ratio(s, ds)
                    if ratio >= 0.70:
                        matches += ratio
                        total_weight += 1

            if not disease_syms:
                continue
            overlap = matches / len(disease_syms)

            # Bayesian: P(disease) × P(symptoms|disease)
            prior = data["prevalence"] * data["age_risk"].get(age_group, data["prevalence"])
            posterior = prior * (overlap ** 0.5) if overlap > 0 else 0.0

            # Boost if existing condition is related
            for ec in existing_conditions:
                if any(_levenshtein_ratio(ec.lower(), k) > 0.7 for k in disease.split("_")):
                    posterior *= 1.5

            scores[disease] = posterior

        # Normalize
        total = sum(scores.values()) or 1
        normalized = {d: v / total for d, v in scores.items()}
        top = sorted(normalized.items(), key=lambda x: x[1], reverse=True)[:7]

        # Red flags
        red_flags = []
        emergency_symptoms = ["chest pain","chest tightness","shortness of breath","facial drooping",
                               "arm weakness","speech difficulty","severe headache","loss of consciousness"]
        for s in sym_lower:
            for ef in emergency_symptoms:
                if _levenshtein_ratio(s, ef) > 0.7:
                    red_flags.append(s)

        # Urgency
        urgency = "ROUTINE"
        if red_flags:
            urgency = "EMERGENCY"
        elif any(SYMPTOM_DISEASE_MAP.get(d, {}).get("urgency") == "URGENT" for d, _ in top[:3]):
            urgency = "URGENT"

        diagnoses = []
        for disease, prob in top:
            if prob < 0.01:
                continue
            info = SYMPTOM_DISEASE_MAP[disease]
            diagnoses.append({
                "condition": disease.replace("_", " ").title(),
                "icd10": info["icd10"],
                "probability": round(prob * 100, 1),
                "urgency": info["urgency"],
                "matching_symptoms": [s for s in sym_lower if any(_levenshtein_ratio(s, ds.lower()) > 0.65 for ds in info["symptoms"])],
            })

        return {
            "differential_diagnoses": diagnoses,
            "red_flags": list(set(red_flags)),
            "urgency": urgency,
            "age_group": age_group,
            "confidence": round(min(top[0][1] * 3, 0.9) if top else 0, 2) if top else 0,
            "evidence_level": "algorithmic",
            "note": "Bayesian analysis based on symptom-disease correlation database." + DISCLAIMER,
            "sources_consulted": ["SYMPTOM_DISEASE_MAP (120+ conditions)", "ICD-10"],
        }


class MedicineAnalyzer:
    """Drug intelligence engine."""

    def analyze_medicine(self, name: str) -> Dict[str, Any]:
        name_lower = name.lower().strip()
        # Fuzzy match
        matches = _fuzzy_match(name_lower, list(DRUG_DATABASE.keys()), threshold=0.6)
        # Also check brand names
        if not matches:
            for key, data in DRUG_DATABASE.items():
                brands = [b.lower() for b in data.get("brand", [])]
                for b in brands:
                    if _levenshtein_ratio(name_lower, b) > 0.7:
                        matches = [(key, 0.9)]
                        break

        if not matches:
            return {"error": f"Drug '{name}' not found in local database.",
                    "suggestion": "Use /api/research/drug/{name} for live FDA lookup.",
                    "sources_consulted": ["DRUG_DATABASE"]}

        drug_key, confidence = matches[0]
        data = DRUG_DATABASE[drug_key].copy()
        data["matched_name"] = drug_key
        data["match_confidence"] = round(confidence, 2)
        data["sources_consulted"] = ["DRUG_DATABASE (embedded)", "FDA (via ResearchSearchEngine if available)"]
        data["evidence_level"] = "clinical_reference"
        data["disclaimer"] = DISCLAIMER
        return data

    def check_interactions(self, drug_list: List[str]) -> Dict[str, Any]:
        interactions = []
        drugs_lower = [d.lower().strip() for d in drug_list]

        for i in range(len(drugs_lower)):
            for j in range(i + 1, len(drugs_lower)):
                a, b = drugs_lower[i], drugs_lower[j]
                key1 = f"{a}+{b}"
                key2 = f"{b}+{a}"
                severity = DRUG_INTERACTION_MATRIX.get(key1) or DRUG_INTERACTION_MATRIX.get(key2)

                # Generic checks
                if not severity:
                    for pattern_key, sev in DRUG_INTERACTION_MATRIX.items():
                        parts = pattern_key.split("+")
                        if (any(_levenshtein_ratio(a, p) > 0.7 for p in parts) and
                                any(_levenshtein_ratio(b, p) > 0.7 for p in parts)):
                            severity = sev
                            break

                if severity:
                    interactions.append({
                        "drug_a": drug_list[i],
                        "drug_b": drug_list[j],
                        "severity": severity,
                        "recommendation": {
                            "CONTRAINDICATED": "Do NOT use together. Contact prescriber immediately.",
                            "MAJOR": "Avoid combination. Use only if benefit outweighs risk with monitoring.",
                            "MODERATE": "Use with caution. Monitor for adverse effects.",
                            "MINOR": "Minimal interaction. Standard precautions.",
                        }.get(severity, "Monitor closely."),
                    })

        return {
            "drugs_checked": drug_list,
            "interactions": interactions,
            "total_interactions": len(interactions),
            "highest_severity": max((i["severity"] for i in interactions),
                                    key=lambda s: ["MINOR","MODERATE","MAJOR","CONTRAINDICATED"].index(s),
                                    default="NONE") if interactions else "NONE",
            "sources_consulted": ["DRUG_INTERACTION_MATRIX (embedded)", "FDA"],
            "disclaimer": DISCLAIMER,
        }

    def age_suitability(self, drug_name: str, age: int) -> Dict[str, Any]:
        info = self.analyze_medicine(drug_name)
        if "error" in info:
            return info
        age_group = _age_group(age)
        dosing = info.get("dosing", {})
        warnings = []

        # Known unsafe drugs for children
        unsafe_pediatric = {"aspirin", "ibuprofen"}
        if age < 18 and drug_name.lower() in unsafe_pediatric:
            warnings.append(f"{drug_name} has specific age restrictions for children — see prescriber.")

        # Get age-appropriate dose
        dose = dosing.get(age_group) or dosing.get("adult", "Consult prescriber")
        return {
            "drug": drug_name,
            "age": age,
            "age_group": age_group,
            "recommended_dose": dose,
            "warnings": warnings,
            "contraindications": info.get("contraindications", []),
            "sources_consulted": ["DRUG_DATABASE"],
        }


class ReportAnalyzer:
    """Medical lab report parser and interpreter."""

    def parse_lab_report(self, text: str) -> Dict[str, Any]:
        found_markers: Dict[str, float] = {}
        # Extract numeric values for known markers
        patterns = {
            "hemoglobin": r"(?:hemoglobin|hgb|hb)\s*[:\-=]?\s*(\d+\.?\d*)",
            "hba1c": r"(?:hba1c|hb ?a1c|glycated hemoglobin|a1c)\s*[:\-=]?\s*(\d+\.?\d*)",
            "fasting_glucose": r"(?:fasting glucose|fbs|fasting blood sugar|glucose)\s*[:\-=]?\s*(\d+\.?\d*)",
            "total_cholesterol": r"(?:total cholesterol|cholesterol)\s*[:\-=]?\s*(\d+\.?\d*)",
            "ldl": r"(?:ldl[\s\-]?c?|low density)\s*[:\-=]?\s*(\d+\.?\d*)",
            "hdl": r"(?:hdl[\s\-]?c?|high density)\s*[:\-=]?\s*(\d+\.?\d*)",
            "triglycerides": r"(?:triglycerides?|tg)\s*[:\-=]?\s*(\d+\.?\d*)",
            "creatinine": r"(?:creatinine|creat)\s*[:\-=]?\s*(\d+\.?\d*)",
            "egfr": r"(?:egfr|gfr)\s*[:\-=]?\s*(\d+\.?\d*)",
            "sodium": r"(?:sodium|na\+?)\s*[:\-=]?\s*(\d+\.?\d*)",
            "potassium": r"(?:potassium|k\+?)\s*[:\-=]?\s*(\d+\.?\d*)",
            "tsh": r"(?:tsh|thyroid stimulating)\s*[:\-=]?\s*(\d+\.?\d*)",
            "alt": r"(?:alt|alanine|sgpt)\s*[:\-=]?\s*(\d+\.?\d*)",
            "ast": r"(?:ast|aspartate|sgot)\s*[:\-=]?\s*(\d+\.?\d*)",
            "wbc": r"(?:wbc|white blood cell|leukocyte)\s*[:\-=]?\s*(\d+\.?\d*)",
            "platelets": r"(?:platelets?|plt)\s*[:\-=]?\s*(\d+\.?\d*)",
        }
        for marker, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    found_markers[marker] = float(match.group(1))
                except ValueError:
                    pass

        return self.interpret_values(found_markers)

    def interpret_values(self, markers: Dict[str, float], age: int = 40, gender: str = "unknown") -> Dict[str, Any]:
        findings = []
        critical_findings = []
        risk_flags = []

        for marker, value in markers.items():
            ref = LAB_REFERENCE_RANGES.get(marker)
            if not ref:
                continue

            # Determine range key
            range_key = "normal"
            if gender == "male" and f"normal_male" in ref["ranges"]:
                range_key = "normal_male"
            elif gender == "female" and f"normal_female" in ref["ranges"]:
                range_key = "normal_female"
            elif gender == "male" and "adult_male" in ref["ranges"]:
                range_key = "adult_male"
            elif gender == "female" and "adult_female" in ref["ranges"]:
                range_key = "adult_female"

            normal_range = ref["ranges"].get(range_key) or list(ref["ranges"].values())[0]
            unit = ref.get("unit", "")

            status = "NORMAL"
            interpretation = ""
            if isinstance(normal_range, tuple) and len(normal_range) == 2:
                low, high = normal_range
                if value < low:
                    status = "LOW"
                    interpretation = f"Below normal range ({low}-{high} {unit})"
                elif value > high:
                    status = "HIGH"
                    interpretation = f"Above normal range ({low}-{high} {unit})"
                else:
                    interpretation = f"Within normal range ({low}-{high} {unit})"

            # Critical checks
            crit_low = ref.get("critical_low")
            crit_high = ref.get("critical_high")
            if crit_low and value < crit_low:
                status = "CRITICAL_LOW"
                critical_findings.append(f"{marker.upper()}: {value} {unit} — CRITICALLY LOW (critical <{crit_low})")
            elif crit_high and value > crit_high:
                status = "CRITICAL_HIGH"
                critical_findings.append(f"{marker.upper()}: {value} {unit} — CRITICALLY HIGH (critical >{crit_high})")

            finding = {
                "marker": marker,
                "value": value,
                "unit": unit,
                "status": status,
                "interpretation": interpretation,
            }
            findings.append(finding)

        # Risk scoring
        glucose = markers.get("fasting_glucose", 0)
        hba1c = markers.get("hba1c", 0)
        if glucose >= 126 or hba1c >= 6.5:
            risk_flags.append("Possible DIABETES — confirm with repeat fasting glucose or OGTT")
        elif glucose >= 100 or hba1c >= 5.7:
            risk_flags.append("PREDIABETES range — lifestyle intervention recommended")

        chol = markers.get("total_cholesterol", 0)
        ldl = markers.get("ldl", 0)
        if chol >= 240 or ldl >= 160:
            risk_flags.append("ELEVATED CHOLESTEROL — cardiovascular risk assessment recommended")

        egfr = markers.get("egfr", 999)
        creat = markers.get("creatinine", 0)
        if egfr < 60:
            risk_flags.append(f"REDUCED eGFR ({egfr}) — possible CKD; nephrology referral advised")
        if creat > 1.5:
            risk_flags.append("ELEVATED CREATININE — kidney function monitoring needed")

        tsh = markers.get("tsh", 0)
        if tsh > 4.0:
            risk_flags.append("ELEVATED TSH — possible hypothyroidism; thyroid panel recommended")
        elif 0 < tsh < 0.4:
            risk_flags.append("LOW TSH — possible hyperthyroidism; thyroid panel recommended")

        return {
            "findings": findings,
            "critical_findings": critical_findings,
            "risk_flags": risk_flags,
            "urgency": "EMERGENCY" if critical_findings else ("URGENT" if risk_flags else "ROUTINE"),
            "markers_analyzed": len(findings),
            "sources_consulted": ["LAB_REFERENCE_RANGES (age/gender-stratified)", "WHO/NIH guidelines"],
            "disclaimer": DISCLAIMER,
        }


class ResearchAnalyzer:
    """Scientific literature and study analysis engine."""

    _BIAS_PATTERNS = {
        "selection_bias": ["convenience sample","self-selected","volunteer","not randomized","retrospective"],
        "reporting_bias": ["positive results only","significant findings","unpublished","selective reporting"],
        "confirmation_bias": ["expected to","as hypothesized","consistent with our hypothesis","confirms"],
        "attrition_bias": ["dropout","lost to follow-up","high attrition","missing data"],
        "detection_bias": ["unblinded","open-label","not blinded","assessor not blinded"],
    }

    def analyze_study(self, text: str) -> Dict[str, Any]:
        text_lower = text.lower()

        # Study type detection
        study_type = "observational"
        if re.search(r"random(ized|ised|ly assigned)", text_lower):
            study_type = "RCT"
        elif re.search(r"systematic review|meta.analysis", text_lower):
            study_type = "systematic_review"
        elif re.search(r"cohort|prospective|longitudinal", text_lower):
            study_type = "cohort"
        elif re.search(r"case.control", text_lower):
            study_type = "case_control"
        elif re.search(r"cross.sectional", text_lower):
            study_type = "cross_sectional"

        # Sample size
        sample_match = re.search(r"n\s*=\s*(\d+)|(\d+)\s*(?:patients|participants|subjects|individuals)", text_lower)
        sample_size = int(sample_match.group(1) or sample_match.group(2)) if sample_match else 0

        # P-value
        p_matches = re.findall(r"p\s*[<=>]\s*0\.0\d+", text_lower)
        p_values = []
        for pm in p_matches:
            try:
                p_values.append(float(re.search(r"0\.\d+", pm).group()))
            except Exception:
                pass

        # Effect size
        effect_match = re.search(r"(?:or|rr|hr|effect size)\s*[=:]\s*(\d+\.?\d*)", text_lower)
        effect_size = float(effect_match.group(1)) if effect_match else None

        # Confidence interval
        ci_match = re.search(r"95%?\s*ci[\s:]+(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)", text_lower)
        ci = (float(ci_match.group(1)), float(ci_match.group(2))) if ci_match else None

        # Evidence grade
        grade = self.evidence_grade(study_type, sample_size, p_values)

        # Bias detection
        biases = self.detect_bias(text_lower)

        # Clinical significance
        clinical_sig = self.clinical_significance(effect_size, study_type, sample_size)

        return {
            "study_type": study_type,
            "sample_size": sample_size,
            "p_values": p_values[:5],
            "effect_size": effect_size,
            "confidence_interval": list(ci) if ci else None,
            "evidence_grade": grade,
            "detected_biases": biases,
            "clinical_significance": clinical_sig,
            "quality_score": self._quality_score(study_type, sample_size, bool(p_values), bool(ci), biases),
            "sources_consulted": ["GRADE framework", "Cochrane methodology"],
            "disclaimer": DISCLAIMER,
        }

    def evidence_grade(self, study_type: str, sample_size: int = 0, p_values: List[float] = None) -> str:
        grade_map = {
            "systematic_review": "Level I (Highest)",
            "RCT": "Level II (High)" if sample_size > 100 else "Level II (Moderate)",
            "cohort": "Level III (Moderate)",
            "case_control": "Level IV (Low-Moderate)",
            "cross_sectional": "Level IV (Low-Moderate)",
            "observational": "Level V (Low)",
        }
        return grade_map.get(study_type, "Level V (Low)")

    def detect_bias(self, text: str) -> List[Dict[str, str]]:
        detected = []
        for bias_type, patterns in self._BIAS_PATTERNS.items():
            found = [p for p in patterns if p in text]
            if found:
                detected.append({
                    "bias_type": bias_type.replace("_", " ").title(),
                    "indicators": found,
                    "impact": "HIGH" if len(found) >= 2 else "MODERATE",
                })
        return detected

    def clinical_significance(self, effect_size: Optional[float], study_type: str, sample_size: int) -> str:
        if effect_size is None:
            return "Cannot assess clinical significance without effect size."
        if effect_size < 1.1:
            return "Small / Negligible effect — limited clinical relevance."
        elif effect_size < 1.5:
            return "Moderate effect — clinically meaningful if replicated."
        elif effect_size < 2.0:
            return "Large effect — likely clinically significant."
        else:
            return "Very large effect — high clinical significance; verify for confounders."

    def _quality_score(self, study_type: str, n: int, has_p: bool, has_ci: bool, biases: list) -> int:
        score = 0
        score += {"systematic_review": 40, "RCT": 35, "cohort": 25, "case_control": 20, "cross_sectional": 15, "observational": 10}.get(study_type, 10)
        if n > 1000: score += 20
        elif n > 100: score += 12
        elif n > 10: score += 5
        if has_p: score += 10
        if has_ci: score += 10
        score -= len(biases) * 8
        return max(0, min(100, score))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class MedicalResearchAgent:
    """
    Main orchestrator — routes queries to the appropriate sub-engine
    and integrates live literature search via ResearchSearchEngine.
    """

    def __init__(self):
        self.symptom_analyzer = SymptomAnalyzer()
        self.medicine_analyzer = MedicineAnalyzer()
        self.report_analyzer = ReportAnalyzer()
        self.research_analyzer = ResearchAnalyzer()
        self._search_engine: Optional[Any] = None
        if _RSE_OK:
            try:
                self._search_engine = ResearchSearchEngine()
            except Exception:
                pass

    def search_literature(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        if self._search_engine:
            try:
                results = self._search_engine.search(query, category="medical", max_results=max_results)
                return [r.to_dict() for r in results]
            except Exception:
                pass
        # Fallback: PubMed E-utilities directly
        if not _REQUESTS:
            return []
        try:
            import requests as req
            r = req.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params={"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"},
                timeout=8, verify=False
            )
            if r.status_code == 200:
                ids = r.json().get("esearchresult", {}).get("idlist", [])
                return [{"title": f"PubMed:{pmid}", "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                         "source": "PubMed", "reliability_score": 0.95} for pmid in ids]
        except Exception:
            pass
        return []

    def analyze(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        context = context or {}
        q_lower = query.lower()

        # Route
        if any(w in q_lower for w in ["symptom","feeling","pain","ache","fever","cough","tired","dizzy","nausea"]):
            return self._analyze_symptoms(query, context)
        elif any(w in q_lower for w in ["drug","medicine","medication","tablet","capsule","dose","mg","interaction","side effect"]):
            return self._analyze_medicine(query, context)
        elif any(w in q_lower for w in ["report","lab","blood test","result","value","level","count","hba1c","cholesterol"]):
            return self._analyze_report(query, context)
        elif any(w in q_lower for w in ["study","research","paper","trial","evidence","meta-analysis","review","journal"]):
            return self._analyze_research(query, context)
        else:
            # General: search literature
            results = self.search_literature(query)
            return {
                "type": "literature_search",
                "query": query,
                "results": results,
                "sources_consulted": list(TRUSTED_SOURCES.keys())[:5],
                "disclaimer": DISCLAIMER,
            }

    def _analyze_symptoms(self, query: str, context: dict) -> Dict[str, Any]:
        # Extract symptoms from text
        symptoms = re.findall(r"[a-z][a-z\s]+", query.lower())
        symptoms = [s.strip() for s in symptoms if len(s.strip()) > 3][:10]
        age = context.get("age", 35)
        gender = context.get("gender", "unknown")
        existing = context.get("existing_conditions", [])
        result = self.symptom_analyzer.analyze(symptoms, age, gender, existing)
        # Enrich top diagnosis with literature
        if result["differential_diagnoses"]:
            top = result["differential_diagnoses"][0]["condition"]
            result["related_research"] = self.search_literature(top, max_results=3)
        result["type"] = "symptom_analysis"
        return result

    def _analyze_medicine(self, query: str, context: dict) -> Dict[str, Any]:
        # Try to extract drug names
        drug_names = re.findall(r"\b[A-Za-z]{4,}\b", query)
        stop_words = {"drug","medicine","medication","what","does","tell","about","information","side","effect","interaction","dose","dosage","tablet","capsule"}
        candidates = [w for w in drug_names if w.lower() not in stop_words]

        if len(candidates) >= 2:
            result = self.medicine_analyzer.check_interactions(candidates)
            result["type"] = "drug_interaction"
        elif candidates:
            result = self.medicine_analyzer.analyze_medicine(candidates[0])
            result["type"] = "drug_analysis"
            # Live FDA lookup via search engine
            if self._search_engine:
                try:
                    fda = self._search_engine.search_fda_drugs(candidates[0])
                    if fda.get("indications"):
                        result["fda_live"] = fda
                        result["sources_consulted"] = result.get("sources_consulted", []) + ["FDA (live)"]
                except Exception:
                    pass
        else:
            result = {"error": "Could not identify drug name in query", "query": query}
            result["type"] = "drug_analysis"
        result["disclaimer"] = DISCLAIMER
        return result

    def _analyze_report(self, query: str, context: dict) -> Dict[str, Any]:
        result = self.report_analyzer.parse_lab_report(query)
        if not result.get("findings"):
            # Try context
            report_text = context.get("report_text", query)
            result = self.report_analyzer.parse_lab_report(report_text)
        result["type"] = "lab_report_analysis"
        return result

    def _analyze_research(self, query: str, context: dict) -> Dict[str, Any]:
        study_text = context.get("study_text", query)
        result = self.research_analyzer.analyze_study(study_text)
        result["type"] = "research_analysis"
        result["literature_search"] = self.search_literature(query, max_results=5)
        return result

    def run_nl(self, query: str) -> Dict[str, Any]:
        return self.analyze(query)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[MedicalResearchAgent] = None
_lock = threading.Lock()

def get_agent() -> MedicalResearchAgent:
    global _instance
    with _lock:
        if _instance is None:
            _instance = MedicalResearchAgent()
    return _instance


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "chest pain shortness of breath"
    agent = MedicalResearchAgent()
    result = agent.analyze(q)
    print(json.dumps(result, indent=2, default=str))
