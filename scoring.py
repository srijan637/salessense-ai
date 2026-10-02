"""
SalesSense AI - rule-based scoring engine.

The score is calculated ONLY here, with fixed business rules.
Gemini never changes it. This keeps every score explainable:
each point can be traced back to one rule in the table below.
"""

import pandas as pd

# ---------------------------------------------------------------
# Allowed values for each input field
# ---------------------------------------------------------------
COMPANY_SIZES = ["1-50", "51-200", "201-1000", "1000+"]
URGENCY_LEVELS = ["High", "Medium", "Low"]
INTERACTIONS = ["Positive", "Neutral", "No prior contact", "Negative"]
YES_NO = ["Yes", "No"]

REQUIRED_COLUMNS = [
    "Company", "Industry", "Company_Size", "Decision_Maker", "Demo_Requested",
    "Purchase_Urgency", "Website_Visits", "Email_Opens", "Current_Vendor",
    "Previous_Interaction",
]
OPTIONAL_COLUMNS = ["Lead_ID", "Sales_Notes"]

# ---------------------------------------------------------------
# Scoring rules (total = 100). Shown to users in the app as-is.
# ---------------------------------------------------------------
SCORING_RULES = [
    ("Demo requested", 20, "Yes = 20, No = 0"),
    ("Purchase urgency", 15, "High = 15, Medium = 8, Low = 0"),
    ("Decision maker involved", 15, "Yes = 15, No = 0"),
    ("Company size (employees)", 10, "1000+ = 10, 201-1000 = 6, 51-200 = 3, 1-50 = 0"),
    ("Website visits (last 30 days)", 10, "10 or more = 10, 5-9 = 5, under 5 = 0"),
    ("Email opens (last 30 days)", 10, "8 or more = 10, 4-7 = 5, under 4 = 0"),
    ("Current solution", 10, "Uses a competitor = 10, In-house/manual = 7, None identified = 0"),
    ("Previous interaction", 10, "Positive = 10, Neutral = 5, No prior contact = 3, Negative = 0"),
]

TIERS = [(80, "HOT"), (50, "WARM"), (0, "COLD")]

# Default next step per tier. Used in the export and whenever AI is unavailable.
DEFAULT_ACTIONS = {
    "HOT": "Contact within 24 hours and schedule a discovery call or demo.",
    "WARM": "Nurture: share a relevant case study and follow up within 7 days.",
    "COLD": "Add to a nurture email sequence and re-score in 30 days.",
}


def classify(score):
    for threshold, tier in TIERS:
        if score >= threshold:
            return tier
    return "COLD"


def vendor_type(vendor):
    """Turns the free-text Current_Vendor value into one of three types."""
    v = str(vendor).strip().lower()
    if v in ("", "none", "nan", "no", "none identified", "-"):
        return "none"
    if any(word in v for word in ("in-house", "inhouse", "manual", "internal", "excel", "spreadsheet")):
        return "in-house"
    return "competitor"


def score_lead(lead):
    """
    Returns (score, tier, breakdown).
    breakdown is a list of dicts: factor, points, max_points, reason.
    """
    breakdown = []

    def add(factor, points, max_points, reason):
        breakdown.append({"Factor": factor, "Points": points,
                          "Max": max_points, "Reason": reason})

    # 1. Demo requested
    if lead["Demo_Requested"] == "Yes":
        add("Demo requested", 20, 20, "Prospect has requested a demo")
    else:
        add("Demo requested", 0, 20, "No demo requested yet")

    # 2. Purchase urgency
    urgency_points = {"High": 15, "Medium": 8, "Low": 0}[lead["Purchase_Urgency"]]
    add("Purchase urgency", urgency_points, 15,
        f"{lead['Purchase_Urgency']} purchase urgency")

    # 3. Decision maker
    if lead["Decision_Maker"] == "Yes":
        add("Decision maker", 15, 15, "A decision maker is involved")
    else:
        add("Decision maker", 0, 15, "No decision maker involved yet")

    # 4. Company size
    size_points = {"1000+": 10, "201-1000": 6, "51-200": 3, "1-50": 0}[lead["Company_Size"]]
    add("Company size", size_points, 10, f"{lead['Company_Size']} employees")

    # 5. Website visits
    visits = int(lead["Website_Visits"])
    visit_points = 10 if visits >= 10 else 5 if visits >= 5 else 0
    add("Website engagement", visit_points, 10, f"{visits} website visits in 30 days")

    # 6. Email opens
    opens = int(lead["Email_Opens"])
    open_points = 10 if opens >= 8 else 5 if opens >= 4 else 0
    add("Email engagement", open_points, 10, f"{opens} email opens in 30 days")

    # 7. Current solution
    vtype = vendor_type(lead["Current_Vendor"])
    if vtype == "competitor":
        add("Current solution", 10, 10,
            f"Uses {lead['Current_Vendor']} (need established, displacement opportunity)")
    elif vtype == "in-house":
        add("Current solution", 7, 10, "Uses an in-house/manual process: need exists, no vendor lock-in")
    else:
        add("Current solution", 0, 10, "No current solution identified: need not yet established")

    # 8. Previous interaction
    interaction_points = {"Positive": 10, "Neutral": 5,
                          "No prior contact": 3, "Negative": 0}[lead["Previous_Interaction"]]
    add("Previous interaction", interaction_points, 10,
        f"Previous interaction: {lead['Previous_Interaction']}")

    score = sum(item["Points"] for item in breakdown)
    return score, classify(score), breakdown


def top_reasons(breakdown, n=4):
    """The factors that earned points, strongest first (for the export)."""
    earned = [b for b in breakdown if b["Points"] > 0]
    earned.sort(key=lambda b: b["Points"], reverse=True)
    return "; ".join(b["Reason"] for b in earned[:n]) or "No positive signals"


# ---------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------
def _yes_no(value):
    v = str(value).strip().lower()
    if v in ("yes", "y", "true", "1"):
        return "Yes"
    if v in ("no", "n", "false", "0"):
        return "No"
    return None


def _match(value, allowed):
    """Case-insensitive match against a list of allowed values."""
    v = str(value).strip().lower()
    for option in allowed:
        if v == option.lower():
            return option
    return None


def _non_negative_int(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):  # blank cell or infinity
        return None
    if number < 0 or number != int(number):
        return None
    return int(number)


def validate_lead(raw):
    """
    Cleans one lead and checks every field.
    Returns (clean_lead, errors). If errors is non-empty, the lead is not scored.
    """
    errors = []
    lead = {}

    company = str(raw.get("Company", "") or "").strip()
    if company.lower() in ("", "nan"):
        company = ""
        errors.append("Please enter a company name.")
    lead["Company"] = company

    industry = str(raw.get("Industry", "") or "").strip()
    lead["Industry"] = "Not specified" if industry.lower() in ("", "nan") else industry

    lead["Company_Size"] = _match(raw.get("Company_Size"), COMPANY_SIZES)
    if lead["Company_Size"] is None:
        errors.append(f"Company size must be one of: {', '.join(COMPANY_SIZES)}.")

    for field, label in (("Decision_Maker", "Decision maker"), ("Demo_Requested", "Demo requested")):
        lead[field] = _yes_no(raw.get(field))
        if lead[field] is None:
            errors.append(f"{label} must be Yes or No.")

    lead["Purchase_Urgency"] = _match(raw.get("Purchase_Urgency"), URGENCY_LEVELS)
    if lead["Purchase_Urgency"] is None:
        errors.append("Purchase urgency must be High, Medium or Low.")

    for field, label in (("Website_Visits", "Website visits"), ("Email_Opens", "Email opens")):
        lead[field] = _non_negative_int(raw.get(field))
        if lead[field] is None:
            errors.append(f"Invalid input: {label} must be a whole number of 0 or more.")

    vendor = str(raw.get("Current_Vendor", "") or "").strip()
    lead["Current_Vendor"] = "None" if vendor.lower() in ("", "nan") else vendor

    lead["Previous_Interaction"] = _match(raw.get("Previous_Interaction"), INTERACTIONS)
    if lead["Previous_Interaction"] is None:
        errors.append(f"Previous interaction must be one of: {', '.join(INTERACTIONS)}.")

    notes = str(raw.get("Sales_Notes", "") or "").strip()
    lead["Sales_Notes"] = "" if notes.lower() == "nan" else notes[:1000]

    lead_id = str(raw.get("Lead_ID", "") or "").strip()
    lead["Lead_ID"] = "" if lead_id.lower() == "nan" else lead_id

    return lead, errors


def normalise_columns(df):
    """Accepts 'website visits', 'Website Visits', 'website_visits' etc."""
    lookup = {c.lower(): c for c in REQUIRED_COLUMNS + OPTIONAL_COLUMNS}
    renamed = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_")
        if key in lookup:
            renamed[col] = lookup[key]
    return df.rename(columns=renamed)


def score_dataframe(df):
    """
    Validates and scores every row of an uploaded file.
    Returns (scored_df, skipped) where skipped lists rows that failed validation.
    """
    df = normalise_columns(df)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("The file is missing these columns: " + ", ".join(missing))

    rows, skipped = [], []
    for i, raw in enumerate(df.to_dict("records"), start=2):  # row 1 is the header
        lead, errors = validate_lead(raw)
        if errors:
            skipped.append({"Row": i, "Company": lead["Company"] or "(blank)",
                            "Problem": " ".join(errors)})
            continue
        score, tier, breakdown = score_lead(lead)
        lead.update({"Score": score, "Tier": tier,
                     "Reasons": top_reasons(breakdown),
                     "Recommended_Action": DEFAULT_ACTIONS[tier]})
        rows.append(lead)

    scored = pd.DataFrame(rows)
    if not scored.empty:
        scored = scored.sort_values("Score", ascending=False).reset_index(drop=True)
    return scored, skipped
