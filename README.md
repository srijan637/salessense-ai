# SalesSense AI

AI-assisted B2B sales lead scoring. Leads get a transparent rule-based score (0-100) and a Hot/Warm/Cold tier. Gemini then explains the result, suggests a next action, and reads the salesperson's notes to flag risks the score can't see.

Built for the AI for Managers end-term project (Use Case 7: Sales Lead-Scoring Assistant). All lead data in `data/` is synthetic.

| File | What it does |
|---|---|
| `app.py` | The Streamlit interface: lead form, upload, dashboard, export |
| `scoring.py` | Validation and the 100-point scoring rules. The score is calculated only here. |
| `ai.py` | The Gemini prompt, guardrails and error handling |
| `data/sample_leads.csv` | 30 synthetic leads (8 Hot, 12 Warm, 10 Cold) |
| `data/sample_leads_with_errors.csv` | 5 rows, 3 of them invalid, for testing validation |
| `requirements.txt` | Python packages |

The Gemini API key is read from Streamlit secrets (`GEMINI_API_KEY`) and is never stored in the code. `GEMINI_MODEL` is optional and defaults to `gemini-3.5-flash-lite`.
