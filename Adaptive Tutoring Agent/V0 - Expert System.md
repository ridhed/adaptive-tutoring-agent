# 1. Dataset  Input
**Dataset:** [Assistment2009](https://doi.org/10.57760/sciencedb.j00133.00253 )
**Input Parameters:**
- `student_id`: String (Unique student identifier)  
- `kc_id`: String (Knowledge component/skill tag)
- `correct`: Boolean (0/1 ground truth)
- `time_seconds`: Float (Time from display to submission)
- `attempt_number`: Integer (1st, 2nd, or 3rd attempt on this problem)

**Discretization Rules:**
- `response_time`: FAST (< expected_time × 0.3), SLOW (≥ expected_time × 0.3) [2]
# 2. Hidden States

{`careless`, `misconception`, `guessing`, `knowledge_gap`, `mastery`} [1]
# 3. Actions

{`ASK`, `HINT`, `TEACH_PRIOR`, `ANSWER`}
# 4. Decision Table

|**is_correct**|**attempt_number**|**response_time**|**Hidden State**|**Action**|
|---|---|---|---|---|
|False|> 1|Any|misconception|ASK|
|False|1|FAST|careless|ASK|
|False|1|SLOW|knowledge_gap|TEACH_PRIOR|
|True|> 1|Any|guessing|ASK|
|True|1|Any|mastery|ANSWER|
# 5. Architecture: Expert System (if/elif policy)
![[V0.pdf]]
# 6. What's Next

- **V1 - Bayesian Knowledge Tracing:** Update beliefs with probabilities after each interaction.
# 7. References

[1] Corbett, A. T., & Anderson, J. R. (1995). _Knowledge Tracing: Modeling the Acquisition of Procedural Knowledge.

[2] Mislevy, R. J., Steinberg, L. S., & Almond, R. G. (2003). _On the Structure of Educational Assessments.