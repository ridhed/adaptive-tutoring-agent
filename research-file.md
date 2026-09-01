# 1. Technical Terms and What does it mean

| Term                                                     | What it means                                                                                                                                               |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Intelligent Tutoring System (ITS)**                    | The field of AI systems that provide personalized instruction.                                                                                              |
| **Student Modeling / Learner Modeling**                  | The process of inferring what a student knows from their behavior. This is your core task.                                                                  |
| **Bayesian Knowledge Tracing (BKT)**                     | A classic algorithm that models each concept as having a hidden binary state: `Learned` or `Not Learned`. It updates beliefs based on correctness evidence. |
| **Deep Knowledge Tracing (DKT)**                         | A neural-network version of BKT that uses a continuous hidden state instead of a binary one.                                                                |
| **Partially Observable Markov Decision Process (POMDP)** | The formal framework for decision-making when the true state (student knowledge) is hidden and must be inferred from observations.                          |
| **Hidden State / Latent State**                          | The true student knowledge that you cannot observe directly. You only see evidence (responses).                                                             |
| **Evidence / Observation**                               | The observable signals you use to infer the hidden state: correctness, response time, hint requests, etc.                                                   |
| **Action Space / Policy**                                | The set of actions your agent can choose from (`answer`, `ask`, `hint`, `teach`) and the rule for choosing among them.                                      |
| **Exploration-Exploitation Tradeoff**                    | The dilemma of choosing an action to gain information (exploration) versus choosing an action to maximize immediate learning (exploitation).                |
| **Cold-Start Problem**                                   | The challenge of making decisions when you have no prior data about a new student.                                                                          |
| **Diagnostic Assessment**                                | A short initial phase designed to reduce uncertainty about the student's knowledge before full instruction begins.                                          |
| **Computerized Adaptive Testing (CAT)**                  | A system that selects the next question based on the student's previous answers to efficiently estimate ability.                                            |
| **Item Response Theory (IRT)**                           | A statistical framework for modeling the probability of a correct response as a function of student ability and item difficulty.                            |
| **Zone of Proximal Development (ZPD)**                   | The pedagogical principle that learning is optimal when tasks are slightly above the student's current level.                                               |
| **Scaffolding**                                          | Providing support (hints, worked examples) that is gradually removed as the student masters the concept.                                                    |
| **Misconception Detection**                              | Identifying specific incorrect beliefs a student holds, as opposed to simply marking an answer wrong.                                                       |
# 2. Useful Search Queries

**For the core problem:**
- `Bayesian Knowledge Tracing tutorial`
- `POMDP tutoring policy student model`
- `intelligent tutoring system action selection`
- `cold start knowledge tracing diagnostic assessment`    

**For the hidden state:**
- `hidden Markov model student knowledge`
- `deep knowledge tracing explained`
- `learner modeling uncertainty quantification`

**For the action space:**
- `reinforcement learning tutoring policy`
- `exploration exploitation tradeoff education`
- `adaptive scaffolding intelligent tutoring system`

**For the evidence:**
- `student response time as evidence tutoring`
- `hint usage modeling intelligent tutor`
- `metacognitive calibration confidence tutoring`

**For practical building:**
- `open source intelligent tutoring system python`
- `pyBKT tutorial student modeling`
- `building adaptive learning agent from scratch`

# 3. Relevant Reddit Communities

| #   | Community                      | Members / Activity                                                                                    |
| --- | ------------------------------ | ----------------------------------------------------------------------------------------------------- |
| 1   | **r/MachineLearning**          | Large, general ML community. POMDP and RL questions are common here.                                  |
| 2   | **r/reinforcementlearning**    | Specialized for RL. Your action-selection problem is fundamentally an RL problem.                     |
| 3   | **r/learnmachinelearning**     | Beginner-friendly. Good for asking "how do I build this?" and getting implementation advice.          |
| 4   | **r/edtech**                   | Teachers and developers discussing education technology. Good for understanding real classroom needs. |
| 5   | **r/EducationalAI**            | Focused specifically on AI in education. Directly relevant to your domain.                            |
| 6   | **r/Teachers**                 | Practitioners who can tell you what actually works pedagogically versus what sounds good in theory.   |
| 7   | **r/MLQuestions**              | For specific, technical ML questions without the noise of larger subs.                                |
| 8   | **r/Scholar**                  | For requesting specific academic papers you cannot access.                                            |
| 9   | **r/Using\_AI\_in\_Education** | Practical discussions about deploying AI tools in learning environments.                              |
| 10  | **r/education**                | Broader education discourse. Useful for understanding student and teacher perspectives.               |
