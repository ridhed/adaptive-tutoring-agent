class AgentV0:
    def predict(self, is_correct: bool, attempt_number: int, response_time: str) -> tuple[str, str]:
        if not is_correct:
            if attempt_number > 1:
                return "misconception", "ASK"
            if response_time == "FAST":
                return "careless", "ASK"
            return "knowledge_gap", "TEACH_PRIOR"
        else:
            if attempt_number > 1:
                return "guessing", "ASK"
            return "mastery", "ANSWER"