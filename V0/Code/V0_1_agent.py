class AgentV01:
    def predict(
        self,
        is_correct: bool,
        kc_success_history: int,
        response_time: str,
        attempt_number: int,
        hint_count: int,
        mistake_history: int,
    ) -> tuple[str, str]:
        if is_correct:
            return "mastery", "ANSWER"

        if attempt_number > 1 or hint_count > 0 or mistake_history >= 2:
            return "possible_conceptual_difficulty", "TEACH_PRIOR"

        if attempt_number == 1 and hint_count == 0 and kc_success_history >= 1:
            if response_time == "FAST":
                return "possible_carelessness", "ASK"
            return "possible_execution_error", "HINT"

        return "uncertain", "ASK"