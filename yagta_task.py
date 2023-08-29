# YAGTA - Yet another autonomous task agent - experiments in autonomous GPT agents that learn over time
# jack@latrobe.group

## yagta_task.py - Task class for YAGTA

# AgentTask - define class
class AgentTask:
    def __init__(self, task_description: str, task_objective: str, task_result: str = None) -> None:
        self.task_description = task_description
        self.task_objective = task_objective
        self.task_result = task_result

    def set_result(self, task_result: str) -> None:
        self.task_result = task_result

    def get_result(self) -> str:
        return str(self.task_result)
