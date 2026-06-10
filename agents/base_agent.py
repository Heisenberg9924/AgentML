
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from datetime import datetime
import logging
import time


class BaseAgent(ABC):
    """
    Base class for all agents in the
    Autonomous ML Scientist system.
    """

    def __init__(
        self,
        llm: Any,
        state: Optional[Dict[str, Any]] = None,
        name: str = "BaseAgent",
        version: str = "1.0"
    ):
        self.llm = llm
        self.state = state if state is not None else {}

        self.name = name
        self.version = version

        self.logger = logging.getLogger(self.name)
    
    def invoke(
        self,
        prompt: str
    ) -> str:
        """
        Wrapper around think() to handle LLM invocation.
        """

        return self.llm.invoke(prompt)

    def log(
        self,
        message: str,
        level: str = "info"
    ) -> None:

        level = level.lower()

        if level == "info":
            self.logger.info(message)

        elif level == "warning":
            self.logger.warning(message)

        elif level == "error":
            self.logger.error(message)

        else:
            self.logger.debug(message)

    def think(
        self,
        prompt: str
    ) -> str:
        """
        Send prompt to LLM and return response.
        """

        if self.llm is None:
            raise ValueError(
                f"{self.name} requires an LLM."
            )

        try:

            self.log("Generating LLM response...")

            if hasattr(self.llm, "invoke"):
                response = self.llm.invoke(prompt)

            elif callable(self.llm):
                response = self.llm(prompt)

            else:
                raise TypeError(
                    "Unsupported LLM interface."
                )

            return str(response)

        except Exception as e:

            self.log(
                f"LLM Error: {str(e)}",
                level="error"
            )

            raise

    def update_state(
        self,
        key: str,
        value: Any
    ) -> None:
        """
        Update shared workflow state.
        """

        self.state[key] = value

    def get_state(
        self,
        key: str,
        default: Any = None
    ) -> Any:
        """
        Retrieve value from shared state.
        """

        return self.state.get(
            key,
            default
        )

    def create_result(
        self,
        status: str,
        data: Dict[str, Any],
        execution_time: Optional[float] = None
    ) -> Dict[str, Any]:

        return {
            "agent": self.name,
            "version": self.version,
            "timestamp": datetime.utcnow().isoformat(),
            "status": status,
            "execution_time": execution_time,
            "data": data
        }

    def execute(
        self,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Wrapper around run() that
        tracks execution time and
        handles exceptions.
        """

        start_time = time.time()

        try:

            result = self.run(
                *args,
                **kwargs
            )

            elapsed = (
                time.time() - start_time
            )

            result["execution_time"] = round(
                elapsed,
                4
            )

            return result

        except Exception as e:

            elapsed = (
                time.time() - start_time
            )

            self.log(
                f"Execution failed: {str(e)}",
                level="error"
            )

            return self.create_result(
                status="failed",
                data={
                    "error": str(e)
                },
                execution_time=round(
                    elapsed,
                    4
                )
            )

    @abstractmethod
    def run(
        self,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Core agent logic.
        """
        pass